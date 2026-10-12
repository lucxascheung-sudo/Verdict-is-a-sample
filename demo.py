"""Reproduce the headline finding.

    python demo.py OUTPUT.png --reference REFERENCE.png    # comparison (the real setting)
    python demo.py IMAGE.png                               # single-image attribute question
    python demo.py --mock                                  # aggregation logic only, no model

What it does
------------
1. Copies the judged image to a second file, so the two are byte-identical.
2. Proves that with MAD (should print 0.00).
3. Asks the same question about BOTH copies, `--repeat` times each.
4. Prints the per-run verdicts and the gated aggregate for each copy.

What you should see
-------------------
The two copies are the same bytes, so any difference between their verdicts is
produced by the judge. A 2:1 or 3:2 split is a coin flip wearing a number.

**And you may see nothing at all.** If every run agrees, the tool says so
instead of pretending otherwise — that means your item is nowhere near this
judge's decision boundary. Instability is a boundary phenomenon, not a uniform
noise floor: easy items are stable, hard ones are not. The recorded headline
result came from a fine-grained *identity* comparison, which is exactly the kind
of question a small vision model is bad at. Bring your own hard item; the tool
will tell you whether you found one.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from judge import (
    ATTRIBUTE_LABELS,
    ATTRIBUTE_QUESTION,
    COMPARE_LABELS,
    COMPARE_QUESTION,
    MockJudge,
    OllamaJudge,
    parse_label,
)
from verdict import AGREE_MIN, gate_span, aggregate


@dataclass(frozen=True)
class Task:
    """A question plus the only labels that count as answers to it."""

    key: str
    text: str
    labels: tuple[str, ...]


def build_task(reference: str | None) -> Task:
    if reference:
        return Task("same-subject", COMPARE_QUESTION, COMPARE_LABELS)
    return Task("attribute", ATTRIBUTE_QUESTION, ATTRIBUTE_LABELS)


def ask(judge, images: list[str], task: Task, repeat: int) -> list[str]:
    """Ask `task` about `images`, `repeat` times, parsed to `task.labels`."""
    return [parse_label(judge(images, task.text), task.labels) for _ in range(repeat)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", help="the image being judged (the 'output')")
    parser.add_argument("--reference", default=None,
                        help="compare against this image instead of asking a single-image question")
    parser.add_argument("--repeat", type=int, default=5,
                        help="judge runs per copy (default 5; the gate needs >=3 to mean anything)")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama vision model")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="keep this above 0; a deterministic judge hides the phenomenon")
    parser.add_argument("--seed", type=int, default=None,
                        help="fixing the seed freezes ONE sample; it does not remove the problem")
    parser.add_argument("--mock", action="store_true",
                        help="run the gate on scripted verdicts (logic demo, NOT evidence)")
    args = parser.parse_args()

    print(f"agreement gate: majority must reach {AGREE_MIN:.0%}")
    print(f"  {gate_span()}")
    print()

    if args.mock:
        return run_mock(args)

    if not args.image:
        parser.error("an image path is required unless --mock is used")
    source = Path(args.image)
    if not source.is_file():
        parser.error(f"no such file: {source}")
    if args.reference and not Path(args.reference).is_file():
        parser.error(f"no such reference file: {args.reference}")

    # imported here so that --mock runs with no third-party dependencies at all
    from mad import mad

    task = build_task(args.reference)
    print(f"task: {task.key}")
    print(f"  {task.text}")
    print(f"  legal answers: {', '.join(task.labels)}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        copy_b = Path(tmp) / source.name
        shutil.copyfile(source, copy_b)

        distance = mad(str(source), str(copy_b))
        print(f"copy A: {source}")
        print(f"copy B: {copy_b}")
        print(f"MAD(A, B) = {distance:.2f}   <- 0.00 means byte-identical, as expected")
        if distance != 0.0:
            print("  NOT 0.00: the copies differ, so this run proves nothing about the judge.")
        print()

        judge = OllamaJudge(model=args.model, temperature=args.temperature, seed=args.seed)
        results = {}
        for name, path in (("A", str(source)), ("B", str(copy_b))):
            images = [args.reference, path] if args.reference else [path]
            raw = ask(judge, images, task, args.repeat)
            results[name] = aggregate(raw)
            print(f"copy {name}: {raw}")
            print(f"  -> {results[name]}")
            print()

    return summarise(results, args.repeat, args.model)


def summarise(results: dict, repeat: int, model: str) -> int:
    a, b = results["A"], results["B"]
    print("=" * 70)
    print(f"copy A reported: {a}")
    print(f"copy B reported: {b}")
    print("=" * 70)
    print()

    if a.value is None or b.value is None:
        print("At least one copy FAILED THE GATE. The judge could not agree with itself")
        print("well enough to report anything. That is what the gate is for: without it,")
        print("the majority label would have been printed as if it were a result.")
        print("'uncertain' is a finding, not a failure.")
    elif a.value != b.value:
        print("Two BYTE-IDENTICAL images produced DIFFERENT verdicts.")
        print("The verdict is a sample, not a measurement.")
    elif a.fraction < 1.0 or b.fraction < 1.0:
        print("Both copies happened to agree, but neither was unanimous. The underlying")
        print("distribution is not deterministic, so rerun — the aggregate can flip.")
    else:
        print("Every run agreed on both copies, so this item is nowhere near the judge's")
        print("decision boundary. That is a real result about the item, not a failure of")
        print("the tool: instability is a boundary phenomenon. Try a fine-grained question")
        print("(identity, exact counts, small attribute differences) or a weaker model.")

    print()
    print(f"n = {repeat} per copy, model = {model}. That is small.")
    print("Nothing here is a rate estimate; it is a demonstration of a failure mode.")
    return 0


def run_mock(args: argparse.Namespace) -> int:
    print("MOCK MODE — scripted verdicts. This demonstrates the gate's logic only")
    print("and says nothing about any real judge.")
    print()
    judge = MockJudge()
    for label in ("A", "B"):
        raw = [judge([], "mock") for _ in range(args.repeat)]
        print(f"copy {label}: {raw}")
        print(f"  -> {aggregate(raw)}")
    print()
    print("Note that 3:2 never passes the gate, at any n.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
