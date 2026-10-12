"""The expensive tier: a model-as-judge.

The judge is pluggable on purpose. The finding in the README is not about
Ollama, or about any particular model — it is about treating a sampled verdict
as a measurement. Swap in whatever you actually use.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

COMPARE_QUESTION = (
    "You are shown two images. Is the main subject in the second image the same "
    "subject as in the first? Answer with exactly one word: yes, partial, or no."
)
ATTRIBUTE_QUESTION = (
    "Is this a photograph or an illustration? "
    "Answer with exactly one word: photograph, illustration, or unclear."
)

COMPARE_LABELS = ("yes", "partial", "no")
ATTRIBUTE_LABELS = ("photograph", "illustration", "unclear")


class Judge(Protocol):
    """Anything with this signature can be judged against the gate.

    Takes a **list** of images so the same interface covers both a single-image
    attribute question and a reference-versus-output comparison.

    Returns the model's **raw** text. Parsing a free-form answer down to a label
    is the caller's job, because only the caller knows which labels are legal for
    the question it asked. (Parsing inside the judge silently hard-codes one
    label set, which is a bug this repository had.)
    """

    def __call__(self, images: Sequence[str], question: str) -> str: ...


def _encode(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def parse_label(text: str, labels: tuple[str, ...]) -> str:
    """Pull one of `labels` out of a free-form answer.

    Returns "other" when the model says none of them. Keeping "other" as a
    distinct outcome matters: a judge that fails to answer is not a judge that
    answered negatively.
    """
    low = text.lower()
    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", low):
            return label
    return "other"


class OllamaJudge:
    """Reference implementation against a local Ollama vision model.

    `temperature` is deliberately above zero. Setting it to zero (or fixing a
    seed) makes the judge deterministic — which hides the phenomenon rather
    than removing it. See README, "Fixing the seed is not a fix".
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        host: str = "http://localhost:11434",
        temperature: float = 0.8,
        timeout: int = 300,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.seed = seed

    def __call__(self, images: Sequence[str], question: str) -> str:
        options: dict[str, object] = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": question,
                "images": [_encode(p) for p in images],
                "stream": False,
                "options": options,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"Ollama answered HTTP {exc.code} for model {self.model!r}.\n"
                f"The server is up, but that model is probably not pulled:\n"
                f"    ollama pull {self.model}\n"
                f"To see the aggregation logic without a model: python demo.py --mock"
            ) from exc
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Could not reach Ollama at {self.host} ({exc}).\n"
                f"Is `ollama serve` running?\n"
                f"To see the aggregation logic without a model: python demo.py --mock"
            ) from exc

        return body.get("response", "").strip()


class MockJudge:
    """Scripted verdicts, for exercising the gate without a model.

    This is a logic demo. It produces no evidence about any real judge.
    """

    def __init__(self, script: list[str] | None = None) -> None:
        self.script = script or ["yes", "partial", "partial", "yes", "partial"]
        self.calls = 0

    def __call__(self, images: Sequence[str], question: str) -> str:
        value = self.script[self.calls % len(self.script)]
        self.calls += 1
        return value


if __name__ == "__main__":
    for answer in ("yes", "Yes.", "partial", "no", "I cannot tell."):
        print(f"  {answer!r:18s} -> {parse_label(answer, COMPARE_LABELS)}")
