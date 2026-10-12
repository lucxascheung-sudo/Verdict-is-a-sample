"""Aggregation with an agreement gate.

The point of this module is small enough to read in one sitting: a majority vote
is not a measurement unless the majority is large enough to be one.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

AGREE_MIN = 0.8


@dataclass(frozen=True)
class Verdict:
    """The outcome of aggregating repeated judge runs.

    `value is None` means UNCERTAIN: the judge did not agree with itself well
    enough to report anything. That is a result, not an error.
    """

    value: str | None
    fraction: float
    counts: dict[str, int]
    n: int

    @property
    def stable(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        if self.value is None:
            spread = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
            return f"uncertain ({self.fraction:.0%}, n={self.n}: {spread})"
        return f"{self.value} ({self.fraction:.0%}, n={self.n})"


def aggregate(verdicts: Sequence[str], agree_min: float = AGREE_MIN) -> Verdict:
    """Majority vote, reported only if the majority reaches `agree_min`.

    At n=3 only 3:0 passes. At n=5, 5:0 and 4:1 pass; 3:2 does not.
    """
    if not verdicts:
        raise ValueError("aggregate() needs at least one verdict")
    counts = dict(Counter(verdicts))
    top, n_top = Counter(verdicts).most_common(1)[0]
    fraction = n_top / len(verdicts)
    if fraction < agree_min:
        return Verdict(None, fraction, counts, len(verdicts))
    return Verdict(top, fraction, counts, len(verdicts))


def gate_span(agree_min: float = AGREE_MIN) -> str:
    """Human-readable description of what the gate accepts at common n."""
    out = []
    for n in (3, 5, 7):
        passing = [k for k in range(n, 0, -1) if k / n >= agree_min]
        out.append(f"n={n}: accepts {', '.join(f'{k}:{n - k}' for k in passing)}")
    return " | ".join(out)


if __name__ == "__main__":
    print(gate_span())
    for trial in (["yes", "yes", "yes"], ["yes", "partial", "partial"],
                  ["yes", "partial", "yes"], ["partial", "no", "no"]):
        print(f"  {trial} -> {aggregate(trial)}")
