"""
Load the three Stage2 evaluation score JSON files and print
the mean score of each metric, rounded to 2 decimal places.
"""

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path("AesFormer_ICML2026/test/Stage2/evaluate")

SCORE_FILES = {
    "ArtiMuse": BASE_DIR / "ArtiMuse_Score.json",
    "LAION-V2": BASE_DIR / "LAION-V2_Score.json",
    "Q-align": BASE_DIR / "Q-align_Score.json",
}


def load_scores(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Score file must be a JSON object: {path}")

    return data


def compute_mean(score_dict: dict[str, float], label: str) -> float:
    if not score_dict:
        raise ValueError(f"No scores found in {label}.")

    values: list[float] = []
    for key, value in score_dict.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Score for '{key}' in {label} is not numeric: {value}")
        values.append(float(value))

    return sum(values) / len(values)


def main() -> None:
    print("Final score summary:")
    for label, path in SCORE_FILES.items():
        scores = load_scores(path)
        mean_score = compute_mean(scores, label)
        print(f"{label} mean score: {mean_score:.2f}")


if __name__ == "__main__":
    main()
