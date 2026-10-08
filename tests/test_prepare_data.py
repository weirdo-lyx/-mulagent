from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_data import (
    Config,
    DataValidationError,
    PreparationStats,
    allocate_counts,
    deduplicate_candidates,
    load_config,
    prepare,
    validate_claim_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def annotation(*facts: tuple[str, str]) -> dict[str, object]:
    return {
        "text": "sentence",
        "is-relevant": True,
        "human-atomic-facts": [{"text": text, "label": label} for text, label in facts],
    }


class PrepareDataUnitTests(unittest.TestCase):
    def test_allocate_counts_uses_requested_ratio(self) -> None:
        self.assertEqual(
            allocate_counts(40, {"train": 0.6, "dev": 0.2, "test": 0.2}),
            {"train": 24, "dev": 8, "test": 8},
        )

    def test_duplicate_policy_keeps_one_same_label_and_drops_conflicts(self) -> None:
        stats = PreparationStats()
        candidates = [
            (1, "A fact.", "S"),
            (2, "  a   fact. ", "S"),
            (3, "Conflict.", "S"),
            (4, "conflict.", "NS"),
            (5, "Unique.", "NS"),
        ]
        retained = deduplicate_candidates(candidates, stats)
        self.assertEqual(retained, [(1, "A fact.", "S"), (5, "Unique.", "NS")])
        self.assertEqual(stats.exclusions["same_label_duplicate_claims"], 1)
        self.assertEqual(stats.exclusions["conflicting_duplicate_claims"], 2)

    def test_validation_rejects_duplicate_claim_ids(self) -> None:
        row = {
            "answer_id": "a1",
            "claim_id": "c1",
            "topic": "T",
            "source_model": "M",
            "claim_text": "Fact.",
            "gold_label": "supported",
            "gold_label_binary": 0,
            "source_file": "data/labeled/M.jsonl",
        }
        with self.assertRaisesRegex(DataValidationError, "duplicate claim_id"):
            validate_claim_rows([row, row], "fixture")

    def test_fixture_split_is_deterministic_and_leakage_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            source_files = ("A.jsonl", "B.jsonl", "C.jsonl")
            topics = [f"Topic {index}" for index in range(10)]
            for source_file in source_files:
                with (raw_dir / source_file).open("w", encoding="utf-8", newline="\n") as handle:
                    for topic in topics:
                        record = {
                            "topic": topic,
                            "output": "answer",
                            "annotations": [
                                annotation(
                                    (f"{topic} supported by {source_file}.", "S"),
                                    (f"{topic} unsupported by {source_file}.", "NS"),
                                    (f"{topic} irrelevant for {source_file}.", "IR"),
                                    (f"{topic} unknown for {source_file}.", "UNKNOWN"),
                                )
                            ],
                        }
                        handle.write(json.dumps(record) + "\n")
            config = Config(
                seed=42,
                max_answers=30,
                split_ratios={"train": 0.6, "dev": 0.2, "test": 0.2},
                source_files=source_files,
            )
            first = prepare(raw_dir, config)
            second = prepare(raw_dir, config)
            self.assertEqual(first.splits, second.splits)
            self.assertEqual(first.split_topics, second.split_topics)
            self.assertEqual(
                {name: len(topics) for name, topics in first.split_topics.items()},
                {"train": 6, "dev": 2, "test": 2},
            )
            topic_sets = [set(first.split_topics[name]) for name in ("train", "dev", "test")]
            self.assertFalse(topic_sets[0] & topic_sets[1])
            self.assertFalse(topic_sets[0] & topic_sets[2])
            self.assertFalse(topic_sets[1] & topic_sets[2])
            self.assertEqual(first.stats.exclusions["label_IR_irrelevant"], 30)
            self.assertEqual(first.stats.exclusions["unknown_labels"], 30)


class RealDataTests(unittest.TestCase):
    def test_real_data_meets_step_one_acceptance_criteria(self) -> None:
        config = load_config(ROOT / "config" / "data_config.json")
        raw_dir = ROOT / "data" / "raw" / "factscore" / "release" / "data" / "labeled"
        prepared = prepare(raw_dir, config)
        self.assertEqual(len(prepared.selected_topics), 40)
        self.assertEqual(
            {name: len(topics) for name, topics in prepared.split_topics.items()},
            {"train": 24, "dev": 8, "test": 8},
        )
        answer_counts = {
            name: len({row["answer_id"] for row in rows}) for name, rows in prepared.splits.items()
        }
        self.assertEqual(answer_counts, {"train": 72, "dev": 24, "test": 24})
        for rows in prepared.splits.values():
            self.assertEqual({row["gold_label"] for row in rows}, {"supported", "unsupported"})


if __name__ == "__main__":
    unittest.main()
