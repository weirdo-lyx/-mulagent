from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LABEL_MAP: dict[str, tuple[str, int]] = {
    "S": ("supported", 0),
    "NS": ("unsupported", 1),
}
KNOWN_EXCLUDED_LABELS = {"IR": "irrelevant"}
VALID_GOLD_LABELS = {value[0] for value in LABEL_MAP.values()}
SPLIT_NAMES = ("train", "dev", "test")


class DataValidationError(ValueError):
    """Raised when source or generated data violates a required invariant."""


@dataclass(frozen=True)
class Config:
    seed: int
    max_answers: int
    split_ratios: dict[str, float]
    source_files: tuple[str, ...]


@dataclass
class PreparationStats:
    raw_answers: int = 0
    raw_human_facts: int = 0
    source_label_counts: Counter[str] = field(default_factory=Counter)
    exclusions: Counter[str] = field(default_factory=Counter)
    source_answer_counts: Counter[str] = field(default_factory=Counter)
    source_claim_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class PreparedData:
    all_claims: tuple[dict[str, Any], ...]
    splits: dict[str, tuple[dict[str, Any], ...]]
    split_topics: dict[str, tuple[str, ...]]
    selected_topics: tuple[str, ...]
    stats: PreparationStats


def load_config(path: Path) -> Config:
    payload = read_json(path)
    ratios = payload.get("split_ratios")
    if not isinstance(ratios, dict) or set(ratios) != set(SPLIT_NAMES):
        raise DataValidationError(f"split_ratios must contain exactly {SPLIT_NAMES}")
    parsed_ratios = {name: float(ratios[name]) for name in SPLIT_NAMES}
    if any(value <= 0 for value in parsed_ratios.values()):
        raise DataValidationError("all split ratios must be positive")
    if abs(sum(parsed_ratios.values()) - 1.0) > 1e-9:
        raise DataValidationError("split ratios must sum to 1")

    source_files = payload.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise DataValidationError("source_files must be a non-empty list")
    if len(source_files) != len(set(source_files)):
        raise DataValidationError("source_files contains duplicates")

    seed = int(payload.get("seed"))
    max_answers = int(payload.get("max_answers"))
    if max_answers <= 0:
        raise DataValidationError("max_answers must be positive")
    return Config(seed, max_answers, parsed_ratios, tuple(source_files))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DataValidationError(f"expected a JSON object in {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DataValidationError(f"blank JSONL line: {path}:{line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"invalid JSON: {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise DataValidationError(f"expected object: {path}:{line_number}")
            yield line_number, payload


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def model_name(source_file: str) -> str:
    return Path(source_file).stem


def model_slug(source_file: str) -> str:
    stem = model_name(source_file)
    return "".join(character.lower() for character in stem if character.isalnum())


def extract_fact_candidates(record: Mapping[str, Any], stats: PreparationStats) -> list[tuple[int, str, str]]:
    annotations = record.get("annotations")
    if annotations is None:
        stats.exclusions["answers_with_null_annotations"] += 1
        return []
    if not isinstance(annotations, list):
        stats.exclusions["answers_with_invalid_annotations"] += 1
        return []

    candidates: list[tuple[int, str, str]] = []
    fact_index = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            stats.exclusions["invalid_annotation_objects"] += 1
            continue
        facts = annotation.get("human-atomic-facts")
        if facts is None:
            stats.exclusions["sentences_with_null_human_facts"] += 1
            continue
        if not isinstance(facts, list):
            stats.exclusions["sentences_with_invalid_human_facts"] += 1
            continue
        for fact in facts:
            fact_index += 1
            stats.raw_human_facts += 1
            if not isinstance(fact, dict):
                stats.exclusions["invalid_fact_objects"] += 1
                continue
            text = fact.get("text")
            label = fact.get("label")
            label_key = label if isinstance(label, str) else "<missing>"
            stats.source_label_counts[label_key] += 1
            if not isinstance(text, str) or not text.strip():
                stats.exclusions["empty_claim_text"] += 1
                continue
            candidates.append((fact_index, text.strip(), label_key))
    return candidates


def deduplicate_candidates(
    candidates: Sequence[tuple[int, str, str]], stats: PreparationStats
) -> list[tuple[int, str, str]]:
    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for candidate in candidates:
        grouped[normalize_text(candidate[1])].append(candidate)

    retained: list[tuple[int, str, str]] = []
    for group in grouped.values():
        labels = {item[2] for item in group}
        if len(labels) > 1:
            stats.exclusions["conflicting_duplicate_claims"] += len(group)
            continue
        retained.append(group[0])
        if len(group) > 1:
            stats.exclusions["same_label_duplicate_claims"] += len(group) - 1
    retained.sort(key=lambda item: item[0])
    return retained


def convert_sources(raw_labeled_dir: Path, config: Config) -> tuple[list[dict[str, Any]], PreparationStats]:
    stats = PreparationStats()
    claims: list[dict[str, Any]] = []
    seen_answer_ids: set[str] = set()
    seen_claim_ids: set[str] = set()
    seen_source_topic: set[tuple[str, str]] = set()

    for source_file in config.source_files:
        source_path = raw_labeled_dir / source_file
        if not source_path.is_file():
            raise DataValidationError(f"missing configured source file: {source_path}")
        source_model = model_name(source_file)
        slug = model_slug(source_file)
        for line_number, record in iter_jsonl(source_path):
            stats.raw_answers += 1
            stats.source_answer_counts[source_model] += 1
            topic = record.get("topic")
            output = record.get("output")
            if not isinstance(topic, str) or not topic.strip():
                stats.exclusions["answers_with_empty_topic"] += 1
                continue
            topic = topic.strip()
            if not isinstance(output, str) or not output.strip():
                stats.exclusions["answers_with_empty_output"] += 1

            source_topic = (source_model, normalize_text(topic))
            if source_topic in seen_source_topic:
                stats.exclusions["duplicate_source_topic_answers"] += 1
                continue
            seen_source_topic.add(source_topic)

            answer_id = f"factscore-{slug}-{line_number:06d}"
            if answer_id in seen_answer_ids:
                raise DataValidationError(f"duplicate generated answer_id: {answer_id}")
            seen_answer_ids.add(answer_id)

            candidates = deduplicate_candidates(extract_fact_candidates(record, stats), stats)
            for original_fact_index, claim_text, raw_label in candidates:
                if raw_label in KNOWN_EXCLUDED_LABELS:
                    stats.exclusions[f"label_{raw_label}_{KNOWN_EXCLUDED_LABELS[raw_label]}"] += 1
                    continue
                if raw_label not in LABEL_MAP:
                    stats.exclusions["unknown_labels"] += 1
                    continue
                gold_label, gold_binary = LABEL_MAP[raw_label]
                claim_id = f"{answer_id}-c{original_fact_index:04d}"
                if claim_id in seen_claim_ids:
                    raise DataValidationError(f"duplicate generated claim_id: {claim_id}")
                seen_claim_ids.add(claim_id)
                claims.append(
                    {
                        "answer_id": answer_id,
                        "claim_id": claim_id,
                        "topic": topic,
                        "source_model": source_model,
                        "claim_text": claim_text,
                        "gold_label": gold_label,
                        "gold_label_binary": gold_binary,
                        "source_file": f"data/labeled/{source_file}",
                    }
                )
                stats.source_claim_counts[source_model] += 1
    return claims, stats


def allocate_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(exact[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def choose_and_split_topics(
    claims: Sequence[dict[str, Any]], config: Config
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    answer_claim_counts: Counter[str] = Counter(row["answer_id"] for row in claims)
    topic_models: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in claims:
        topic_models[row["topic"]][row["source_model"]].add(row["answer_id"])

    required_models = {model_name(filename) for filename in config.source_files}
    eligible_topics: list[str] = []
    for topic, models in topic_models.items():
        if set(models) != required_models:
            continue
        if any(len(answer_ids) != 1 for answer_ids in models.values()):
            continue
        answer_ids = [next(iter(answer_ids)) for answer_ids in models.values()]
        if all(answer_claim_counts[answer_id] > 0 for answer_id in answer_ids):
            eligible_topics.append(topic)

    answers_per_topic = len(required_models)
    topic_limit = min(len(eligible_topics), config.max_answers // answers_per_topic)
    if topic_limit < len(SPLIT_NAMES):
        raise DataValidationError("not enough eligible topics for three non-empty splits")

    shuffled = sorted(eligible_topics, key=str.casefold)
    random.Random(config.seed).shuffle(shuffled)
    selected_topics = tuple(shuffled[:topic_limit])
    split_counts = allocate_counts(topic_limit, config.split_ratios)
    split_topics: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        next_cursor = cursor + split_counts[name]
        split_topics[name] = tuple(selected_topics[cursor:next_cursor])
        cursor = next_cursor
    return split_topics, selected_topics


def build_splits(
    claims: Sequence[dict[str, Any]], split_topics: Mapping[str, Sequence[str]]
) -> dict[str, tuple[dict[str, Any], ...]]:
    topic_to_split: dict[str, str] = {}
    for split_name, topics in split_topics.items():
        for topic in topics:
            if topic in topic_to_split:
                raise DataValidationError(f"topic assigned to multiple splits: {topic}")
            topic_to_split[topic] = split_name

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for row in claims:
        split_name = topic_to_split.get(row["topic"])
        if split_name is not None:
            rows[split_name].append(row)
    return {name: tuple(rows[name]) for name in SPLIT_NAMES}


def validate_claim_rows(rows: Sequence[Mapping[str, Any]], description: str) -> None:
    required = {
        "answer_id",
        "claim_id",
        "topic",
        "source_model",
        "claim_text",
        "gold_label",
        "gold_label_binary",
        "source_file",
    }
    claim_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if set(row) != required:
            raise DataValidationError(f"{description} row {index} has unexpected fields")
        if row["claim_id"] in claim_ids:
            raise DataValidationError(f"duplicate claim_id in {description}: {row['claim_id']}")
        claim_ids.add(str(row["claim_id"]))
        if row["gold_label"] not in VALID_GOLD_LABELS:
            raise DataValidationError(f"invalid label in {description}: {row['gold_label']}")
        expected_binary = 0 if row["gold_label"] == "supported" else 1
        if row["gold_label_binary"] != expected_binary:
            raise DataValidationError(f"label/binary mismatch in {description}: {row['claim_id']}")
        for field_name in ("answer_id", "claim_id", "topic", "source_model", "claim_text", "source_file"):
            if not isinstance(row[field_name], str) or not row[field_name].strip():
                raise DataValidationError(f"empty {field_name} in {description} row {index}")


def validate_prepared(data: PreparedData, config: Config) -> None:
    validate_claim_rows(data.all_claims, "claims.jsonl")
    all_claim_ids: set[str] = set()
    split_topic_sets: dict[str, set[str]] = {}
    for split_name in SPLIT_NAMES:
        rows = data.splits[split_name]
        validate_claim_rows(rows, f"{split_name}.jsonl")
        labels = {row["gold_label"] for row in rows}
        if labels != VALID_GOLD_LABELS:
            raise DataValidationError(f"{split_name} must contain both supported and unsupported labels")
        split_topic_sets[split_name] = {row["topic"] for row in rows}
        for row in rows:
            claim_id = row["claim_id"]
            if claim_id in all_claim_ids:
                raise DataValidationError(f"claim appears in multiple splits: {claim_id}")
            all_claim_ids.add(claim_id)

    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_topic_sets[left] & split_topic_sets[right]
            if overlap:
                raise DataValidationError(f"topic leakage between {left} and {right}: {sorted(overlap)}")

    selected_answers = {row["answer_id"] for rows in data.splits.values() for row in rows}
    if len(selected_answers) > config.max_answers:
        raise DataValidationError("selected answer count exceeds max_answers")


def prepare(raw_labeled_dir: Path, config: Config) -> PreparedData:
    claims, stats = convert_sources(raw_labeled_dir, config)
    split_topics, selected_topics = choose_and_split_topics(claims, config)
    splits = build_splits(claims, split_topics)
    prepared = PreparedData(tuple(claims), splits, split_topics, selected_topics, stats)
    validate_prepared(prepared, config)
    return prepared


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=False))
            handle.write("\n")


def split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row["gold_label"]) for row in rows)
    claim_count = len(rows)
    return {
        "topics": len({row["topic"] for row in rows}),
        "answers": len({row["answer_id"] for row in rows}),
        "claims": claim_count,
        "supported": label_counts["supported"],
        "unsupported": label_counts["unsupported"],
        "unsupported_ratio": label_counts["unsupported"] / claim_count if claim_count else 0.0,
    }


def render_report(data: PreparedData, config: Config, manifest: Mapping[str, Any]) -> str:
    summaries = {name: split_summary(data.splits[name]) for name in SPLIT_NAMES}
    selected_rows = [row for name in SPLIT_NAMES for row in data.splits[name]]
    selected_summary = split_summary(selected_rows)
    stats = data.stats
    archive = manifest["archive"]
    lines = [
        "# FActScore 数据准备报告",
        "",
        "## 来源与完整性",
        "",
        f"- 官方仓库：{manifest['official_repository']}",
        f"- 官方下载目录：{manifest['official_drive_folder_url']}",
        f"- 下载日期：{manifest['downloaded_at']}",
        f"- 原始压缩包：`{archive['path']}`",
        f"- 文件大小：{archive['size_bytes']} bytes",
        f"- SHA-256：`{archive['sha256']}`",
        "- 未下载 Wikipedia 数据库。",
        "",
        "## 字段与标签",
        "",
        "官方每行是一条模型回答，主要字段为 `input`、`output`、`topic`、`cat` 和 `annotations`。",
        "本实验读取 `annotations[].human-atomic-facts[]`：`S` 映射为 `supported/0`，",
        "`NS` 映射为 `unsupported/1`；`IR` 表示无关事实，按方案排除并计数。",
        "",
        "## 原始数据统计",
        "",
        f"- 回答数：{stats.raw_answers}",
        f"- 人工原子事实数：{stats.raw_human_facts}",
        f"- 原标签分布：{json.dumps(dict(sorted(stats.source_label_counts.items())), ensure_ascii=False)}",
        f"- 转换后可用 Claim 数：{len(data.all_claims)}",
        f"- 各来源回答数：{json.dumps(dict(sorted(stats.source_answer_counts.items())), ensure_ascii=False)}",
        f"- 各来源可用 Claim 数：{json.dumps(dict(sorted(stats.source_claim_counts.items())), ensure_ascii=False)}",
        "",
        "## 排除与清洗",
        "",
    ]
    if stats.exclusions:
        lines.extend(f"- `{key}`：{value}" for key, value in sorted(stats.exclusions.items()))
    else:
        lines.append("- 无排除项。")
    lines.extend(
        [
            "",
            "同一回答内文本规范化后完全相同且标签相同的 Claim 只保留第一次；",
            "标签冲突的重复 Claim 整组排除。未知标签不会被猜测或映射。",
            "",
            "## 抽样与切分",
            "",
            f"- 随机种子：{config.seed}",
            f"- 最大回答数：{config.max_answers}",
            f"- 选中 topic 数：{len(data.selected_topics)}",
            f"- 实际回答数：{selected_summary['answers']}",
            f"- 实际 Claim 数：{selected_summary['claims']}",
            "- 切分单位：topic；同一 topic 的三个来源回答始终在同一集合。",
            "",
            "| 集合 | Topic | 回答 | Claim | supported | unsupported | unsupported 比例 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in SPLIT_NAMES:
        item = summaries[name]
        lines.append(
            f"| {name} | {item['topics']} | {item['answers']} | {item['claims']} | "
            f"{item['supported']} | {item['unsupported']} | {item['unsupported_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 验收检查",
            "",
            "- [x] JSONL 可逐行解析；",
            "- [x] Claim ID 全局唯一；",
            "- [x] 标签仅为 `supported` / `unsupported` 且二值映射一致；",
            "- [x] train/dev/test 的 topic 两两零重叠；",
            "- [x] 每个集合均含正、负两类 Claim；",
            "- [x] 固定配置重复运行得到逐字节一致的四个 JSONL 文件。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(root: Path, data: PreparedData, config: Config, manifest: Mapping[str, Any]) -> list[Path]:
    paths = [root / "data" / "processed" / "claims.jsonl"]
    write_jsonl(paths[0], data.all_claims)
    for name in SPLIT_NAMES:
        path = root / "data" / "splits" / f"{name}.jsonl"
        write_jsonl(path, data.splits[name])
        paths.append(path)
    report_path = root / "data" / "reports" / "data_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(data, config, manifest), encoding="utf-8", newline="\n")
    paths.append(report_path)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reproducibility(root: Path, config: Config, manifest: Mapping[str, Any]) -> dict[str, str]:
    raw_dir = root / "data" / "raw" / "factscore" / "release" / "data" / "labeled"
    results: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        for target in (Path(first_dir), Path(second_dir)):
            prepared = prepare(raw_dir, config)
            paths = write_outputs(target, prepared, config, manifest)
            results.append({str(path.relative_to(target)): sha256(path) for path in paths})
    if results[0] != results[1]:
        raise DataValidationError("reproducibility check failed: output hashes differ")
    return results[0]


def validate_source_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise DataValidationError("manifest archive section is invalid")
    archive_path = root / archive["path"]
    if archive_path.stat().st_size != archive["size_bytes"]:
        raise DataValidationError("raw archive size does not match manifest")
    if sha256(archive_path) != archive["sha256"]:
        raise DataValidationError("raw archive SHA-256 does not match manifest")

    release_root = root / "data" / "raw" / "factscore" / "release"
    for item in manifest.get("labeled_files", []):
        path = release_root / item["path"]
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise DataValidationError(f"raw labeled file does not match manifest: {item['path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe FActScore claim splits")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-reproducibility", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config = load_config(root / "config" / "data_config.json")
    manifest = read_json(root / "data" / "raw" / "factscore" / "source_manifest.json")
    validate_source_manifest(root, manifest)
    raw_dir = root / "data" / "raw" / "factscore" / "release" / "data" / "labeled"
    prepared = prepare(raw_dir, config)
    output_paths = write_outputs(root, prepared, config, manifest)
    print(f"Prepared {len(prepared.all_claims)} usable claims from {prepared.stats.raw_answers} answers.")
    for name in SPLIT_NAMES:
        summary = split_summary(prepared.splits[name])
        print(f"{name}: {summary['topics']} topics, {summary['answers']} answers, {summary['claims']} claims")
    if args.check_reproducibility:
        hashes = verify_reproducibility(root, config, manifest)
        print("Reproducibility check passed:")
        for path, digest in sorted(hashes.items()):
            print(f"  {path}: {digest}")
    print("Wrote:")
    for path in output_paths:
        print(f"  {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
