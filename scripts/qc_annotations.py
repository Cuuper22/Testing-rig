#!/usr/bin/env python3
"""Quality control checks for Coptic manuscript annotations."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTANCES_PATH = ROOT / "data" / "annotations" / "instances_coptic.json"
TRANSLATIONS_PATH = ROOT / "data" / "annotations" / "translations.json"
SPLITS_DIR = ROOT / "data" / "splits"
REPORT_PATH = ROOT / "reports" / "qc_annotation_report.md"


class QCError(Exception):
    """Raised when a fatal QC issue is encountered."""


def load_json(path: Path) -> dict:
    if not path.exists():
        raise QCError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def check_annotations(instances: dict, translations: dict) -> tuple[list[str], set[str]]:
    messages: list[str] = []

    images = {img["id"]: img for img in instances.get("images", [])}
    image_filenames = {img["file_name"] for img in images.values()}

    annotations = instances.get("annotations", [])
    categories = {cat["id"] for cat in instances.get("categories", [])}

    if not categories:
        messages.append("⚠️ No categories defined in instances file.")

    seen_annotation_ids = Counter()
    transcription_to_ann = defaultdict(list)

    for ann in annotations:
        ann_id = ann.get("id")
        seen_annotation_ids[ann_id] += 1
        if seen_annotation_ids[ann_id] > 1:
            messages.append(f"❌ Duplicate annotation id detected: {ann_id}")

        image_id = ann.get("image_id")
        if image_id not in images:
            messages.append(f"❌ Annotation {ann_id} references missing image id {image_id}")

        cat_id = ann.get("category_id")
        if cat_id not in categories:
            messages.append(f"❌ Annotation {ann_id} references undefined category id {cat_id}")

        bbox = ann.get("bbox", [])
        if not (isinstance(bbox, list) and len(bbox) == 4):
            messages.append(f"❌ Annotation {ann_id} missing bbox with 4 entries")
        else:
            if any(not isinstance(v, (int, float)) for v in bbox):
                messages.append(f"❌ Annotation {ann_id} bbox contains non-numeric values: {bbox}")
            elif bbox[2] <= 0 or bbox[3] <= 0:
                messages.append(f"❌ Annotation {ann_id} bbox has non-positive width/height: {bbox}")

        attributes = ann.get("attributes", {})
        transcription_id = attributes.get("transcription_id")
        transcription = attributes.get("transcription")
        if not transcription_id:
            messages.append(f"❌ Annotation {ann_id} missing transcription_id")
        else:
            transcription_to_ann[transcription_id].append(ann_id)

        if not transcription:
            messages.append(f"❌ Annotation {ann_id} missing transcription text")

    duplicates = {k: v for k, v in transcription_to_ann.items() if len(v) > 1}
    for transcription_id, ann_ids in duplicates.items():
        joined = ", ".join(map(str, ann_ids))
        messages.append(
            f"❌ Transcription id {transcription_id} used by multiple annotations: {joined}"
        )

    translation_entries = {
        entry.get("transcription_id"): entry for entry in translations.get("translations", [])
    }

    missing_translation = sorted(set(transcription_to_ann) - set(translation_entries))
    if missing_translation:
        pretty = ", ".join(missing_translation)
        messages.append(f"⚠️ Missing translations for transcription ids: {pretty}")

    orphan_translations = sorted(set(translation_entries) - set(transcription_to_ann))
    if orphan_translations:
        pretty = ", ".join(orphan_translations)
        messages.append(f"⚠️ Translations without matching annotations: {pretty}")

    messages.append(
        f"✅ Checked {len(annotations)} annotations across {len(images)} images with {len(transcription_to_ann)} unique transcriptions."
    )

    return messages, image_filenames


def check_splits(image_filenames: set[str]) -> list[str]:
    messages: list[str] = []
    split_files = sorted(SPLITS_DIR.glob("*.txt"))
    if not split_files:
        messages.append("⚠️ No split files found.")
        return messages

    images_in_splits = set()
    for path in split_files:
        with path.open("r", encoding="utf-8") as fh:
            listed = [line.strip() for line in fh if line.strip()]
        images_in_splits.update(listed)
        missing = sorted(set(listed) - image_filenames)
        if missing:
            pretty = ", ".join(missing)
            messages.append(f"❌ {path.name} references unknown images: {pretty}")

        messages.append(f"✅ {path.name}: {len(listed)} entries")

    uncovered = sorted(image_filenames - images_in_splits)
    if uncovered:
        pretty = ", ".join(uncovered)
        messages.append(f"⚠️ Images not assigned to any split: {pretty}")

    return messages


def write_report(messages: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        fh.write("# Annotation QC Report\n\n")
        for message in messages:
            fh.write(f"- {message}\n")


if __name__ == "__main__":
    try:
        instances = load_json(INSTANCES_PATH)
        translations = load_json(TRANSLATIONS_PATH)

        annotation_messages, image_filenames = check_annotations(instances, translations)
        split_messages = check_splits(image_filenames)

        all_messages = annotation_messages + split_messages
        write_report(all_messages)

        print(f"QC completed. Report written to {REPORT_PATH}")
    except QCError as exc:
        print(f"QC failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"QC failed: invalid JSON - {exc}", file=sys.stderr)
        sys.exit(1)
