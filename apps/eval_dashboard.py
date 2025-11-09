"""Streamlit dashboard for inspecting OCR predictions and capturing corrections.

The app visualises each manuscript page alongside the model prediction and
reference transcription. Annotators can correct the prediction and export
annotations that will later be consumed by the active learning pipeline.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import streamlit as st

from src.active_learning.select import prioritize_samples

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations" / "active_learning"
ANNOTATIONS_FILE = ANNOTATIONS_DIR / "annotations.jsonl"
CSS_INJECTED_KEY = "__eval_dashboard_css_injected__"


@dataclass
class ManuscriptRecord:
    """Representation of a manuscript prediction."""

    page_id: str
    image_path: Optional[Path]
    prediction: str
    ground_truth: str
    metadata: Dict[str, object]

    @property
    def display_image(self) -> Optional[Path]:
        if self.image_path and self.image_path.exists():
            return self.image_path
        if self.image_path:
            st.warning(f"Image file not found: {self.image_path}")
        return None


def ensure_directories() -> None:
    """Create required directories for predictions and annotations."""

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def inject_css() -> None:
    """Inject lightweight styling used for highlighting errors."""

    if st.session_state.get(CSS_INJECTED_KEY):
        return
    st.markdown(
        """
        <style>
        .ocr-equal { color: inherit; }
        .ocr-replace { background-color: rgba(255, 193, 7, 0.35); }
        .ocr-delete { background-color: rgba(220, 53, 69, 0.35); text-decoration: line-through; }
        .ocr-insert { background-color: rgba(40, 167, 69, 0.35); }
        .ocr-text { font-family: "Fira Code", "SFMono-Regular", monospace; white-space: pre-wrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[CSS_INJECTED_KEY] = True


def load_prediction_file(file_path: Path) -> List[Dict[str, object]]:
    """Load predictions from JSON/JSONL/CSV files."""

    if not file_path.exists():
        st.error(f"Prediction file not found: {file_path}")
        return []

    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("predictions", [])  # type: ignore[return-value]
            st.error("Unsupported JSON structure. Expected list or object with 'predictions' key.")
            return []
    if file_path.suffix.lower() == ".csv":
        import csv

        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    st.error("Unsupported file format. Use .json, .jsonl, or .csv")
    return []


def find_prediction_files(directory: Path) -> List[Path]:
    """Return a sorted list of prediction files in a directory."""

    if not directory.exists():
        return []
    candidates = []
    for suffix in ("*.jsonl", "*.json", "*.csv"):
        candidates.extend(directory.glob(suffix))
    return sorted(candidates)


def prepare_records(predictions: Iterable[Dict[str, object]]) -> List[ManuscriptRecord]:
    """Convert raw prediction dictionaries into :class:`ManuscriptRecord`."""

    records: List[ManuscriptRecord] = []
    for prediction in predictions:
        page_id = str(prediction.get("page_id", ""))
        if not page_id:
            continue
        image_path = prediction.get("image_path")
        resolved_image: Optional[Path] = None
        if image_path:
            resolved_image = (BASE_DIR / image_path).resolve()
        record = ManuscriptRecord(
            page_id=page_id,
            image_path=resolved_image,
            prediction=str(prediction.get("prediction", "")),
            ground_truth=str(prediction.get("ground_truth", "")),
            metadata={key: prediction[key] for key in prediction if key not in {"page_id", "prediction", "ground_truth", "image_path"}},
        )
        records.append(record)
    return records


def render_diff(predicted: str, ground_truth: str) -> Dict[str, str]:
    """Return HTML highlighting the diff between predicted and ground truth text."""

    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, predicted, ground_truth)
    pred_fragments: List[str] = []
    truth_fragments: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        pred_piece = escape(predicted[i1:i2])
        truth_piece = escape(ground_truth[j1:j2])
        if tag == "equal":
            pred_fragments.append(f'<span class="ocr-equal">{pred_piece}</span>')
            truth_fragments.append(f'<span class="ocr-equal">{truth_piece}</span>')
        elif tag == "replace":
            pred_fragments.append(f'<span class="ocr-replace">{pred_piece}</span>')
            truth_fragments.append(f'<span class="ocr-replace">{truth_piece}</span>')
        elif tag == "delete":
            pred_fragments.append(f'<span class="ocr-delete">{pred_piece}</span>')
        elif tag == "insert":
            truth_fragments.append(f'<span class="ocr-insert">{truth_piece}</span>')
    return {
        "prediction": f'<div class="ocr-text">{"".join(pred_fragments)}</div>',
        "ground_truth": f'<div class="ocr-text">{"".join(truth_fragments)}</div>',
    }


def record_to_dict(record: ManuscriptRecord) -> Dict[str, object]:
    return {
        "page_id": record.page_id,
        "prediction": record.prediction,
        "ground_truth": record.ground_truth,
        "image_path": str(record.image_path) if record.image_path else None,
        **record.metadata,
    }


def save_annotation(record: ManuscriptRecord, corrected_text: str, notes: str) -> None:
    """Append an annotation to the JSONL file."""

    ensure_directories()
    payload = {
        "page_id": record.page_id,
        "corrected_text": corrected_text,
        "original_prediction": record.prediction,
        "ground_truth": record.ground_truth,
        "metadata": record.metadata,
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with ANNOTATIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sidebar_controls() -> Optional[ManuscriptRecord]:
    """Render sidebar controls and return the selected record."""

    st.sidebar.header("Dataset")
    files = find_prediction_files(PREDICTIONS_DIR)
    file_options = ["<Upload a file>"] + [file.name for file in files]
    selected_file = st.sidebar.selectbox("Prediction file", file_options)
    uploaded_records: Optional[List[ManuscriptRecord]] = None

    if selected_file == "<Upload a file>":
        uploaded = st.sidebar.file_uploader("Upload predictions (.json/.jsonl/.csv)")
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded.getvalue())
                tmp_path = Path(tmp_file.name)
            loaded = load_prediction_file(tmp_path)
            uploaded_records = prepare_records(loaded)
            tmp_path.unlink(missing_ok=True)
    else:
        data = load_prediction_file(PREDICTIONS_DIR / selected_file)
        uploaded_records = prepare_records(data)

    if not uploaded_records:
        st.sidebar.info("No predictions available. Place files in data/predictions/")
        return None

    st.sidebar.header("Active learning")
    strategy = st.sidebar.selectbox(
        "Prioritisation strategy",
        options=["entropy", "least_confident", "margin", "min_token"],
        index=0,
        help="Select how pages are prioritised for review.",
    )

    prioritized = prioritize_samples(
        [record_to_dict(record) for record in uploaded_records], strategy=strategy
    )

    st.sidebar.subheader("Queue")
    queue_size = st.sidebar.slider(
        "Queue length",
        min_value=1,
        max_value=min(len(prioritized), 50),
        value=min(10, len(prioritized)),
        step=1,
    )
    queue = prioritized[:queue_size]
    queue_ids = [item["page_id"] for item in queue]
    selection = st.sidebar.selectbox("Select page", options=queue_ids)

    selected_record = next((rec for rec in uploaded_records if rec.page_id == selection), None)

    with st.sidebar.expander("Queue preview", expanded=False):
        st.write(
            [
                {"page_id": item["page_id"], "uncertainty": round(item.get("uncertainty_score", 0), 4)}
                for item in queue
            ]
        )
    return selected_record


def render_record(record: ManuscriptRecord) -> None:
    """Render the manuscript page details and annotation tools."""

    st.subheader(f"Page {record.page_id}")
    columns = st.columns([1, 1])
    if record.display_image:
        columns[0].image(str(record.display_image), use_column_width=True)
    else:
        columns[0].info("No page image available.")

    diff_html = render_diff(record.prediction, record.ground_truth)
    with columns[1]:
        st.markdown("**Prediction**", unsafe_allow_html=True)
        st.markdown(diff_html["prediction"], unsafe_allow_html=True)
        st.markdown("**Ground truth**", unsafe_allow_html=True)
        st.markdown(diff_html["ground_truth"], unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Annotation")
    corrected_text = st.text_area(
        "Corrected transcription",
        value=record.prediction,
        height=200,
        help="Edit the prediction before saving.",
    )
    notes = st.text_area(
        "Notes (optional)",
        value="",
        height=100,
        help="Add context about difficult regions, layout quirks, etc.",
    )

    if st.button("Save annotation", type="primary"):
        save_annotation(record, corrected_text, notes)
        st.success("Annotation saved to data/annotations/active_learning/annotations.jsonl")

    with st.expander("Metadata", expanded=False):
        st.json(record.metadata)


def main() -> None:
    st.set_page_config(page_title="OCR Evaluation Dashboard", layout="wide")
    ensure_directories()
    inject_css()
    st.title("OCR Evaluation Dashboard")
    st.write(
        "Review OCR predictions, compare them with ground truth, and capture corrections "
        "for active learning."
    )

    selected_record = sidebar_controls()
    if selected_record is None:
        st.info(
            "Upload or select a predictions file to begin. Each record must contain "
            "'page_id', 'prediction', and 'ground_truth' fields."
        )
        return

    render_record(selected_record)


if __name__ == "__main__":
    main()
