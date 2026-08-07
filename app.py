from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from brain_tumor.config import (
    CLASS_NAMES,
    DEFAULT_DATASET_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
)
from brain_tumor.data import discover_images, manifest_dataframe, preprocessing_stages
from brain_tumor.history import clear_history, load_history, record_prediction
from brain_tumor.inference import load_model, predict_image, predict_image_with_explanation
from brain_tumor.training import TrainingConfig, train_model
from brain_tumor.visualization import generate_result_plots


st.set_page_config(
    page_title="NeuroScan | Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _ensure_dirs() -> None:
    DEFAULT_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _dataset_status(dataset_dir: Path) -> tuple[int, dict[str, int], str | None]:
    try:
        records = discover_images(dataset_dir)
    except (FileNotFoundError, ValueError) as error:
        return 0, {}, str(error)
    counts = manifest_dataframe(records)["label"].value_counts().to_dict()
    return len(records), {key: int(value) for key, value in counts.items()}, None


def _save_uploaded_zip(uploaded_file) -> Path:
    _ensure_dirs()
    archive_path = DEFAULT_DATASET_DIR.parent / "uploaded_dataset.zip"
    archive_path.write_bytes(uploaded_file.getvalue())
    if DEFAULT_DATASET_DIR.exists():
        shutil.rmtree(DEFAULT_DATASET_DIR)
    DEFAULT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(DEFAULT_DATASET_DIR)
    return DEFAULT_DATASET_DIR


def _metric_card(label: str, value: str, note: str) -> None:
    st.metric(label, value, help=note)


def _load_metrics() -> dict | None:
    path = DEFAULT_OUTPUT_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _diagram_files() -> list[Path]:
    """Architecture / pipeline / flowchart images produced by generate_diagrams.py."""
    diagrams_dir = DEFAULT_OUTPUT_DIR / "diagrams"
    if not diagrams_dir.exists():
        return []
    return sorted(
        path for path in diagrams_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    )


def _prettify_filename(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def _image_to_data_uri(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    mime = "svg+xml" if ext == "svg" else ext
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def _build_output_zip() -> bytes:
    """Zip every file currently in the outputs directory (diagrams, plots, metrics, logs)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in DEFAULT_OUTPUT_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(DEFAULT_OUTPUT_DIR))
    return buffer.getvalue()


def _build_html_report(dataset_dir: Path, total: int, counts: dict, metrics: dict | None) -> str:
    """Self-contained HTML report (images embedded as base64) for offline viewing/printing."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def img_block(path: Path, caption: str) -> str:
        if not path.exists():
            return ""
        return (
            f'<figure style="margin:0 0 20px 0;">'
            f'<img src="{_image_to_data_uri(path)}" '
            f'style="max-width:100%;border:1px solid #ddd;border-radius:8px;">'
            f'<figcaption style="color:#666;font-size:0.9em;margin-top:4px;">{caption}</figcaption>'
            f"</figure>"
        )

    if metrics:
        metrics_html = f"""
        <table style="border-collapse:collapse;width:100%;max-width:420px;">
        <tr><th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">Metric</th>
            <th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">Value</th></tr>
        <tr><td style="padding:6px;">Accuracy</td><td style="padding:6px;">{metrics['accuracy']:.3f}</td></tr>
        <tr><td style="padding:6px;">Precision</td><td style="padding:6px;">{metrics['precision']:.3f}</td></tr>
        <tr><td style="padding:6px;">Recall</td><td style="padding:6px;">{metrics['recall']:.3f}</td></tr>
        <tr><td style="padding:6px;">F1-score</td><td style="padding:6px;">{metrics['f1_score']:.3f}</td></tr>
        <tr><td style="padding:6px;">Support</td><td style="padding:6px;">{metrics.get('support', 'N/A')}</td></tr>
        <tr><td style="padding:6px;">Device</td><td style="padding:6px;">{metrics.get('device', 'N/A')}</td></tr>
        </table>
        """
    else:
        metrics_html = "<p><em>Model not trained yet — no metrics available.</em></p>"

    diagrams = _diagram_files()
    diagram_html = "".join(img_block(path, _prettify_filename(path)) for path in diagrams) or (
        "<p><em>No architecture/pipeline diagrams found.</em></p>"
    )

    results_html = img_block(DEFAULT_OUTPUT_DIR / "training_curves.png", "Training and validation curves")
    results_html += img_block(DEFAULT_OUTPUT_DIR / "confusion_matrix.png", "Confusion matrix (Healthy vs. Tumor)")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NeuroScan Compiled Report</title></head>
<body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:40px auto;color:#222;padding:0 16px;">
<h1>NeuroScan — Compiled Project Report</h1>
<p style="color:#666;">Generated {generated}</p>
<hr>
<h2>1. Dataset Summary</h2>
<p>Dataset path: <code>{dataset_dir}</code></p>
<ul>
<li>Total images: {total:,}</li>
<li>Healthy: {counts.get('healthy', 0):,}</li>
<li>Tumor: {counts.get('tumor', 0):,}</li>
</ul>
<h2>2. Architecture &amp; Processing Pipeline</h2>
{diagram_html}
<h2>3. Model Performance Metrics</h2>
{metrics_html}
<h2>4. Training Curves and Confusion Matrix</h2>
{results_html}
<hr>
<p style="color:#999;font-size:0.85em;">NeuroScan is a research prototype for educational use only and is not a medical device.</p>
</body></html>"""


def _render_home(dataset_dir: Path) -> None:
    st.title("NeuroScan")
    st.subheader("Brain tumor screening research prototype")
    st.write(
        "A reproducible IPPR project that preprocesses MRI images, extracts learned "
        "features with a convolutional neural network, and classifies images as "
        "healthy or tumor."
    )
    st.warning(
        "Research and educational use only. This classifier is not a medical device "
        "and must not be used for diagnosis or treatment decisions."
    )
    total, counts, error = _dataset_status(dataset_dir)
    columns = st.columns(4)
    with columns[0]:
        _metric_card("Dataset images", f"{total:,}", "Images discovered in the configured dataset.")
    with columns[1]:
        _metric_card("Healthy", f"{counts.get('healthy', 0):,}", "Images assigned to the healthy class.")
    with columns[2]:
        _metric_card("Tumor", f"{counts.get('tumor', 0):,}", "Images assigned to the tumor class.")
    with columns[3]:
        _metric_card("Model", "Ready" if DEFAULT_MODEL_PATH.exists() else "Not trained", "The model becomes ready after training.")
    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Project coverage")
        st.markdown(
            "- Image input through upload or dataset folder\n"
            "- Grayscale conversion, median denoising, histogram equalization, and resizing\n"
            "- CNN feature extraction and binary classification\n"
            "- Accuracy, precision, recall, F1-score, and confusion matrix\n"
            "- Saved model, manifests, JSON metrics, curves, and report diagrams"
        )
    with right:
        st.markdown("### Expected Kaggle layout")
        st.code(
            "data/brain_tumor_dataset/\n"
            "├── Brain Tumor/\n"
            "│   ├── Cancer (1).jpg\n"
            "│   └── ...\n"
            "└── Healthy/\n"
            "    ├── Not Cancer (1).jpg\n"
            "    └── ...",
            language="text",
        )
        if error:
            st.info("Add the extracted Kaggle folder or use the Dataset page to upload a ZIP.")


def _render_dataset(dataset_dir: Path) -> None:
    st.title("Dataset and preprocessing")
    st.write(
        "Upload the Kaggle dataset as a ZIP, or point the app at an extracted folder. "
        "The loader supports class folders and the notebook's metadata.csv format."
    )
    uploaded = st.file_uploader("Upload dataset ZIP", type=["zip"])
    if uploaded is not None and st.button("Extract dataset", type="primary"):
        extracted = _save_uploaded_zip(uploaded)
        st.success(f"Dataset extracted to {extracted}")
        st.rerun()
    configured = st.text_input("Extracted dataset folder", value=str(dataset_dir))
    configured_path = Path(configured).expanduser()
    total, counts, error = _dataset_status(configured_path)
    if error:
        st.error(error)
        return
    records = discover_images(configured_path)
    frame = manifest_dataframe(records)
    st.success(f"Validated {total:,} readable labeled images.")
    columns = st.columns(3)
    columns[0].metric("Healthy images", f"{counts.get('healthy', 0):,}")
    columns[1].metric("Tumor images", f"{counts.get('tumor', 0):,}")
    columns[2].metric("Unique file contents", f"{frame['file_hash'].nunique():,}")
    st.dataframe(
        frame[["path", "label", "width", "height"]].head(25),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The train/validation/test split is grouped by file hash to reduce duplicate-image leakage.")

    st.markdown("### IPPR preprocessing preview")
    preview_file = st.file_uploader("Upload an MRI image for preprocessing preview", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if preview_file is not None:
        image = Image.open(preview_file)
        stages = preprocessing_stages(image, DEFAULT_IMAGE_SIZE)
        columns = st.columns(len(stages))
        for column, (name, stage) in zip(columns, stages.items()):
            with column:
                st.image(stage, caption=name.title(), use_container_width=True)


def _render_training(dataset_dir: Path) -> None:
    st.title("Train the CNN")
    st.write(
        "Training is deterministic by default. The best validation-loss checkpoint is "
        "restored before test evaluation, then all artifacts are saved in outputs/."
    )
    if not dataset_dir.exists():
        st.info("Add the dataset on the Dataset page before starting training.")
        return
    total, _, error = _dataset_status(dataset_dir)
    if error:
        st.error(error)
        return
    with st.form("training_form"):
        columns = st.columns(4)
        epochs = columns[0].number_input("Epochs", min_value=1, max_value=100, value=12)
        batch_size = columns[1].selectbox("Batch size", [16, 32, 64], index=1)
        image_size = columns[2].selectbox("Image size", [128, 160, 224], index=2)
        seed = columns[3].number_input("Random seed", min_value=0, value=42)
        learning_rate = st.number_input(
            "Learning rate", min_value=0.00001, max_value=0.1, value=0.0005, format="%.5f"
        )
        submitted = st.form_submit_button("Start training", type="primary")
    if submitted:
        if total < 100:
            st.warning("The guidelines recommend at least 100 images; training can continue, but results may be unstable.")
        progress = st.progress(0, text="Preparing dataset and model...")
        started = time.perf_counter()
        with st.spinner("Training CNN. This may take several minutes on CPU."):
            summary = train_model(
                TrainingConfig(
                    dataset_dir=str(dataset_dir),
                    output_dir=str(DEFAULT_OUTPUT_DIR),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    image_size=int(image_size),
                    learning_rate=float(learning_rate),
                    seed=int(seed),
                )
            )
            generate_result_plots(DEFAULT_OUTPUT_DIR)
        progress.progress(100, text="Training complete.")
        st.success(f"Completed in {summary['duration_seconds']:.1f}s using {summary['device']}.")
        st.session_state["last_summary"] = summary
        st.rerun()
    metrics = _load_metrics()
    if metrics:
        st.markdown("### Latest test-set metrics")
        columns = st.columns(4)
        columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        columns[1].metric("Precision", f"{metrics['precision']:.3f}")
        columns[2].metric("Recall", f"{metrics['recall']:.3f}")
        columns[3].metric("F1-score", f"{metrics['f1_score']:.3f}")


def _render_prediction() -> None:
    st.title("Upload and classify")
    st.write("Upload a single MRI image to run the complete preprocessing and CNN inference path.")
    if not DEFAULT_MODEL_PATH.exists():
        st.info("Train a model first. No prediction is shown until a real checkpoint exists.")
        return
    uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if uploaded is None:
        st.info("Supported formats: JPG, JPEG, PNG, BMP, TIFF.")
        return
    file_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(file_bytes))
    model, class_names, image_size, device = load_model(DEFAULT_MODEL_PATH)
    show_gradcam = st.checkbox(
        "Show Grad-CAM heatmap (what the model is actually looking at)", value=True
    )
    if show_gradcam:
        try:
            result = predict_image_with_explanation(model, image, class_names, image_size, device)
        except Exception as error:  # noqa: BLE001 - surface the failure but keep the page usable
            st.warning(f"Grad-CAM failed ({error}); showing prediction without heatmap.")
            result = predict_image(model, image, class_names, image_size, device)
    else:
        result = predict_image(model, image, class_names, image_size, device)

    # Record to history once per distinct uploaded file, not on every rerun
    # (Streamlit reruns this whole function whenever any widget on the page changes).
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("last_recorded_prediction_hash") != file_hash:
        record_prediction(image, uploaded.name, result)
        st.session_state["last_recorded_prediction_hash"] = file_hash

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
        if "gradcam_image" in result:
            st.image(
                result["gradcam_image"],
                caption="Grad-CAM: red/yellow = regions that drove the prediction",
                use_container_width=True,
            )
        else:
            st.image(result["processed_image"], caption=f"Processed input ({image_size} × {image_size})", use_container_width=True)
    with right:
        label = result["label"].title()
        st.metric("Predicted class", label, f"{result['confidence']:.1%} confidence")
        for class_name, probability in result["probabilities"].items():
            st.progress(probability, text=f"{class_name.title()}: {probability:.1%}")
        if "gradcam_image" in result:
            st.caption(
                "If the hot (red/yellow) region sits on scan borders, corners, or "
                "text rather than brain tissue, the model is likely relying on "
                "dataset-specific artifacts rather than genuine tumor features."
            )
        st.warning("Prediction is for demonstration and research only; consult a qualified clinician for medical interpretation.")
    st.caption("This prediction has been saved to the History page.")


def _render_results() -> None:
    st.title("Results and evaluation")
    metrics = _load_metrics()
    if metrics is None:
        st.info("Train the model to populate evaluation results.")
        return
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    columns[1].metric("Precision", f"{metrics['precision']:.3f}")
    columns[2].metric("Recall", f"{metrics['recall']:.3f}")
    columns[3].metric("F1-score", f"{metrics['f1_score']:.3f}")
    st.caption(f"Test support: {metrics['support']} images • Device: {metrics['device']}")
    columns = st.columns(2)
    with columns[0]:
        curve = DEFAULT_OUTPUT_DIR / "training_curves.png"
        if curve.exists():
            st.image(str(curve), caption="Training and validation curves", use_container_width=True)
    with columns[1]:
        matrix = DEFAULT_OUTPUT_DIR / "confusion_matrix.png"
        if matrix.exists():
            st.image(str(matrix), caption="Test-set confusion matrix", use_container_width=True)
    st.markdown("### Per-class report")
    report = metrics.get("classification_report", {})
    st.dataframe(
        [
            {"class": name.title(), **{key: round(float(value), 3) for key, value in values.items() if key in {"precision", "recall", "f1-score", "support"}}}
            for name, values in report.items()
            if name in CLASS_NAMES
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download metrics JSON",
        data=json.dumps(metrics, indent=2),
        file_name="metrics.json",
        mime="application/json",
    )


def _render_history() -> None:
    st.title("Prediction history")
    st.write(
        "Every image run through Upload & predict, saved locally so you can review, "
        "search, or export past results — persists across app restarts."
    )
    records = load_history()
    if not records:
        st.info("No predictions recorded yet. Run one on the Upload & predict page.")
        return

    tumor_count = sum(1 for record in records if record["label"] == "tumor")
    columns = st.columns(3)
    columns[0].metric("Total predictions", len(records))
    columns[1].metric("Tumor calls", tumor_count)
    columns[2].metric("Healthy calls", len(records) - tumor_count)

    st.divider()
    search = st.text_input("Filter by filename", "")
    filtered = [r for r in records if search.lower() in r["filename"].lower()] if search else records
    filtered = list(reversed(filtered))  # newest first

    for record in filtered:
        header = f"{record['timestamp']} • {record['filename']} • {record['label'].title()} ({record['confidence']:.1%})"
        with st.expander(header):
            record_columns = st.columns([1, 1, 1])
            image_path = Path(record["image_path"])
            if image_path.exists():
                record_columns[0].image(str(image_path), caption="Uploaded image", use_container_width=True)
            gradcam_path = record.get("gradcam_path")
            if gradcam_path and Path(gradcam_path).exists():
                record_columns[1].image(str(gradcam_path), caption="Grad-CAM", use_container_width=True)
            with record_columns[2]:
                st.metric("Predicted class", record["label"].title(), f"{record['confidence']:.1%} confidence")
                for class_name, probability in record["probabilities"].items():
                    st.progress(probability, text=f"{class_name.title()}: {probability:.1%}")

    st.divider()
    st.markdown("### Export or clear")
    export_columns = st.columns(3)
    with export_columns[0]:
        st.download_button(
            "Download history (JSON)",
            data=json.dumps(records, indent=2),
            file_name="prediction_history.json",
            mime="application/json",
        )
    with export_columns[1]:
        csv_lines = ["id,timestamp,filename,label,confidence," + ",".join(CLASS_NAMES)]
        for record in records:
            probability_values = ",".join(f"{record['probabilities'].get(name, 0.0):.4f}" for name in CLASS_NAMES)
            csv_lines.append(
                f"{record['id']},{record['timestamp']},{record['filename']},"
                f"{record['label']},{record['confidence']:.4f},{probability_values}"
            )
        st.download_button(
            "Download history (CSV)",
            data="\n".join(csv_lines),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    with export_columns[2]:
        if st.button("Clear history", type="secondary"):
            clear_history()
            st.session_state.pop("last_recorded_prediction_hash", None)
            st.success("History cleared.")
            st.rerun()


def _render_report(dataset_dir: Path) -> None:
    st.title("Project report")
    st.write(
        "A compiled summary of the dataset, system architecture, processing pipeline, "
        "and model performance in one place — ready to review or export for the IPPR write-up."
    )
    total, counts, _ = _dataset_status(dataset_dir)
    metrics = _load_metrics()

    columns = st.columns(4)
    columns[0].metric("Dataset images", f"{total:,}")
    columns[1].metric("Model status", "Trained" if metrics else "Not trained")
    columns[2].metric("Accuracy", f"{metrics['accuracy']:.3f}" if metrics else "—")
    columns[3].metric("F1-score", f"{metrics['f1_score']:.3f}" if metrics else "—")

    st.divider()
    st.markdown("### Architecture & pipeline diagrams")
    diagrams = _diagram_files()
    if diagrams:
        diagram_columns = st.columns(2)
        for index, path in enumerate(diagrams):
            with diagram_columns[index % 2]:
                st.image(str(path), caption=_prettify_filename(path), use_container_width=True)
    else:
        st.info(
            "No diagrams found yet. These are generated automatically by "
            "generate_diagrams.py the first time the app starts."
        )

    st.divider()
    st.markdown("### Performance summary")
    if metrics:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        metric_columns[1].metric("Precision", f"{metrics['precision']:.3f}")
        metric_columns[2].metric("Recall", f"{metrics['recall']:.3f}")
        metric_columns[3].metric("F1-score", f"{metrics['f1_score']:.3f}")
        st.caption(f"Test support: {metrics['support']} images • Device: {metrics['device']}")
        result_columns = st.columns(2)
        curve = DEFAULT_OUTPUT_DIR / "training_curves.png"
        matrix = DEFAULT_OUTPUT_DIR / "confusion_matrix.png"
        if curve.exists():
            result_columns[0].image(str(curve), caption="Training and validation curves", use_container_width=True)
        if matrix.exists():
            result_columns[1].image(str(matrix), caption="Confusion matrix", use_container_width=True)
    else:
        st.info("Train the model on the Train model page to populate performance results.")

    st.divider()
    st.markdown("### Download compiled report")
    st.write(
        "Export everything above as a single, self-contained HTML file (images embedded, "
        "opens offline), or grab every raw file in outputs/ as a ZIP."
    )
    download_columns = st.columns(2)
    with download_columns[0]:
        html_report = _build_html_report(dataset_dir, total, counts, metrics)
        st.download_button(
            "Download report (HTML)",
            data=html_report,
            file_name="neuroscan_report.html",
            mime="text/html",
            type="primary",
        )
    with download_columns[1]:
        if DEFAULT_OUTPUT_DIR.exists() and any(DEFAULT_OUTPUT_DIR.iterdir()):
            st.download_button(
                "Download all outputs (ZIP)",
                data=_build_output_zip(),
                file_name="neuroscan_outputs.zip",
                mime="application/zip",
            )
        else:
            st.caption("No output files yet.")


def main() -> None:
    _ensure_dirs()
    if not (DEFAULT_OUTPUT_DIR / "diagrams").exists():
        import subprocess

        subprocess.run(["python", "generate_diagrams.py"], check=True)
    st.sidebar.title("NeuroScan")
    st.sidebar.caption("IPPR • CNN brain tumor classification")
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Dataset", "Train model", "Upload & predict", "History", "Results", "Report"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Reference: Kaggle Brain Tumor Dataset / PyTorch CNN notebook")
    dataset_dir = Path(
        st.sidebar.text_input("Dataset path", value=str(DEFAULT_DATASET_DIR))
    ).expanduser()
    pages = {
        "Home": lambda: _render_home(dataset_dir),
        "Dataset": lambda: _render_dataset(dataset_dir),
        "Train model": lambda: _render_training(dataset_dir),
        "Upload & predict": _render_prediction,
        "History": _render_history,
        "Results": _render_results,
        "Report": lambda: _render_report(dataset_dir),
    }
    pages[page]()


if __name__ == "__main__":
    main()