from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_DIR = Path("outputs/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOX_FILL = "#EFECE4"
BOX_EDGE = "#3A3A3A"
ARROW_COLOR = "#8A8A8A"
TITLE_COLOR = "#1A1A1A"
DETAIL_COLOR = "#595959"
CAPTION_COLOR = "#1F3864"


def _draw_box(ax, x, y, w, h, title, lines, title_size=12.5, detail_size=10.0):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1,
        edgecolor=BOX_EDGE,
        facecolor=BOX_FILL,
    )
    ax.add_patch(rect)
    cx = x + w / 2
    n_lines = len(lines)
    title_y = y + h * (0.70 if n_lines else 0.5)
    ax.text(
        cx, title_y, title,
        ha="center", va="center",
        fontsize=title_size, fontweight="bold", color=TITLE_COLOR,
    )
    if lines:
        detail_y = y + h * 0.32
        ax.text(
            cx, detail_y, "\n".join(lines),
            ha="center", va="center",
            fontsize=detail_size, color=DETAIL_COLOR, linespacing=1.6,
        )
    return cx


def vertical_flow(
    filename: str,
    caption: str,
    boxes: list[dict],
    box_width: float = 6.0,
    box_height: float = 1.05,
    v_gap: float = 0.55,
    fig_width: float = 7.2,
    title_size: float = 12.5,
    detail_size: float = 10.0,
) -> None:
    n = len(boxes)
    fig_height = n * box_height + (n - 1) * v_gap + 1.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, box_width + 1.0)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    x0 = 0.5
    y = fig_height - 0.55 - box_height
    tops_bottoms = []
    for box in boxes:
        cx = _draw_box(
            ax, x0, y, box_width, box_height,
            box["title"], box.get("lines", []),
            title_size=title_size, detail_size=detail_size,
        )
        tops_bottoms.append((cx, y, y + box_height))
        y -= (box_height + v_gap)

    for i in range(n - 1):
        cx = tops_bottoms[i][0]
        y_start = tops_bottoms[i][1]
        y_end = tops_bottoms[i + 1][2]
        ax.annotate(
            "", xy=(cx, y_end), xytext=(cx, y_start),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, linewidth=1.3, mutation_scale=14),
        )

    ax.text(
        x0 + box_width / 2, 0.35, caption,
        ha="center", va="center", fontsize=11, style="italic",
        color=CAPTION_COLOR, family="serif",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def tree_diagram(
    filename: str,
    caption: str,
    parent: dict,
    children: list[dict],
    parent_width: float = 6.0,
    parent_height: float = 1.05,
    child_width: float = 2.6,
    child_height: float = 1.05,
    v_gap: float = 1.3,
    h_gap: float = 0.4,
) -> None:
    n = len(children)
    total_children_width = n * child_width + (n - 1) * h_gap
    fig_width = max(parent_width + 1.0, total_children_width + 1.0)
    fig_height = parent_height + v_gap + child_height + 1.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    parent_x = fig_width / 2 - parent_width / 2
    parent_y = fig_height - 0.55 - parent_height
    cx_parent = _draw_box(
        ax, parent_x, parent_y, parent_width, parent_height,
        parent["title"], parent.get("lines", []),
    )

    children_top_y = parent_y - v_gap
    child_bottom_y = children_top_y - child_height
    start_x = fig_width / 2 - total_children_width / 2

    centers = []
    for i, child in enumerate(children):
        cx = start_x + i * (child_width + h_gap) + child_width / 2
        _draw_box(
            ax, cx - child_width / 2, child_bottom_y, child_width, child_height,
            child["title"], child.get("lines", []),
            title_size=11.5, detail_size=9.3,
        )
        centers.append(cx)

    branch_y = parent_y - v_gap / 2
    ax.plot([cx_parent, cx_parent], [parent_y, branch_y], color=ARROW_COLOR, linewidth=1.3)
    ax.plot([min(centers), max(centers)], [branch_y, branch_y], color=ARROW_COLOR, linewidth=1.3)
    for cx in centers:
        ax.annotate(
            "", xy=(cx, children_top_y), xytext=(cx, branch_y),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, linewidth=1.3, mutation_scale=13),
        )

    ax.text(
        fig_width / 2, 0.35, caption,
        ha="center", va="center", fontsize=11, style="italic",
        color=CAPTION_COLOR, family="serif",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


vertical_flow(
    "system_flowchart.png",
    "Figure 1  System Flowchart",
    [
        {"title": "Image input", "lines": ["Upload MRI scan (jpg / png)", "Read into memory as array"]},
        {"title": "Preprocessing", "lines": ["Grayscale → denoise (median filter)", "Histogram equalize → resize 224×224"]},
        {"title": "Feature extraction", "lines": ["CNN convolution filters", "Extract spatial tumor patterns"]},
        {"title": "Classification", "lines": ["Softmax output, 2 classes", "Healthy = 0, tumor = 1"]},
        {"title": "Output", "lines": ["Label + confidence score", "Saved to result log"]},
    ],
)

vertical_flow(
    "architecture_diagram.png",
    "Figure 2 System Architecture Diagram",
    [
        {"title": "User interface", "lines": ["Built with Streamlit", "Handles upload + display"]},
        {"title": "Processing engine", "lines": ["Pillow for image operations", "PyTorch for model inference"]},
        {"title": "CNN model", "lines": ["4 convolution blocks", "Trained on labeled MRI set"]},
        {"title": "Result display", "lines": ["Metrics + accuracy plots", "Prediction with confidence"]},
        {"title": "Artifacts", "lines": ["Model weights (.pt), JSON", "CSV log, PNG snapshot"]},
    ],
)

tree_diagram(
    "dataset_class_diagram.png",
    "Figure 3  Dataset / Class Diagram",
    {"title": "Dataset", "lines": ["Labeled MRI images, single source folder"]},
    [
        {"title": "Classes", "lines": ["Healthy = class 0", "Tumor = class 1"]},
        {"title": "Split", "lines": ["Train 70% / val 15%", "Test 15%, stratified"]},
        {"title": "Record", "lines": ["Path, label, hash", "Width, height, checksum"]},
    ],
)

vertical_flow(
    "processing_pipeline.png",
    "Figure 4 Processing Pipeline Diagram",
    [
        {"title": "Original", "lines": ["Raw RGB image as loaded"]},
        {"title": "Grayscale", "lines": ["Single channel, drop color"]},
        {"title": "Denoise", "lines": ["Median filter removes noise"]},
        {"title": "Equalize", "lines": ["Histogram equalization, boosts contrast"]},
        {"title": "Resize", "lines": ["224 × 224 pixels, bilinear"]},
        {"title": "Normalize", "lines": ["Mean 0.5, std 0.5"]},
        {"title": "CNN", "lines": ["Learned convolutional features"]},
        {"title": "Decision", "lines": ["Binary: healthy or tumor"]},
    ],
    box_width=4.6,
    box_height=0.82,
    v_gap=0.38,
    fig_width=5.6,
    title_size=11.5,
    detail_size=9.5,
)

print(f"Generated diagrams in {OUTPUT_DIR}")