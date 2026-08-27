from pathlib import Path
import csv

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(".")
ANALYSIS = ROOT / "outputs/robustness_5seed5model_frozen6/complete_analysis_150models"
MANIFEST = ANALYSIS / "pymol_representative_local/pymol_representative_manifest.csv"
WHOLE_SOURCE = ROOT / "自己整理 figure list/Figure7_Figure8_optimized/PyMOL_sources"
LOCAL_SOURCE = ROOT / "自己整理 figure list/Figure7_Figure8_optimized/Figure8_v3_continuous_chain_sources"
OUT = ROOT / "自己整理 figure list/Figure7_Figure8_optimized"


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 17,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def trim_white(image, padding=32):
    rgb = np.asarray(image.convert("RGB"))
    mask = np.min(rgb, axis=2) < 248
    ys, xs = np.where(mask)
    if not len(xs):
        return image
    left = max(0, int(xs.min()) - padding)
    right = min(image.width, int(xs.max()) + padding + 1)
    top = max(0, int(ys.min()) - padding)
    bottom = min(image.height, int(ys.max()) + padding + 1)
    return image.crop((left, top, right, bottom))


def site_bbox(image):
    """Locate the colored donor residues in the whole-protein rendering."""
    rgb = np.asarray(image.convert("RGB"), dtype=float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = (b > 125) & (b > r * 1.18) & (b > g * 1.10)
    orange = (r > 175) & (g > 45) & (g < 195) & (b < 145)
    yellow = (r > 175) & (g > 145) & (b < 125)
    ys, xs = np.where(blue | orange | yellow)
    if not len(xs):
        return 0.38, 0.38, 0.20, 0.20
    height, width = rgb.shape[:2]
    xmin, xmax = xs.min() / width, xs.max() / width
    ymin, ymax = ys.min() / height, ys.max() / height
    pad_x, pad_y = 0.06, 0.07
    xmin, xmax = max(0, xmin - pad_x), min(1, xmax + pad_x)
    ymin, ymax = max(0, ymin - pad_y), min(1, ymax + pad_y)
    return xmin, 1 - ymax, xmax - xmin, ymax - ymin


def draw_panel(ax, record):
    panel = record["panel"]
    pid = record["protein_id"]
    whole = trim_white(Image.open(WHOLE_SOURCE / f"{panel}_{pid}_whole.png"), 50)
    local = trim_white(
        Image.open(LOCAL_SOURCE / f"{panel}_{pid}_continuous_local.png"), 34
    )

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.00, 0.995, f"({panel})", ha="left", va="top",
            fontsize=25, fontweight="bold", color="black")
    ax.text(0.105, 0.995, pid, ha="left", va="top",
            fontsize=21, fontweight="bold", color="black")

    # A compact locator view preserves the whole fold, but the local structural
    # region dominates the panel as in the published NCC1 Supplementary Fig. S13.
    whole_extent = (0.00, 0.42, 0.08, 0.89)
    ax.imshow(whole, extent=whole_extent, aspect="auto", zorder=1)
    bx, by, bw, bh = site_bbox(whole)
    x = whole_extent[0] + bx * (whole_extent[1] - whole_extent[0])
    y = whole_extent[2] + by * (whole_extent[3] - whole_extent[2])
    w = bw * (whole_extent[1] - whole_extent[0])
    h = bh * (whole_extent[3] - whole_extent[2])
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="black", lw=2.0, zorder=6))

    # The full protein cartoon remains displayed in the local rendering.  Only
    # the camera is moved toward the donor pair, so the surrounding peptide
    # chain stays continuous rather than appearing as disconnected fragments.
    local_ax = ax.inset_axes([0.43, 0.17, 0.56, 0.68], zorder=8)
    local_ax.imshow(local)
    local_ax.set_xticks([])
    local_ax.set_yticks([])
    for spine in local_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.7)

    ax.add_artist(ConnectionPatch(
        xyA=(x + w, y + h), coordsA=ax.transData,
        xyB=(0, 1), coordsB=local_ax.transAxes,
        color="black", lw=1.6, ls=(0, (2.2, 2.2)), zorder=7,
    ))
    ax.add_artist(ConnectionPatch(
        xyA=(x + w, y), coordsA=ax.transData,
        xyB=(0, 0), coordsB=local_ax.transAxes,
        color="black", lw=1.6, ls=(0, (2.2, 2.2)), zorder=7,
    ))

    pair = record["frozen_pair"].replace("-", "–")
    atoms = f"{record['donor_atom_a']}–{record['donor_atom_b']}"
    ax.text(0.71, 0.895, f"{pair} ({atoms})", ha="center", va="bottom",
            fontsize=18.5, fontweight="bold", color="black")

    # Place each distance beside (not over) its dashed donor line, following the
    # NCC1 Supplementary Figure S13 convention. Positions are panel-specific so
    # labels do not obscure residues or the surrounding peptide backbone.
    label_positions = {
        "A": (0.54, 0.39),
        "B": (0.67, 0.42),
        "C": (0.64, 0.31),
        "D": (0.69, 0.35),
        "E": (0.64, 0.27),
        "F": (0.56, 0.35),
    }
    label_x, label_y = label_positions[panel]
    local_ax.text(
        label_x, label_y,
        f"{float(record['donor_distance_A']):.2f} Å",
        transform=local_ax.transAxes,
        ha="center", va="center", fontsize=21.5, fontweight="bold", color="black",
        bbox=dict(boxstyle="square,pad=0.10", facecolor="white", edgecolor="none", alpha=0.90),
        zorder=20,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open() as handle:
        records = sorted(csv.DictReader(handle), key=lambda row: row["panel"])

    fig, axes = plt.subplots(3, 2, figsize=(16.5, 18.0), facecolor="white")
    for ax, record in zip(axes.flat, records):
        draw_panel(ax, record)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.982, bottom=0.025,
                        wspace=0.055, hspace=0.085)

    base = OUT / "Figure8_NCC1_style_v3"
    fig.savefig(base.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)

    with (OUT / "Figure8_NCC1_style_v3_source_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(base)


if __name__ == "__main__":
    main()
