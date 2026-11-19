"""Stage and header figures for the semantic segmentation article.

Both are drawn from segment.py's own stage, with its seeds, so what
the reader sees is what the script measured.  The header needs no
trained model: it shows the ground truth and the band of pixels where
the whole result lives.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

sys.path.insert(0, r"D:\Projects\Medium\algorithms-in-python"
                   r"\14-computer-vision\03-semantic-segmentation")
import segment as S

OUT = ("D:/Projects/Medium/algorithms-in-python/"
       "14-computer-vision/03-semantic-segmentation/")

BG = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#6B7280"
# background then the six shapes
PALETTE = ["#F1F5F9", "#DC2626", "#2563EB", "#059669",
           "#D97706", "#7C3AED", "#DB2777"]
CMAP = ListedColormap(PALETTE)


def show_image(ax, img, title=None):
    ax.imshow(img, cmap="gray_r", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    _frame(ax, title)


def show_labels(ax, lab, title=None):
    ax.imshow(lab, cmap=CMAP, vmin=0, vmax=S.N_CLS - 1,
              interpolation="nearest")
    _frame(ax, title)


def _frame(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#94A3B8")
        sp.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=10.5, color=TEXT, pad=5,
                     fontfamily="sans-serif")


def make_stage():
    rng = np.random.default_rng(3)
    picks = []
    while len(picks) < 4:
        img, lab, placed = S.scene(rng)
        if len(placed) >= 2:
            picks.append((img + rng.normal(0, S.NOISE, img.shape),
                          lab))

    fig, axes = plt.subplots(2, 4, figsize=(1800 / 150, 1120 / 150),
                             dpi=150)
    fig.patch.set_facecolor(BG)
    for i, (img, lab) in enumerate(picks):
        show_image(axes[0, i], img,
                   "input" if i == 0 else None)
        show_labels(axes[1, i], lab,
                    "label for every pixel" if i == 0 else None)
    fig.suptitle("The stage: six filled shapes, and a label for "
                 "every pixel",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.975)
    handles = [Rectangle((0, 0), 1, 1, facecolor=PALETTE[i],
                         edgecolor="#94A3B8", lw=0.6)
               for i in range(S.N_CLS)]
    fig.legend(handles, S.CLASS_NAMES, loc="lower center", ncol=7,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.072,
             "The label map is not an annotation \u2014 the script "
             "chose each shape's class, centre and radius and then "
             "drew the pixels, so every boundary is exact and no "
             "annotator can disagree with it.",
             ha="center", fontsize=10, color=MUTED, fontstyle="italic")
    plt.tight_layout(rect=(0, 0.10, 1, 0.94), h_pad=2.0)
    fig.savefig(OUT + "stage.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote stage.png")


def make_pairs():
    """The pierced shapes beside their solid twins."""
    fig, axes = plt.subplots(2, 4, figsize=(1500 / 150, 880 / 150),
                             dpi=150)
    fig.patch.set_facecolor(BG)
    for row, (solid, pierced) in enumerate(((1, 5), (2, 6))):
        for col, r in enumerate((4, 7)):
            for k, cls in enumerate((solid, pierced)):
                ax = axes[row, col * 2 + k]
                lab = np.zeros((2 * r + 3, 2 * r + 3), np.int64)
                lab[1:-1, 1:-1][S.shape_mask(cls, r)] = cls
                show_labels(ax, lab, "%s, r=%d"
                            % (S.CLASS_NAMES[cls], r))
    fig.suptitle("Identical outside, different inside",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.99)
    fig.text(0.5, 0.055,
             "A pierced shape is its solid twin minus a small central "
             "hole: same silhouette, differing on 5 pixels at r=4 and "
             "13 at r=7. A pixel on the RIM is only labelled correctly "
             "by a model whose receptive field reaches the middle.",
             ha="center", fontsize=10, color=MUTED, fontstyle="italic")
    plt.tight_layout(rect=(0, 0.10, 1, 0.94), h_pad=1.6)
    fig.savefig(OUT + "pairs.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote pairs.png")


def make_header():
    """One scene: the input, the truth, and the pixels that decide it."""
    rng = np.random.default_rng(3)
    img = lab = None
    while lab is None or len(np.unique(lab)) < 3:
        img, lab, placed = S.scene(rng)
    noisy = img + rng.normal(0, S.NOISE, img.shape)
    dist = S.boundary_distance(lab)
    edge = (dist == 0) & (lab > 0)

    fig, axes = plt.subplots(1, 3, figsize=(1700 / 150, 720 / 150),
                             dpi=150)
    fig.patch.set_facecolor(BG)
    show_image(axes[0], noisy, "the input")
    show_labels(axes[1], lab, "a label for every pixel")
    axes[2].imshow(np.zeros_like(lab), cmap="gray_r", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[2].imshow(np.where(edge, 1.0, np.nan), cmap="autumn",
                   vmin=0, vmax=1, interpolation="nearest")
    _frame(axes[2], "the %d%% that decide it"
           % round(100 * edge.sum() / max(1, (lab > 0).sum())))
    fig.suptitle("Semantic Segmentation: the bill for throwing away "
                 "position",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.99)
    fig.text(0.5, 0.055,
             "Pooling buys translation invariance by discarding where "
             "things are. Segmentation needs it back \u2014 and the "
             "entire difference between an encoder-decoder and a "
             "U-Net turns out to live on the highlighted pixels.",
             ha="center", fontsize=10.5, color=MUTED,
             fontstyle="italic")
    plt.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(OUT + "header_segment.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote header_segment.png  (boundary share %.4f)"
          % (edge.sum() / (lab > 0).sum()))


if __name__ == "__main__":
    make_stage()
    make_pairs()
    make_header()
