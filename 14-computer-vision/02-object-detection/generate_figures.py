"""Stage and header figures for the object-detection article.

Both are drawn from objdet.py's own stage and assignment rules, with
the same seeds, so what the reader sees is what the script measured.
Neither figure needs a trained model: the header shows what the RULES
do to the ground truth, which is the article's subject.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, r"D:\Projects\Medium\algorithms-in-python"
                   r"\14-computer-vision\02-object-detection")
import objdet as O

OUT = ("D:/Projects/Medium/algorithms-in-python/"
       "14-computer-vision/02-object-detection/")

BG = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#6B7280"
GRIDC = "#CBD5E1"
CLS_COLOR = ["#DC2626", "#2563EB", "#059669",
             "#D97706", "#7C3AED", "#DB2777"]


def draw_scene(ax, img, objs, grid=True, boxes=True, lit=None,
               lit_color="#FBBF24", title=None):
    # Show the real (noisy) input, but window it to [0,1]: negative
    # noise clips to white instead of rendering as mid-grey, which
    # keeps the strokes legible without altering the data.
    ax.imshow(img, cmap="gray_r", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    if lit:
        for p in lit:
            gy, gx = divmod(p, O.GRID)
            ax.add_patch(Rectangle((gx * O.CELL - 0.5, gy * O.CELL - 0.5),
                                   O.CELL, O.CELL, facecolor=lit_color,
                                   alpha=0.45, edgecolor="none", zorder=1))
    if grid:
        for i in range(1, O.GRID):
            ax.axhline(i * O.CELL - 0.5, color="#64748B", lw=0.9, zorder=3)
            ax.axvline(i * O.CELL - 0.5, color="#64748B", lw=0.9, zorder=3)
    if boxes:
        for (cls, cy, cx, s) in objs:
            ax.add_patch(Rectangle((cx - s - 0.5, cy - s - 0.5),
                                   2 * s + 1, 2 * s + 1, fill=False,
                                   edgecolor=CLS_COLOR[cls], lw=1.8,
                                   zorder=4))
    ax.set_xlim(-0.5, O.SIDE - 0.5)
    ax.set_ylim(O.SIDE - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#94A3B8")
        sp.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=10.5, color=TEXT, pad=5,
                     fontfamily="sans-serif")


# ----------------------------------------------------------------- stage
def make_stage():
    rng = np.random.default_rng(4)
    picks = []
    while len(picks) < 8:
        img, objs = O.scene(rng)
        if len(objs) >= 2 or len(picks) < 2:
            picks.append((img + rng.normal(0, O.NOISE, img.shape), objs))

    # 4 wide x 2 tall SQUARE panels need a 2:1 canvas before any room
    # for titles, legend and footnote -- too short and the lower row's
    # titles land inside the upper row's images.
    fig, axes = plt.subplots(2, 4, figsize=(1800 / 150, 1300 / 150),
                             dpi=150)
    fig.patch.set_facecolor(BG)
    for ax, (img, objs) in zip(axes.ravel(), picks):
        draw_scene(ax, img, objs,
                   title="%d object%s" % (len(objs),
                                          "" if len(objs) == 1 else "s"))
    fig.suptitle("The stage: six shapes, one to four per scene, "
                 "never overlapping",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.975)
    handles = [Rectangle((0, 0), 1, 1, fill=False, lw=2,
                         edgecolor=CLS_COLOR[i])
               for i in range(O.N_CLS)]
    fig.legend(handles, O.CLASS_NAMES, loc="lower center", ncol=6,
               frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.068,
             "Boxes are the generative factors, not annotations: the "
             "script chose the class, centre and size, so the ground "
             "truth is exact. The 8x8 grid is where the detector speaks "
             "from -- one predictor per %d px cell." % O.CELL,
             ha="center", fontsize=10, color=MUTED, fontstyle="italic")
    plt.tight_layout(rect=(0, 0.10, 1, 0.94), h_pad=3.0)
    fig.savefig(OUT + "stage.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote stage.png  (%d scenes, objects %s)"
          % (len(picks), [len(o) for _, o in picks]))


# ---------------------------------------------------------------- header
def make_header():
    """One scene, three ways of deciding who answers for it."""
    rng = np.random.default_rng(11)
    img = objs = None
    while objs is None or len(objs) != 3:
        img, objs = O.scene(rng)
    noisy = img + rng.normal(0, O.NOISE, img.shape)

    centre = [r[0] for r in O.assign_centre(objs)]
    many = [r[0] for r in O.assign_many(objs)]

    fig, axes = plt.subplots(1, 3, figsize=(1800 / 150, 760 / 150),
                             dpi=150)
    fig.patch.set_facecolor(BG)
    draw_scene(axes[0], noisy, objs, lit=None,
               title="%d objects, %d predictors" % (len(objs), O.N_PRED))
    draw_scene(axes[1], noisy, objs, lit=centre, lit_color="#34D399",
               title="one owner each  (%d positives)" % len(centre))
    draw_scene(axes[2], noisy, objs, lit=many, lit_color="#FBBF24",
               title="many owners each  (%d positives)" % len(many))
    fig.suptitle("Object Detection: who answers for this object?",
                 fontsize=17, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.5, 0.055,
             "The network is identical in all three panels. Only the "
             "rule assigning objects to predictors changes -- and that "
             "rule decides what the loss is, whether duplicates appear, "
             "and whether suppression helps or hurts.",
             ha="center", fontsize=10.5, color=MUTED,
             fontstyle="italic")
    plt.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(OUT + "header_objdet.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote header_objdet.png  (centre %d positives, many %d)"
          % (len(centre), len(many)))
    assert len(centre) == len(objs), "centre rule should give one each"
    assert len(many) > len(objs), "many rule should give more"
    return len(objs), len(centre), len(many)


if __name__ == "__main__":
    make_stage()
    n_obj, n_centre, n_many = make_header()
    print("header: %d objects -> %d / %d positives"
          % (n_obj, n_centre, n_many))
