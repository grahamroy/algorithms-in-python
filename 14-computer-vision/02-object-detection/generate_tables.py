"""Render the object-detection article's data tables as PNGs.

Medium collapses runs of whitespace, so a monospaced ASCII table
pasted into it loses its columns.  Every table therefore ships as an
image.  All values below are transcribed from a single frozen run of
objdet.py; the asserts at the bottom of each builder are there so a
typo cannot survive quietly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = ("D:/Projects/Medium/algorithms-in-python/"
       "14-computer-vision/02-object-detection/")

BG = "#FFFFFF"
TEXT = "#1F2937"
HEADER_BG = "#1F2937"
HEADER_TEXT = "#FFFFFF"
ROW_EVEN = "#F8FAFC"
ROW_ODD = "#FFFFFF"
BORDER = "#E2E8F0"
RULE_C = "#94A3B8"
NOTE_C = "#6B7280"
GOOD = "#059669"
BAD = "#DC2626"
NEUTRAL = "#2563EB"
MONO = "DejaVu Sans Mono"


def render(name, title, headers, rows, col_widths, align, colors,
           footnote=None, dividers=(), width_px=1860, bold_first=True):
    """One table -> one PNG.

    rows   : list of lists of strings, already formatted.
    align  : per column, 'l' or 'c'.
    colors : per column, a colour or a callable(cell_text) -> colour.
    """
    total_w = sum(col_widths)
    # Two-line column labels need a taller band or the first line is
    # clipped by the top of the header rectangle.
    header_h = 1.45 if any("\n" in h for h in headers) else 0.95
    row_h = 0.80
    n = len(rows)
    table_h = header_h + n * row_h
    # Font sizes are absolute (points) but the data box is in table
    # units, so a wider table drawn at a fixed pixel width shrinks
    # every column until the text collides.  Pin pixels-per-unit and
    # let the canvas grow with the content instead.
    px_per_unit = width_px / 22.8
    width_px = int((total_w + 2.0) * px_per_unit)
    height_px = int((table_h + 2.0) * px_per_unit)

    fig, ax = plt.subplots(figsize=(width_px / 150, height_px / 150),
                           dpi=150)
    ax.set_xlim(0, total_w + 2.0)
    ax.set_ylim(0, table_h + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    x0, y0 = 1.0, 0.8

    def col_x(i):
        return x0 + sum(col_widths[:i])

    ax.text(total_w / 2 + 1.0, table_h + 1.3, title, fontsize=16,
            fontweight="bold", ha="center", va="center", color=TEXT,
            fontfamily="sans-serif")

    header_y = y0 + n * row_h
    ax.add_patch(Rectangle((x0, header_y), total_w, header_h,
                           facecolor=HEADER_BG, edgecolor=HEADER_BG,
                           linewidth=0, zorder=1))
    for i, h in enumerate(headers):
        ax.text(col_x(i) + col_widths[i] / 2, header_y + header_h / 2,
                h, fontsize=12.5, fontweight="bold", ha="center",
                va="center", color=HEADER_TEXT,
                fontfamily="sans-serif", zorder=2)

    for r, row in enumerate(rows):
        ry = y0 + (n - 1 - r) * row_h
        ax.add_patch(Rectangle((x0, ry), total_w, row_h,
                               facecolor=ROW_EVEN if r % 2 == 0
                               else ROW_ODD, edgecolor=BORDER,
                               linewidth=0.8, zorder=1))
        for i, cell in enumerate(row):
            c = colors[i]
            col = c(cell, r) if callable(c) else c
            mono = i > 0 or not bold_first
            if align[i] == "l":
                ax.text(col_x(i) + 0.25, ry + row_h / 2, cell,
                        fontsize=11 if mono else 11.5,
                        fontweight="normal" if i else
                        ("bold" if bold_first else "normal"),
                        ha="left", va="center", color=col,
                        fontfamily=MONO if mono else "sans-serif",
                        zorder=2)
            else:
                ax.text(col_x(i) + col_widths[i] / 2, ry + row_h / 2,
                        cell, fontsize=11, ha="center", va="center",
                        color=col, fontfamily=MONO, zorder=2)

    ax.add_patch(Rectangle((x0, y0), total_w, table_h, facecolor="none",
                           edgecolor=BORDER, linewidth=1.2, zorder=3))
    for i in range(1, len(col_widths)):
        ax.plot([col_x(i), col_x(i)], [y0, header_y + header_h],
                color=BORDER, linewidth=0.8, zorder=3)
    for after in dividers:
        ry = y0 + (n - after) * row_h
        ax.plot([x0, x0 + total_w], [ry, ry], color=RULE_C,
                linewidth=1.4, zorder=5)

    if footnote:
        ax.text(total_w / 2 + 1.0, y0 - 0.45, footnote, fontsize=10,
                ha="center", va="center", color=NOTE_C,
                fontfamily="sans-serif", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(OUT + name, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none", pad_inches=0.2)
    plt.close(fig)
    print("wrote %s  (%d rows)" % (name, n))


# =====================================================================
# Measurement 1 -- what a fixed-size output can represent.
# =====================================================================
def table_representation():
    rows = [
        ["1x1  (Part 1's output)", "1", "32.0 px", "0.6016"],
        ["2x2", "4", "16.0 px", "0.1231"],
        ["4x4", "16", "8.0 px", "0.0016"],
        ["8x8  (used here)", "64", "4.0 px", "0.0000"],
        ["16x16", "256", "2.0 px", "0.0000"],
    ]
    crowd = [
        ["4 objects, spaced", "0.0023", "0.0000", "0.0000"],
        ["8 objects, touching", "0.0565", "0.0000", "0.0000"],
        ["12 objects, overlapping", "0.1502", "0.0063", "0.0000"],
    ]

    def lost_colour(v, _r=0):
        try:
            x = float(v)
        except ValueError:
            return TEXT
        return BAD if x > 0.05 else (NEUTRAL if x > 0 else GOOD)

    render("representation_table.png",
           "What the output grid can even represent",
           ["Output grid", "Predictors", "Cell", "Objects it cannot hold"],
           rows, [7.0, 4.2, 3.6, 7.4], ["l", "c", "c", "c"],
           [TEXT, NEUTRAL, NOTE_C, lost_colour],
           footnote="4000 scenes, 10041 objects. An object is "
                    "unrepresentable when another object's centre "
                    "shares its cell \u2014 the loss has nowhere to "
                    "put it.",
           dividers=(2,))

    render("crowding_table.png",
           "When grid resolution starts to bite",
           ["Scene density", "4x4 grid", "8x8 grid", "16x16 grid"],
           crowd, [7.4, 4.4, 4.4, 4.4], ["l", "c", "c", "c"],
           [TEXT, lost_colour, lost_colour, lost_colour],
           footnote="Objects this stage cannot represent, by grid. "
                    "The stage as built is the top row; the lower rows "
                    "force objects together to show where the choice "
                    "matters.")
    assert rows[0][3] == "0.6016" and rows[3][3] == "0.0000"


# =====================================================================
# Measurement 4 -- the assignment rule is the design decision.
# =====================================================================
def table_rules():
    rows = [
        ["centre  (YOLOv1)", "1.00", "0.8458", "0.8653", "-0.0196",
         "0.04"],
        ["many  (anchor-style)", "5.25", "0.4422", "0.3547", "+0.0874",
         "1.25"],
        ["hungarian  (DETR)", "1.00", "0.7438", "0.7650", "-0.0212",
         "0.07"],
    ]

    def gain_colour(v, _r=0):
        return GOOD if v.startswith("+") else BAD

    def ap_colour(v, _r=0):
        return GOOD if float(v) > 0.8 else (
            BAD if float(v) < 0.5 else NEUTRAL)

    render("rules_table.png",
           "Same network, same loss, same budget \u2014 only the "
           "assignment rule changes",
           ["Assignment rule", "Positives\nper object", "AP\n(NMS on)",
            "AP\n(NMS off)", "What NMS\nis worth", "Duplicates\nper image"],
           rows, [6.6, 3.4, 3.4, 3.4, 3.4, 3.6],
           ["l", "c", "c", "c", "c", "c"],
           [TEXT, NEUTRAL, ap_colour, ap_colour, gain_colour, NOTE_C],
           footnote="5,130 parameters, 1000 steps, 3 seeds, 250 test "
                    "scenes. AP spread across seeds was at most "
                    "\u00b10.0272. Suppression repays only the rule "
                    "that manufactures duplicates.")
    assert rows[1][4] == "+0.0874", "the NMS gain for `many` is the point"


# =====================================================================
# Measurement 5 + 7 -- attributing every miss, and confidence quality.
# =====================================================================
# ranking AUC, from the frozen run (Measurement 7)
AUC = {"centre": 0.9930, "many": 0.9976, "hungarian": 0.9507}
TOP_IOU = {"centre": 0.7020, "many": 0.5954,
           "hungarian": 0.6973}


def table_errors():
    rows = [
        ["centre  (YOLOv1)", "0.7566", "0.1716", "0.0567", "0.0151",
         "0.9542"],
        ["many  (anchor-style)", "0.7358", "0.0000", "0.1752", "0.0889",
         "0.6276"],
        ["hungarian  (DETR)", "0.7114", "0.1544", "0.1128", "0.0213",
         "0.2681"],
    ]

    def found_c(v, _r=0):
        return GOOD if float(v) > 0.75 else NEUTRAL

    def fail_c(v, _r=0):
        return GOOD if float(v) < 0.02 else (
            BAD if float(v) > 0.12 else NEUTRAL)

    render("errors_table.png",
           "Where every object went",
           ["Assignment rule", "Found", "Nothing\nfired",
            "Box too\nfar off", "Wrong\nlabel",
            "Found by\nits owner"],
           rows, [6.4, 3.2, 3.2, 3.2, 3.2, 3.6],
           ["l", "c", "c", "c", "c", "c"],
           [TEXT, found_c, fail_c, fail_c, fail_c, NOTE_C],
           footnote="Columns 2\u20135 partition the objects "
                    "exactly (they sum to 1). 'Found by its "
                    "owner' is the share of hits that came from "
                    "the predictor the centre rule holds "
                    "answerable \u2014 the optimal match mostly "
                    "picks someone else.")
    s = sum(float(x) for x in rows[0][1:5])
    assert abs(s - 1.0) < 1e-9, "centre row must partition, got %f" % s


# =====================================================================
# Measurement 6 -- the suppression threshold.
# =====================================================================
def table_nms():
    rows = [
        ["centre  (YOLOv1)", "0.8154", "0.8359", "0.8413", "0.8485",
         "0.8563", "0.8563"],
        ["many  (anchor-style)", "0.4567", "0.4404", "0.4299", "0.3987",
         "0.3552", "0.3327"],
        ["hungarian  (DETR)", "0.7196", "0.7226", "0.7340", "0.7558",
         "0.7646", "0.7668"],
    ]
    def best_in_row(v, r):
        """Highlight the row's best value, ties included.

        A tie here is a real tie at the precision displayed, so
        highlighting one cell and not its equal would be misleading.
        """
        vals = [float(x) for x in rows[r][1:]]
        return GOOD if float(v) == max(vals) else TEXT

    render("nms_table.png",
           "Average precision against the suppression threshold",
           ["Assignment rule", "IoU 0.0\n(suppress\nmost)", "IoU 0.1",
            "IoU 0.3", "IoU 0.5", "IoU 0.7",
            "IoU 1.0\n(suppress\nnothing)"],
           rows, [6.4, 3.4, 3.0, 3.0, 3.0, 3.0, 3.4],
           ["l", "c", "c", "c", "c", "c", "c"],
           [TEXT] + [best_in_row] * 6,
           footnote="One trained detector per rule, re-scored; no "
                    "retraining. Best value per row in green. The "
                    "one-to-many rule wants the most aggressive "
                    "suppression available; the one-to-one rules want "
                    "none at all.")
    # The claim this table supports: the one-to-many rule is best
    # under the MOST aggressive suppression, and the one-to-one rules
    # are best with none.  Checked at the 4 decimals shown -- centre's
    # 0.7 and 1.0 columns tie at this precision (they differ further
    # out, which is why the script reports 1.0 as its peak).
    for i, r in enumerate(rows):
        vals = [float(x) for x in r[1:]]
        if i == 1:
            assert vals[0] == max(vals), "many should peak at IoU 0.0"
            assert vals[0] > vals[-1] + 0.1, "suppression must pay"
        else:
            assert vals[-1] == max(vals),                 "%s should peak unsuppressed" % r[0]
            assert vals[-1] > vals[0], "suppression must cost it"


if __name__ == "__main__":
    table_representation()
    table_rules()
    table_nms()
    table_errors()
