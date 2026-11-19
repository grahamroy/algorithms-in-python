"""Render the segmentation article's data tables as PNGs.

Medium collapses runs of whitespace, so a monospaced ASCII table
pasted into it loses its columns.  Every table therefore ships as an
image.  All values below are transcribed from a single frozen run of
segment.py; the asserts at the bottom of each builder are there so a
typo cannot survive quietly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = ("D:/Projects/Medium/algorithms-in-python/"
       "14-computer-vision/03-semantic-segmentation/")

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


