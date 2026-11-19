"""Data tables for the semantic segmentation article.

Medium collapses runs of whitespace, so every table ships as an
image.  All values are transcribed from one frozen run of segment.py
and each builder asserts the claim its table is there to support, so
a typo cannot pass quietly.
"""

from _renderer import (render, TEXT, NOTE_C, GOOD, BAD, NEUTRAL)


# =====================================================================
# Measurement 2 -- the comparison, and when it is safe to take it.
# =====================================================================
def table_curve():
    rows = [
        ["200", "0.1158 ±0.1098", "0.0887 ±0.0893",
         "+0.0271", "both still noise"],
        ["400", "0.4700 ±0.3218", "0.3265 ±0.2344",
         "+0.1435", "'skips are essential'"],
        ["800", "0.7151 ±0.3382", "0.8107 ±0.0458",
         "-0.0956", "'skips HURT'"],
        ["1600", "0.9563 ±0.0268", "0.8924 ±0.0103",
         "+0.0639", "the honest answer"],
    ]

    def gap_c(v, _r=0):
        return GOOD if v.startswith("+") else BAD

    render("curve_table.png",
           "The same comparison, read at four different budgets",
           ["Steps", "U-Net", "Encoder-decoder", "Gap",
            "What you would conclude"],
           rows, [3.0, 5.4, 5.4, 3.6, 7.4],
           ["c", "c", "c", "c", "l"],
           [NEUTRAL, TEXT, TEXT, gap_c, NOTE_C],
           footnote="Mean IoU, 3 seeds, ~9,000 parameters each, "
                    "identical optimiser and learning rate. At 400 "
                    "steps the U-Net's three seeds are 0.8398, 0.5149 "
                    "and 0.0554 — a spread of 0.7844, wider than "
                    "the final gap it is being used to measure.",
           dividers=(1,))
    assert rows[2][3].startswith("-"), "the 800-step reversal is the point"


# =====================================================================
# Measurement 3 -- where the difference lives.
# =====================================================================
def table_boundary():
    rows = [
        ["U-Net", "0.9657", "0.9819", "0.9876", "0.9886", "0.9907",
         "0.9951"],
        ["Encoder-decoder", "0.8841", "0.9807", "0.9891", "0.9923",
         "0.9930", "0.9944"],
        ["Difference", "+0.0816", "+0.0012", "-0.0015", "-0.0036",
         "-0.0022", "+0.0007"],
    ]

    def cell(v, r):
        if r < 2:
            return TEXT
        return GOOD if v == "+0.0816" else NOTE_C

    render("boundary_table.png",
           "Accuracy by exact distance from the nearest boundary",
           ["Architecture", "d = 0\n(touching)", "d = 1", "d = 2",
            "d = 3", "d = 4", "d = 5"],
           rows, [6.0, 3.6, 3.0, 3.0, 3.0, 3.0, 3.0],
           ["l", "c", "c", "c", "c", "c", "c"],
           [TEXT] + [cell] * 6,
           footnote="Foreground pixels only, 1600 steps, 3 seeds. The "
                    "difference at the boundary is +0.0816; the "
                    "largest anywhere else is 0.0036 — 23 times "
                    "smaller, and at three distances the plain "
                    "encoder-decoder is fractionally ahead.",
           dividers=(1,))
    gap0 = abs(float(rows[2][1]))
    rest = max(abs(float(v)) for v in rows[2][2:])
    assert gap0 > 20 * rest, "the whole claim is that d=0 dominates"


# =====================================================================
# Measurement 4 -- depth, receptive field, and what it costs.
# =====================================================================
def table_depth():
    rows = [
        ["1", "18", "9,097", "9 px", "0.8800 ±0.0220",
         "0.9766", "0.9512", "0.9321", "0.6742"],
        ["2", "8", "9,991", "21 px", "0.9563 ±0.0268",
         "0.9872", "0.9870", "0.9939", "0.9803"],
        ["3", "4", "7,043", "45 px", "0.7424 ±0.1701",
         "0.8892", "0.8136", "0.8914", "0.7999"],
        ["3", "5", "10,962", "45 px", "0.4399 ±0.3422",
         "0.6185", "0.6250", "0.6583", "0.6484"],
    ]

    def rim_c(v, r):
        try:
            x = float(v)
        except ValueError:
            return TEXT
        return GOOD if x > 0.95 else (BAD if x < 0.75 else NEUTRAL)

    render("depth_table.png",
           "Depth buys reach, and is paid for in width",
           ["Depth", "Width", "Params", "Receptive\nfield",
            "mean IoU", "Rim\nr=4", "Rim\nr=5", "Rim\nr=6",
            "Rim\nr=7"],
           rows, [3.0, 3.0, 3.6, 3.6, 5.2, 3.0, 3.0, 3.0, 3.0],
           ["c", "c", "c", "c", "c", "c", "c", "c", "c"],
           [NEUTRAL, NEUTRAL, NOTE_C, NOTE_C, TEXT] + [rim_c] * 4,
           footnote="'Rim' is accuracy on the outer pixels of a "
                    "PIERCED shape, whose label depends on a hole up "
                    "to r pixels away. Depth 3 cannot reach 9,000 "
                    "parameters with uniform widths, so it is run at "
                    "both bracketing sizes — and loses at both.",
           dividers=(2,))
    assert float(rows[0][5]) - float(rows[0][8]) > 0.25, \
        "depth 1 must degrade with radius"
    assert float(rows[1][8]) > 0.95, "depth 2 must not"


if __name__ == "__main__":
    table_curve()
    table_boundary()
    table_depth()
