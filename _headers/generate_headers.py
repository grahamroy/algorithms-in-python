"""
Generate header images for Medium articles:
  1. Linear Regression           — scatter + best-fit line
  2. Arrays                      — row of indexed cells
  3. Matrices                    — 2D grid of cells
  4. Tensors                     — stacked 3D grid suggesting depth
  5. Running AI on Your Machine  — neural network on a local chip

Light theme (white background) to match the earlier maths-series articles.
Output: 1600x900 PNGs in the same directory.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection

sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------
# Shared palette & helpers  (LIGHT theme)
# ------------------------------------------------------------------
BG       = "#ffffff"     # pure white background
PANEL    = "#f5f7fa"     # very light panel tint
CELL     = "#eef2f7"     # cell fill
CELL_ED  = "#cbd5e0"     # cell edge — soft grey
GRIDLINE = "#e2e8f0"     # grid lines inside panels
ACCENT   = "#ff7a45"     # warm orange — the "thing being tracked"
ACCENT2  = "#f6ad3b"     # amber — secondary highlight
TEXT     = "#0f1724"     # near-black, matches title colour
MUTED    = "#64748b"     # slate grey — subtitles & labels

WIDTH  = 1600
HEIGHT = 900
DPI    = 100
FIGSIZE = (WIDTH / DPI, HEIGHT / DPI)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def new_fig():
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_axis_off()
    return fig, ax


def draw_title(ax, title, subtitle):
    ax.text(
        0.85, 7.8, title,
        fontsize=44, color=TEXT, weight="bold",
        family="DejaVu Sans", va="center",
    )
    ax.text(
        0.85, 7.0, subtitle,
        fontsize=20, color=MUTED, style="italic",
        family="DejaVu Sans", va="center",
    )


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {name}  ({os.path.getsize(path)/1024:.1f} KB)")


# ------------------------------------------------------------------
# 1. Linear Regression — scatter + best-fit line
# ------------------------------------------------------------------
def linear_regression():
    fig, ax = new_fig()
    draw_title(ax, "Linear Regression",
               "The simplest model that isn't useless")

    # Data area
    x0, y0 = 1.0, 1.0
    w, h   = 14.0, 5.2

    # Background panel
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=PANEL,
                           edgecolor=CELL_ED, linewidth=1, zorder=1))

    # Faint axis lines
    for frac in [0.25, 0.5, 0.75]:
        ax.plot([x0, x0 + w], [y0 + h*frac, y0 + h*frac],
                color=GRIDLINE, lw=0.8, zorder=2)
        ax.plot([x0 + w*frac, x0 + w*frac], [y0, y0 + h],
                color=GRIDLINE, lw=0.8, zorder=2)

    # Scatter points around y = 0.35*x + noise
    rng = np.random.default_rng(7)
    n = 40
    xs = rng.uniform(0.05, 0.95, n)
    ys = 0.2 + 0.7 * xs + rng.normal(0, 0.07, n)
    ys = np.clip(ys, 0.02, 0.98)

    ax.scatter(x0 + xs * w, y0 + ys * h,
               s=85, color=ACCENT, alpha=0.9, edgecolors=BG,
               linewidths=1.2, zorder=3)

    # Best-fit line
    slope, intercept = np.polyfit(xs, ys, 1)
    line_x = np.array([0.02, 0.98])
    line_y = slope * line_x + intercept
    ax.plot(x0 + line_x * w, y0 + line_y * h,
            color=ACCENT2, lw=4, zorder=4)

    save(fig, "01_linear_regression.png")


# ------------------------------------------------------------------
# 2. Arrays — row of indexed cells
# ------------------------------------------------------------------
def arrays():
    fig, ax = new_fig()
    draw_title(ax, "Arrays",
               "Where every algorithm begins")

    # 10 cells centred horizontally
    n = 10
    cell_w = 1.2
    cell_h = 1.6
    gap    = 0.12
    total_w = n * cell_w + (n - 1) * gap
    x_start = (16 - total_w) / 2
    y_start = 2.6

    # Random-ish values — use fixed seed so it's repeatable
    rng = np.random.default_rng(3)
    vals = rng.integers(10, 99, size=n)

    highlight = 4  # highlight one cell to show indexing

    for i in range(n):
        x = x_start + i * (cell_w + gap)
        fill = ACCENT if i == highlight else CELL
        edge = ACCENT2 if i == highlight else CELL_ED
        ax.add_patch(Rectangle((x, y_start), cell_w, cell_h,
                               facecolor=fill, edgecolor=edge,
                               linewidth=2, zorder=2))
        # Value inside the cell
        ax.text(x + cell_w/2, y_start + cell_h/2 + 0.05,
                f"{vals[i]}",
                ha="center", va="center", fontsize=24,
                color=TEXT if i != highlight else BG,
                weight="bold", family="DejaVu Sans")
        # Index below the cell
        ax.text(x + cell_w/2, y_start - 0.45, f"[{i}]",
                ha="center", va="center", fontsize=15,
                color=MUTED, family="DejaVu Sans")

    # Small arrow pointing to highlighted cell
    hx = x_start + highlight * (cell_w + gap) + cell_w/2
    ax.annotate("", xy=(hx, y_start + cell_h + 0.05),
                xytext=(hx, y_start + cell_h + 0.9),
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=3))
    ax.text(hx, y_start + cell_h + 1.15, "O(1) access",
            ha="center", va="bottom", color=ACCENT2,
            fontsize=15, family="DejaVu Sans", style="italic")

    save(fig, "02_arrays.png")


# ------------------------------------------------------------------
# 3. Matrices — 2D grid of cells
# ------------------------------------------------------------------
def matrices():
    fig, ax = new_fig()
    draw_title(ax, "Matrices",
               "The language machine learning thinks in")

    rows, cols = 4, 6
    cell = 1.05
    gap  = 0.08
    total_w = cols * cell + (cols - 1) * gap
    total_h = rows * cell + (rows - 1) * gap
    x_start = (16 - total_w) / 2
    y_start = 1.2

    rng = np.random.default_rng(11)
    values = rng.integers(0, 10, size=(rows, cols))

    for r in range(rows):
        for c in range(cols):
            x = x_start + c * (cell + gap)
            # y inverted so row 0 is top
            y = y_start + (rows - 1 - r) * (cell + gap)
            ax.add_patch(Rectangle((x, y), cell, cell,
                                   facecolor=CELL,
                                   edgecolor=CELL_ED,
                                   linewidth=1.5, zorder=2))
            ax.text(x + cell/2, y + cell/2, f"{values[r, c]}",
                    ha="center", va="center", fontsize=22,
                    color=TEXT, weight="bold", family="DejaVu Sans")

    # Highlight one row and one column to show dot-product intuition
    hr = 1
    hc = 3
    # Row highlight
    for c in range(cols):
        x = x_start + c * (cell + gap)
        y = y_start + (rows - 1 - hr) * (cell + gap)
        ax.add_patch(Rectangle((x, y), cell, cell,
                               facecolor="none",
                               edgecolor=ACCENT, linewidth=3, zorder=3))
    # Column highlight
    for r in range(rows):
        x = x_start + hc * (cell + gap)
        y = y_start + (rows - 1 - r) * (cell + gap)
        ax.add_patch(Rectangle((x, y), cell, cell,
                               facecolor="none",
                               edgecolor=ACCENT2, linewidth=3, zorder=3))
    # Intersection cell filled
    xi = x_start + hc * (cell + gap)
    yi = y_start + (rows - 1 - hr) * (cell + gap)
    ax.add_patch(Rectangle((xi, yi), cell, cell,
                           facecolor=ACCENT,
                           edgecolor=ACCENT2, linewidth=3, zorder=4))
    ax.text(xi + cell/2, yi + cell/2, f"{values[hr, hc]}",
            ha="center", va="center", fontsize=22,
            color=BG, weight="bold", family="DejaVu Sans", zorder=5)

    save(fig, "03_matrices.png")


# ------------------------------------------------------------------
# 4. Tensors — 3 stacked grids to suggest the 3rd dimension
# ------------------------------------------------------------------
def tensors():
    fig, ax = new_fig()
    draw_title(ax, "Tensors",
               "The native data format of deep learning")

    # Three 4x4 grids, offset to suggest depth
    rows, cols = 4, 4
    cell = 0.85
    gap  = 0.08
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = rows * cell + (rows - 1) * gap

    # Depth offset per layer
    dx, dy = 0.55, 0.35
    layers = 4
    # Centre the whole stack
    total_w = grid_w + (layers - 1) * dx
    total_h = grid_h + (layers - 1) * dy
    base_x = (16 - total_w) / 2
    base_y = 1.4

    # Draw back-to-front. Back layer L = layers-1 (deepest offset),
    # front layer L = 0 (should be on TOP in z-order).
    for L in range(layers - 1, -1, -1):
        ox = base_x + L * dx
        oy = base_y + L * dy
        # Higher z for front layers (L small → high z)
        z_base = (layers - L) * 4
        # A subtle panel behind this layer
        ax.add_patch(Rectangle((ox - 0.15, oy - 0.15),
                               grid_w + 0.3, grid_h + 0.3,
                               facecolor=PANEL,
                               edgecolor=CELL_ED, linewidth=1,
                               zorder=z_base + 1))
        for r in range(rows):
            for c in range(cols):
                x = ox + c * (cell + gap)
                y = oy + (rows - 1 - r) * (cell + gap)
                # Front layer gets accent splashes on a few cells
                is_highlight = (L == 0 and (r, c) in
                                {(0, 1), (1, 3), (2, 0), (3, 2)})
                fill = ACCENT if is_highlight else CELL
                edge = ACCENT2 if is_highlight else CELL_ED
                # Back layers slightly desaturated (via alpha)
                alpha = 1.0 if L == 0 else 0.55 + 0.15 * (layers - 1 - L) / (layers - 1)
                ax.add_patch(Rectangle((x, y), cell, cell,
                                       facecolor=fill,
                                       edgecolor=edge,
                                       linewidth=1.2,
                                       alpha=alpha,
                                       zorder=z_base + 2))

    # Dimension labels (N, H, W, C hint) — small & subtle
    # Width arrow along the front bottom
    fx0 = base_x
    fx1 = base_x + grid_w
    fy  = base_y - 0.55
    ax.annotate("", xy=(fx1, fy), xytext=(fx0, fy),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.8))
    ax.text((fx0 + fx1) / 2, fy - 0.35, "W",
            ha="center", va="center", color=MUTED, fontsize=14,
            family="DejaVu Sans", style="italic")

    # Height arrow up the left side
    hx = base_x - 0.55
    hy0 = base_y
    hy1 = base_y + grid_h
    ax.annotate("", xy=(hx, hy1), xytext=(hx, hy0),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.8))
    ax.text(hx - 0.3, (hy0 + hy1) / 2, "H",
            ha="center", va="center", color=MUTED, fontsize=14,
            family="DejaVu Sans", style="italic", rotation=90)

    # Depth arrow along the diagonal
    dx0 = base_x + grid_w + 0.15
    dy0 = base_y + grid_h + 0.15
    dx1 = dx0 + (layers - 1) * dx
    dy1 = dy0 + (layers - 1) * dy
    ax.annotate("", xy=(dx1, dy1), xytext=(dx0, dy0),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.8))
    ax.text(dx1 + 0.2, dy1 + 0.1, "N",
            ha="left", va="center", color=MUTED, fontsize=14,
            family="DejaVu Sans", style="italic")

    save(fig, "04_tensors.png")


# ------------------------------------------------------------------
# 5. Running AI on Your Own Machine — neural net inside a CPU chip
# ------------------------------------------------------------------
def running_ai():
    fig, ax = new_fig()
    draw_title(ax, "Running AI on Your Own Machine",
               "Local inference, zero cloud")

    # CPU chip outline — big rounded rectangle centred in the lower half
    chip_w = 9.6
    chip_h = 4.8
    chip_x = (16 - chip_w) / 2
    chip_y = 0.9

    # Pins on each side — small rectangles sticking out
    pin_len = 0.22
    pin_w   = 0.14
    pin_gap = 0.42

    def draw_pins(x, y, direction, count):
        # direction: 'top', 'bottom', 'left', 'right'
        for i in range(count):
            if direction in ('top', 'bottom'):
                px = x + (i + 0.5) * pin_gap
                py = y - pin_len if direction == 'bottom' else y
                w, h = pin_w, pin_len
            else:
                py = y + (i + 0.5) * pin_gap
                px = x - pin_len if direction == 'left' else x
                w, h = pin_len, pin_w
            ax.add_patch(Rectangle((px, py), w, h,
                                   facecolor=CELL_ED,
                                   edgecolor="none", zorder=1))

    pins_side_tb = int(chip_w / pin_gap) - 1
    pins_side_lr = int(chip_h / pin_gap) - 1
    # Top pins
    tb_start_x = chip_x + (chip_w - pins_side_tb * pin_gap) / 2
    draw_pins(tb_start_x, chip_y + chip_h, 'top', pins_side_tb)
    draw_pins(tb_start_x, chip_y, 'bottom', pins_side_tb)
    # Left / right pins
    lr_start_y = chip_y + (chip_h - pins_side_lr * pin_gap) / 2
    draw_pins(chip_x, lr_start_y, 'left', pins_side_lr)
    draw_pins(chip_x + chip_w, lr_start_y, 'right', pins_side_lr)

    # Chip body (rounded rect)
    chip = FancyBboxPatch(
        (chip_x, chip_y), chip_w, chip_h,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        facecolor=PANEL, edgecolor=CELL_ED, linewidth=2,
        zorder=2)
    ax.add_patch(chip)

    # Inner die — a smaller rounded rect inside the chip
    die_pad = 0.35
    die = FancyBboxPatch(
        (chip_x + die_pad, chip_y + die_pad),
        chip_w - 2 * die_pad, chip_h - 2 * die_pad,
        boxstyle="round,pad=0.01,rounding_size=0.15",
        facecolor=BG, edgecolor=CELL_ED, linewidth=1.2,
        zorder=3)
    ax.add_patch(die)

    # Small "AI" label in the corner of the die
    ax.text(chip_x + chip_w - die_pad - 0.25,
            chip_y + chip_h - die_pad - 0.25,
            "AI",
            ha="right", va="top", fontsize=13,
            color=MUTED, weight="bold",
            family="DejaVu Sans", zorder=4)

    # ---- Neural network INSIDE the die ----
    # 3 layers: input (5), hidden (7), output (3)
    layer_sizes = [5, 7, 3]
    # Layout area within the die (inset a bit)
    nn_x0 = chip_x + die_pad + 0.55
    nn_y0 = chip_y + die_pad + 0.55
    nn_w  = chip_w - 2 * die_pad - 1.1
    nn_h  = chip_h - 2 * die_pad - 1.1

    # Compute x positions for each layer
    xs = np.linspace(nn_x0, nn_x0 + nn_w, len(layer_sizes))
    node_r = 0.17

    # Store node positions
    positions = []
    for Li, size in enumerate(layer_sizes):
        if size == 1:
            ys = [nn_y0 + nn_h / 2]
        else:
            ys = np.linspace(nn_y0 + 0.2, nn_y0 + nn_h - 0.2, size)
        positions.append([(xs[Li], y) for y in ys])

    # Draw edges between consecutive layers
    for Li in range(len(layer_sizes) - 1):
        for (x1, y1) in positions[Li]:
            for (x2, y2) in positions[Li + 1]:
                ax.plot([x1, x2], [y1, y2],
                        color=CELL_ED, lw=0.8,
                        alpha=0.8, zorder=4)

    # Highlight a single activation path from one input to one output
    highlight_path = [
        (0, 2),  # layer 0, node 2
        (1, 3),  # layer 1, node 3
        (2, 1),  # layer 2, node 1
    ]
    for i in range(len(highlight_path) - 1):
        L1, n1 = highlight_path[i]
        L2, n2 = highlight_path[i + 1]
        x1, y1 = positions[L1][n1]
        x2, y2 = positions[L2][n2]
        ax.plot([x1, x2], [y1, y2],
                color=ACCENT, lw=3, zorder=5)

    # Draw nodes on top of edges
    for Li, layer in enumerate(positions):
        for ni, (x, y) in enumerate(layer):
            is_path = (Li, ni) in set(highlight_path)
            fill = ACCENT if is_path else CELL
            edge = ACCENT2 if is_path else CELL_ED
            ax.add_patch(Circle((x, y), node_r,
                                facecolor=fill,
                                edgecolor=edge,
                                linewidth=1.8,
                                zorder=6))

    # Layer labels below the chip
    labels = ["input", "hidden", "output"]
    for i, label in enumerate(labels):
        ax.text(xs[i], chip_y - 0.55, label,
                ha="center", va="center",
                color=MUTED, fontsize=13,
                style="italic", family="DejaVu Sans")

    save(fig, "05_running_ai.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating headers in {OUT_DIR}")
    linear_regression()
    arrays()
    matrices()
    tensors()
    running_ai()
    print("done.")
