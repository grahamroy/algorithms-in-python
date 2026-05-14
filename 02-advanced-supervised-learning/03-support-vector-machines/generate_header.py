"""Generate the header image for the Support Vector Machines article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.svm import SVC


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLASS_BORDER = ['#3B82F6', '#DC2626']  # blue, red
CLASS_FILL = ['#dbeafe', '#fee2e2']
REGION_FILL = ['#eff6ff', '#fef2f2']

HYPERPLANE_COLOR = '#1F2937'
MARGIN_COLOR = '#7C3AED'  # purple
SV_RING_COLOR = '#16A34A'  # green


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'Support Vector Machines: Maximum Margin + Kernel Trick',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Linearly separable data: widen the gap. Non-linear data: replace inner products with a kernel.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 7.2, 6.0)
RIGHT_PANEL = (8.4, 0.9, 7.2, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# LEFT PANEL: linear SVM with margin on a linearly separable problem
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        'Linear: widest margin between classes',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Linearly separable blobs
X_lin, y_lin = make_blobs(n_samples=80,
                          centers=[[-1.6, -1.6], [1.6, 1.6]],
                          cluster_std=0.85, random_state=7)
svm_lin = SVC(kernel='linear', C=10).fit(X_lin, y_lin)

# Project to panel coords
x_min, x_max = X_lin[:, 0].min() - 0.5, X_lin[:, 0].max() + 0.5
y_min, y_max = X_lin[:, 1].min() - 0.5, X_lin[:, 1].max() + 0.5

plot_x0 = lpx + 0.55
plot_x1 = lpx + lpw - 0.55
plot_y0 = lpy + 0.55
plot_y1 = lpy + lph - 1.0


def to_panel(px, py):
    fx = (px - x_min) / (x_max - x_min)
    fy = (py - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)

# Decision regions (faint)
n_grid = 240
xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                     np.linspace(y_min, y_max, n_grid))
Z = svm_lin.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
from matplotlib.colors import ListedColormap
ax.imshow(Z,
          extent=(plot_x0, plot_x1, plot_y0, plot_y1),
          origin='lower', cmap=ListedColormap(REGION_FILL),
          alpha=0.5, aspect='auto', zorder=1)

# Hyperplane: w · x + b = 0; margins: w · x + b = ±1
w = svm_lin.coef_[0]
b = svm_lin.intercept_[0]

# Build lines parameterised in data space, then project
xs_data = np.linspace(x_min, x_max, 50)
# w0 x + w1 y + b = 0  -> y = -(w0 x + b) / w1
plane_y = -(w[0] * xs_data + b) / w[1]
margin_pos = -(w[0] * xs_data + b - 1) / w[1]
margin_neg = -(w[0] * xs_data + b + 1) / w[1]

def project_line(xs, ys):
    out_x, out_y = [], []
    for px, py in zip(xs, ys):
        if y_min <= py <= y_max:
            cx, cy = to_panel(px, py)
            out_x.append(cx)
            out_y.append(cy)
    return out_x, out_y

# Plot hyperplane and margins
hx, hy = project_line(xs_data, plane_y)
ax.plot(hx, hy, color=HYPERPLANE_COLOR, linewidth=2.0, zorder=2)
mx_p, my_p = project_line(xs_data, margin_pos)
ax.plot(mx_p, my_p, color=MARGIN_COLOR, linewidth=1.0,
        linestyle='--', zorder=2)
mx_n, my_n = project_line(xs_data, margin_neg)
ax.plot(mx_n, my_n, color=MARGIN_COLOR, linewidth=1.0,
        linestyle='--', zorder=2)

# Training points
for cls in (0, 1):
    pts = X_lin[y_lin == cls]
    coords = np.array([to_panel(*p) for p in pts])
    ax.scatter(coords[:, 0], coords[:, 1],
               s=42, c=CLASS_FILL[cls],
               edgecolors=CLASS_BORDER[cls],
               linewidths=1.0, zorder=3)

# Ring the support vectors
sv_coords = np.array([to_panel(*p) for p in svm_lin.support_vectors_])
ax.scatter(sv_coords[:, 0], sv_coords[:, 1],
           s=130, facecolors='none',
           edgecolors=SV_RING_COLOR, linewidths=2.0, zorder=4)

# Caption
ax.text(lpx + lpw/2, lpy + 0.3,
        'Support vectors (green) sit on the margin.',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: RBF SVM on moons
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'RBF kernel: curved boundary for free',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

X_m, y_m = make_moons(n_samples=200, noise=0.22, random_state=7)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='auto').fit(X_m, y_m)

x_min2, x_max2 = X_m[:, 0].min() - 0.4, X_m[:, 0].max() + 0.4
y_min2, y_max2 = X_m[:, 1].min() - 0.4, X_m[:, 1].max() + 0.4

plot_x0r = rpx + 0.55
plot_x1r = rpx + rpw - 0.55
plot_y0r = rpy + 0.55
plot_y1r = rpy + rph - 1.0


def to_panel_r(px, py):
    fx = (px - x_min2) / (x_max2 - x_min2)
    fy = (py - y_min2) / (y_max2 - y_min2)
    return plot_x0r + fx * (plot_x1r - plot_x0r), \
           plot_y0r + fy * (plot_y1r - plot_y0r)

xx2, yy2 = np.meshgrid(np.linspace(x_min2, x_max2, n_grid),
                       np.linspace(y_min2, y_max2, n_grid))
Z2 = svm_rbf.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)
ax.imshow(Z2,
          extent=(plot_x0r, plot_x1r, plot_y0r, plot_y1r),
          origin='lower', cmap=ListedColormap(REGION_FILL),
          alpha=0.55, aspect='auto', zorder=1)

for cls in (0, 1):
    pts = X_m[y_m == cls]
    coords = np.array([to_panel_r(*p) for p in pts])
    ax.scatter(coords[:, 0], coords[:, 1],
               s=26, c=CLASS_FILL[cls],
               edgecolors=CLASS_BORDER[cls],
               linewidths=0.9, zorder=3)

sv_coords_r = np.array([to_panel_r(*p) for p in svm_rbf.support_vectors_])
ax.scatter(sv_coords_r[:, 0], sv_coords_r[:, 1],
           s=70, facecolors='none',
           edgecolors=SV_RING_COLOR, linewidths=1.6, zorder=4)

# Caption
ax.text(rpx + rpw/2, rpy + 0.3,
        'Implicit non-linear feature space, no explicit embedding.',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Supervised Learning Part 3',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '02-advanced-supervised-learning/'
       '03-support-vector-machines/header_svm.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
