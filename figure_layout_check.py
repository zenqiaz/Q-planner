"""Layout check for the hand-placed matplotlib figures (figure_architecture.py, figure_workflow.py).

Run a figure script with FIG_CHECK=1 to print every label that crosses a box edge, overlaps
another label, or is crossed by a line or arrow, plus the smallest font size at print scale.
Placement in these figures is by hand, so this replaces checking collisions by eye.
"""
from typing import List

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

PAD = 1.0  # px tolerance
MARGIN = 8.0  # px a label must keep from the side of the box that holds it
MARGIN_V = 4.0  # px, the same for top/bottom edges and for boxes the label sits next to


def _shrink(b: Bbox, d: float = PAD) -> Bbox:
    return Bbox.from_extents(b.x0 + d, b.y0 + d, b.x1 - d, b.y1 - d)


def _inside(inner: Bbox, outer: Bbox) -> bool:
    return (inner.x0 >= outer.x0 - PAD and inner.x1 <= outer.x1 + PAD and
            inner.y0 >= outer.y0 - PAD and inner.y1 <= outer.y1 + PAD)


def _polyline_points(xy: np.ndarray, step: float = 2.0) -> np.ndarray:
    pts = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(pts)


def check(fig, ax, print_scale: float) -> List[str]:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, _shrink(t.get_window_extent(r))) for t in ax.texts if t.get_text().strip()]
    boxes = [p.get_window_extent(r) for p in ax.patches if isinstance(p, FancyBboxPatch)]
    # dashed outlines are group containers that lines are meant to cross
    solid_boxes = [p.get_window_extent(r) for p in ax.patches
                   if isinstance(p, FancyBboxPatch) and p.get_linestyle() in ('-', 'solid')]
    lines = []
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch):
            path = p.get_path()                       # display coordinates after draw()
            lines.append(_polyline_points(path.vertices))
    for ln in ax.lines:
        if isinstance(ln, Line2D):
            lines.append(_polyline_points(ax.transData.transform(ln.get_xydata())))
    problems = []
    for i, (t, b) in enumerate(texts):
        label = t.get_text().replace('\n', ' ')[:32]
        for bx in boxes:
            if b.overlaps(bx) and not _inside(b, bx) and not _inside(bx, b):
                problems.append(f'crosses a box edge : "{label}"')
                break
            # inside a box but closer than MARGIN to its left/right edge reads as touching it
            if _inside(b, bx) and bx.width < fig.bbox.width * 0.6 and (
                    b.x0 - bx.x0 < MARGIN or bx.x1 - b.x1 < MARGIN or
                    b.y0 - bx.y0 < MARGIN_V or bx.y1 - b.y1 < MARGIN_V):
                problems.append(f'touches a box edge : "{label}"')
                break
        # a label just outside a box (e.g. a panel title above a box) reads as touching it
        for bx in boxes:
            near = Bbox.from_extents(bx.x0 - MARGIN_V, bx.y0 - MARGIN_V, bx.x1 + MARGIN_V, bx.y1 + MARGIN_V)
            if not b.overlaps(bx) and b.overlaps(near) and not _inside(bx, b):
                problems.append(f'almost touches box : "{label}"')
                break
        for t2, b2 in texts[i + 1:]:
            if b.overlaps(b2):
                problems.append(f'overlaps label     : "{label}" / "{t2.get_text()[:24]}"')
        for pts in lines:
            if np.any((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)):
                problems.append(f'line through label : "{label}"')
                break
    # a line running through (or along the edge of) a small box it does not start or end at
    for k, pts in enumerate(lines):
        for bx in solid_boxes:
            if bx.width > fig.bbox.width * 0.6:
                continue                                   # panels legitimately contain lines
            inner = Bbox.from_extents(bx.x0 - 2, bx.y0 - 2, bx.x1 + 2, bx.y1 + 2)
            core = Bbox.from_extents(bx.x0 + 4, bx.y0 + 4, bx.x1 - 4, bx.y1 - 4)
            hit = (pts[:, 0] > inner.x0) & (pts[:, 0] < inner.x1) & (pts[:, 1] > inner.y0) & (pts[:, 1] < inner.y1)
            ends = np.zeros(len(pts), bool); ends[:6] = ends[-6:] = True
            if np.any(hit & ~ends) and not np.all(hit):
                cx, cy = (bx.x0 + bx.x1) / 2, (bx.y0 + bx.y1) / 2
                problems.append(f'line touches a box : box centred at ({cx:.0f}, {cy:.0f}) px, line {k}')
    smallest = min(t.get_fontsize() for t, _ in texts)
    print(f'layout check: {len(texts)} labels, {len(problems)} problem(s); '
          f'smallest font {smallest:.1f} pt in file = {smallest * print_scale:.1f} pt printed')
    for p in problems:
        print('  ' + p)
    return problems
