#!/usr/bin/env python3
"""
Q-planner paper -- Figure 1: system architecture.

Drawn programmatically with matplotlib, in the same style as Figure 2, so the
figure is reproducible from code.

Deliberately complementary to Figure 2 rather than a redundant view of it:
    Figure 1 (this one) -- WHERE things run: the client/remote split, what
                           crosses the boundary, and where LLM context lives.
    Figure 2            -- WHEN things happen: the phase-by-phase workflow trace.

The central claim it must make: all LLM context is confined to the client and to
three bounded call sites. Workflow state and the typed artifact store live on the
client; ORCA input/output files and raw outputs stay on the cluster. Everything
crossing the boundary is a structured tool call or a structured result, never a
transcript, and the reporter sees only the structured artifacts.

Usage: python figure_architecture.py [OUTPUT_DIR]   (default: current directory)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Palette (identical to figure_workflow.py so the two figures read as a set) ─
BG        = '#FFFFFF'
C_CLIENT  = '#EAF2FB'
C_REMOTE  = '#EAF6EC'
C_BOX     = '#FFFFFF'
C_TOOL    = '#DDF0E1'
C_SKILL   = '#FAD9AE'
C_STORE   = '#DCEAF7'

E_CLIENT  = '#2E5F8A'
E_REMOTE  = '#2E7D4F'
E_LLM     = '#C0392B'
E_PLAN    = '#B5651D'
EDGE      = '#2C3E50'
DARK      = '#1C2833'
GRAY      = '#4D5656'   # darker than before, for contrast at print size
ABLUE     = '#1A5276'

# Every label is drawn through txt(); its size is multiplied by FONT_SCALE so that the
# smallest label prints at >= ~6 pt. PRINT_SCALE = printed size / drawn size in the paper.
FONT_SCALE = 1.25
PRINT_SCALE = 0.756

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})


def rbox(ax, x, y, w, h, fc=C_BOX, ec=EDGE, lw=1.0, r=0.008, z=2, ls='-'):
    rr = min(r, w / 4, h / 4)
    ax.add_patch(FancyBboxPatch(
        (x + rr, y + rr), max(w - 2 * rr, 1e-3), max(h - 2 * rr, 1e-3),
        boxstyle=f'round,pad={rr}', facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=z, linestyle=ls))


def txt(ax, x, y, s, fs=8.5, c=DARK, ha='center', va='center',
        bold=False, italic=False, z=4):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs * FONT_SCALE, color=c,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal', zorder=z)


def arr(ax, p0, p1, c=EDGE, lw=1.3, ls='-', z=3, rad=0.0, both=False):
    style = '<|-|>' if both else '-|>'
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, color=c, lw=lw,
        linestyle=ls, zorder=z, shrinkA=0, shrinkB=0,
        connectionstyle=f'arc3,rad={rad}'))


def poly(ax, pts, c=EDGE, lw=1.3, ls='-', z=3):
    """Orthogonal polyline with one arrowhead, for routes that would otherwise
    cut through a box."""
    ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
            color=c, lw=lw, linestyle=ls, zorder=z, solid_capstyle='round')
    arr(ax, pts[-2], pts[-1], c=c, lw=lw, ls=ls, z=z)


def box2(ax, cx, cy, w, h, l1, l2=None, fc=C_BOX, ec=EDGE, lw=1.0,
         fs1=8.5, fs2=7.0):
    rbox(ax, cx - w / 2, cy - h / 2, w, h, fc=fc, ec=ec, lw=lw)
    if l2:
        txt(ax, cx, cy + 0.011, l1, fs=fs1, bold=True)
        txt(ax, cx, cy - 0.013, l2, fs=fs2, c=GRAY, italic=True)
    else:
        txt(ax, cx, cy, l1, fs=fs1, bold=True)


fig = plt.figure(figsize=(8.4, 7.2), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_facecolor(BG)

PX, PW = 0.022, 0.956
LDX, LDY = 0.020, 0.040          # same label inset convention as Figure 2

# ── Client panel ──────────────────────────────────────────────────────────────
CY, CH = 0.360, 0.618
rbox(ax, PX, CY, PW, CH, fc=C_CLIENT, ec=E_CLIENT, lw=1.3, z=1)
txt(ax, PX + LDX, CY + CH - LDY, 'CLIENT', fs=11.0, c=E_CLIENT, ha='left', bold=True)
txt(ax, PX + LDX + 0.100, CY + CH - LDY, 'local workstation — all LLM context lives here',
    fs=8.0, c=E_CLIENT, ha='left', italic=True)

box2(ax, 0.140, 0.892, 0.190, 0.052, 'User', 'natural-language request')

# three bounded LLM call sites
# Token costs are deliberately not annotated here: they vary by workflow and are
# reported per workflow in Figure 2 and Table 1.
box2(ax, 0.400, 0.892, 0.240, 0.052, 'Compound Identification',
     'PubChem / OPSIN', ec=E_LLM, lw=1.6, fs1=8.2, fs2=7.0)
box2(ax, 0.400, 0.790, 0.240, 0.052, 'Planner LLM',
     'single bounded call', ec=E_LLM, lw=1.6, fs1=8.2, fs2=7.0)
# Reporter sits under the user, on the left, keeping the centre channel clear
# for the two arrows that cross the MCP boundary.
box2(ax, 0.140, 0.470, 0.200, 0.052, 'Reporter LLM',
     'structured artifacts only', ec=E_LLM, lw=1.6, fs1=8.2, fs2=7.0)

# skill library feeds the planner
rbox(ax, 0.660, 0.748, 0.300, 0.128, fc=C_SKILL, ec=E_PLAN, lw=1.2)
txt(ax, 0.810, 0.857, 'Skill Library', fs=8.6, bold=True)
txt(ax, 0.810, 0.835, 'calibrated domain protocols', fs=6.8, c=GRAY, italic=True)
for j, s in enumerate(['method & basis  ·  pKa calibration',
                       'thermochemistry  ·  microsolvation',
                       'NBO  ·  TD-DFT  ·  PES / TS']):
    txt(ax, 0.810, 0.814 - j * 0.0195, s, fs=6.7, c=DARK)
txt(ax, 0.810, 0.758, 'skills may own deterministic tools', fs=6.5, c=E_PLAN, italic=True)
arr(ax, (0.660, 0.800), (0.520, 0.796), c=E_PLAN, lw=1.1)

# validator + executor
box2(ax, 0.400, 0.690, 0.240, 0.048, 'Plan Validator',
     'structure · deps · arg schemas', fs1=8.2, fs2=6.6)
box2(ax, 0.400, 0.580, 0.240, 0.056, 'LangGraph Executor',
     'deterministic DAG · no LLM in loop', fs1=8.2, fs2=6.6)
box2(ax, 0.735, 0.580, 0.190, 0.056, 'Artifact Store',
     'typed key–value results', fc=C_STORE, fs1=8.2, fs2=6.6)

arr(ax, (0.235, 0.892), (0.280, 0.892))                       # user -> compound id
arr(ax, (0.400, 0.866), (0.400, 0.816))                       # compound id -> planner
arr(ax, (0.400, 0.764), (0.400, 0.714))                       # planner -> validator
arr(ax, (0.400, 0.666), (0.400, 0.608))                       # validator -> executor
arr(ax, (0.520, 0.580), (0.640, 0.580))                       # executor -> artifacts
arr(ax, (0.640, 0.556), (0.245, 0.482))                       # artifacts -> reporter (straight: clears the executor box)
arr(ax, (0.140, 0.496), (0.140, 0.866))                       # reporter -> user
txt(ax, 0.152, 0.690, 'report', fs=7.2, c=GRAY, ha='left', italic=True)

# retry loop, consistent with Figure 2
arr(ax, (0.310, 0.714), (0.310, 0.764), c=E_LLM, lw=1.1, rad=0.0)   # validator top -> planner bottom
txt(ax, 0.298, 0.739, 'bounded retry', fs=6.6, c=E_LLM, ha='right', italic=True)

# ── Boundary ──────────────────────────────────────────────────────────────────
BY = 0.320
for xa, xb in ((PX, 0.330), (0.670, PX + PW)):
    ax.plot([xa, xb], [BY, BY], color=EDGE, lw=1.4, ls=(0, (6, 4)), zorder=3)
# The box sits above the dashed boundary (z=4) so the line does not run through its text.
rbox(ax, 0.330, BY - 0.026, 0.340, 0.052, fc='#FFFFFF', ec=EDGE, lw=1.2, z=4)
txt(ax, 0.500, BY + 0.008, 'MCP over SSH', fs=8.8, bold=True, z=5)
txt(ax, 0.500, BY - 0.010, 'structured tool calls and results only', fs=6.8, c=GRAY,
    italic=True, z=5)

# executor -> remote, and remote -> artifact store, both clear of every box
arr(ax, (0.380, 0.552), (0.380, BY + 0.028), c=ABLUE, lw=1.3)
poly(ax, [(0.520, BY + 0.028), (0.520, 0.512), (0.790, 0.512), (0.790, 0.552)],
     c=E_REMOTE, lw=1.3)
txt(ax, 0.368, 0.420, 'tool calls', fs=7.2, c=ABLUE, ha='right', italic=True)
txt(ax, 0.532, 0.420, 'results', fs=7.2, c=E_REMOTE, ha='left', italic=True)

txt(ax, PX + PW - 0.010, BY + 0.012, 'no transcript crosses this boundary',
    fs=7.2, c=EDGE, ha='right', italic=True)
txt(ax, PX + PW - 0.010, BY - 0.014, 'raw outputs stay on the cluster',
    fs=7.2, c=EDGE, ha='right', italic=True)

# ── Remote panel ──────────────────────────────────────────────────────────────
RY, RH = 0.028, 0.262
rbox(ax, PX, RY, PW, RH, fc=C_REMOTE, ec=E_REMOTE, lw=1.3, z=1)
txt(ax, PX + LDX, RY + RH - LDY, 'REMOTE COMPUTE', fs=11.0, c=E_REMOTE, ha='left', bold=True)
txt(ax, PX + LDX + 0.245, RY + RH - LDY, 'Kubernetes / SLURM cluster',
    fs=8.0, c=E_REMOTE, ha='left', italic=True)

rbox(ax, 0.060, 0.132, 0.880, 0.062, fc='#FFFFFF', ec=E_REMOTE, lw=1.1)
txt(ax, 0.500, 0.176, 'FastMCP tool server', fs=8.4, bold=True)
txt(ax, 0.500, 0.158, 'eleven quantum-chemical tools; argument schemas published to the client',
    fs=6.8, c=GRAY, italic=True)
txt(ax, 0.500, 0.142, 'run_sp_energy · run_opt_job · run_freq_job · run_nbo_job · run_tddft_job · …',
    fs=6.4, c=DARK)

BACKENDS = ['ORCA 6.1.1', 'xTB', 'SOLVATOR', 'OPI', 'PubChem', 'RDKit']
bw, bgap = 0.132, 0.014
bx = 0.060
for nm in BACKENDS:
    rbox(ax, bx, 0.062, bw, 0.036, fc=C_TOOL, ec=E_REMOTE, lw=0.9)
    txt(ax, bx + bw / 2, 0.080, nm, fs=7.2)
    bx += bw + bgap
arr(ax, (0.500, 0.132), (0.500, 0.100), c=E_REMOTE, lw=1.1)

# ── Legend ────────────────────────────────────────────────────────────────────
rbox(ax, 0.640, 0.372, 0.310, 0.052, fc='#FFFFFF', ec='#B0B0B0', lw=0.9)
rbox(ax, 0.656, 0.396, 0.034, 0.014, fc='#FFFFFF', ec=E_LLM, lw=1.6, r=0.004)
txt(ax, 0.700, 0.403, 'red border = LLM call (three in total)', fs=7.0, ha='left')
rbox(ax, 0.656, 0.378, 0.034, 0.014, fc='#FFFFFF', ec=EDGE, lw=1.0, r=0.004)
txt(ax, 0.700, 0.385, 'plain border = deterministic code', fs=7.0, ha='left')

if os.environ.get('FIG_CHECK'):
    from figure_layout_check import check
    check(fig, ax, PRINT_SCALE)

out = os.path.join(sys.argv[1] if len(sys.argv) > 1 else '.', 'architecture_drawn')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight', facecolor=BG)
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor=BG)
print(f'Saved:\n  {out}.png\n  {out}.pdf')
