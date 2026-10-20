#!/usr/bin/env python3
"""
Q-planner paper -- Figure 2: workflow trace for the pKa application example.

Drawn programmatically with matplotlib so the figure is reproducible from code.

What the figure shows, and why it is drawn this way:
  * Pre-planning: only the method_selection and pka_calibration skills fire for
    a pKa request, so only those chips are highlighted.
  * Planning: the JSON plan reaches execution only through the plan validator;
    a failing plan goes back to the planner for a bounded retry.
  * Execution: the executed pKa plan. Per compound, OPT then FREQ for the acid
    (HA) and its conjugate base (A-), then one deterministic
    compute_pka_calibrated node, which also reads the ethanal reference. No llm
    or calc_expr node occurs in this workflow; the node-kind key in the planning
    panel lists the kinds the plan schema allows, not nodes of this plan.
  * OPT(x) feeds only FREQ(x). Remote ORCA jobs are grouped in one outline with
    a single MCP connection.

Token annotations are the means of the three logged pKa runs reported in the
paper (Table 1 / SI Table S1): planner ~10k (10,269 / 10,168 / 10,201) and
reporter ~4k (4,196 / 3,728 / 4,149); compute_pka_calibrated uses no tokens.
Compound identification is not annotated because it was not measured.

Layout rule: the three phase labels sit at an identical inset from the top-left
corner of their own panel and are left-aligned into one vertical line; LABEL_DY
also reserves the band beneath each label so no content collides with it.

Usage: python figure_workflow.py [OUTPUT_DIR]   (default: current directory)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = '#FFFFFF'
C_PRE       = '#EAF2FB'
C_PLAN      = '#FDF1E3'
C_EXEC      = '#EAF6EC'
C_BOX       = '#FFFFFF'
C_TOOL      = '#DDF0E1'
C_CALC      = '#DCEAF7'
C_LLMN      = '#FCE9E7'
C_SKILL_ON  = '#FAD9AE'
C_SKILL_OFF = '#E8E8E8'

E_PRE  = '#2E5F8A'
E_PLAN = '#B5651D'
E_EXEC = '#2E7D4F'
E_LLM  = '#C0392B'
EDGE   = '#2C3E50'
DARK   = '#1C2833'
GRAY   = '#4D5656'   # darker than before, for contrast at print size
ABLUE  = '#1A5276'

# Every label is drawn through txt(); its size is multiplied by FONT_SCALE so that the
# smallest label prints at >= ~6 pt. PRINT_SCALE = printed size / drawn size in the paper.
FONT_SCALE = 1.4
PRINT_SCALE = 0.679

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc=C_BOX, ec=EDGE, lw=1.0, r=0.006, z=2, ls='-'):
    rr = min(r, w / 4, h / 4)
    ax.add_patch(FancyBboxPatch(
        (x + rr, y + rr), max(w - 2 * rr, 1e-3), max(h - 2 * rr, 1e-3),
        boxstyle=f'round,pad={rr}', facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=z, linestyle=ls))


def txt(ax, x, y, s, fs=8.0, c=DARK, ha='center', va='center',
        bold=False, italic=False, z=4):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs * FONT_SCALE, color=c,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal', zorder=z)


def arr(ax, p0, p1, c=EDGE, lw=1.2, ls='-', z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=10, color=c, lw=lw,
        linestyle=ls, zorder=z, shrinkA=0, shrinkB=0,
        connectionstyle=f'arc3,rad={rad}'))


def poly(ax, pts, c=EDGE, lw=1.2, ls='-', z=3):
    """Orthogonal polyline with a single arrowhead on the last segment.
    Used where a straight or curved arrow would cut through a box."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=c, lw=lw, linestyle=ls,
            zorder=z, solid_capstyle='round')
    arr(ax, pts[-2], pts[-1], c=c, lw=lw, ls=ls, z=z)


def box2(ax, cx, cy, w, h, line1, line2=None, fc=C_BOX, ec=EDGE, lw=1.0,
         fs1=8.0, fs2=6.9, r=0.007):
    """Centred box with a bold first line and an optional grey second line."""
    rbox(ax, cx - w / 2, cy - h / 2, w, h, fc=fc, ec=ec, lw=lw, r=r)
    if line2:
        txt(ax, cx, cy + 0.0085, line1, fs=fs1, bold=True)
        txt(ax, cx, cy - 0.0095, line2, fs=fs2, c=GRAY, italic=True)
    else:
        txt(ax, cx, cy, line1, fs=fs1, bold=True)


# ── Canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.4, 10.4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_facecolor(BG)

PX, PW = 0.022, 0.956
LDX, LDY = 0.020, 0.026          # label inset; identical in all three panels

P1_Y, P1_H = 0.706, 0.286        # Pre-planning
P2_Y, P2_H = 0.436, 0.252        # Planning
P3_Y, P3_H = 0.016, 0.402        # Execution

for num, name, y0, hgt, fc, ec in [
        ('PHASE 1', 'Pre-planning', P1_Y, P1_H, C_PRE,  E_PRE),
        ('PHASE 2', 'Planning',     P2_Y, P2_H, C_PLAN, E_PLAN),
        ('PHASE 3', 'Execution',    P3_Y, P3_H, C_EXEC, E_EXEC)]:
    rbox(ax, PX, y0, PW, hgt, fc=fc, ec=ec, lw=1.3, r=0.008, z=1)
    lx, ly = PX + LDX, y0 + hgt - LDY
    txt(ax, lx, ly, num, fs=11.0, c=ec, ha='left', bold=True)
    txt(ax, lx + 0.115, ly, name, fs=11.0, c=ec, ha='left', bold=True)

# ══════════════════════════════════════════════════════════════════ PHASE 1 ═══
R1, R2, R3 = 0.930, 0.858, 0.775          # three content rows, all below label

box2(ax, 0.145, R1, 0.180, 0.046, 'User Request', 'natural language')
box2(ax, 0.470, R1, 0.310, 0.052, 'Compound Identification LLM', 'PubChem / OPSIN',
     ec=E_LLM, lw=1.5, fs1=7.8)

box2(ax, 0.145, R2, 0.180, 0.042, 'Skill Dispatcher', 'keyword matching')
box2(ax, 0.470, R2, 0.310, 0.046, '2D Structure Confirmation (User)', None, fs1=7.8)
box2(ax, 0.800, R2, 0.230, 0.046, 'Compound Cards',
     'formula · SMILES · charge · CID', fs2=6.4)

# Skill chips in a single horizontal row (the vertical stack wasted the panel).
# Matches the logged pKa runs: only method_selection and pka_calibration fire.
SKILLS = [('Method & Basis', 1), ('pKa', 1), ('Thermochemistry', 0),
          ('NBO', 0), ('TD-DFT', 0)]
cw, gap = 0.152, 0.011
x0 = 0.062
for nm, on in SKILLS:
    rbox(ax, x0, R3 - 0.014, cw, 0.028,
         fc=C_SKILL_ON if on else C_SKILL_OFF,
         ec=E_PLAN if on else '#B0B0B0', lw=0.9, r=0.005)
    txt(ax, x0 + cw / 2, R3, nm, fs=7.0, c=DARK if on else '#8A8A8A')
    x0 += cw + gap
txt(ax, 0.875, R3 - 0.030, 'orange = skill activated for this request',
    fs=6.6, c=GRAY, ha='right', italic=True)

arr(ax, (0.235, R1), (0.315, R1))                       # request -> compound id
arr(ax, (0.145, R1 - 0.023), (0.145, R2 + 0.021))       # request -> dispatcher
arr(ax, (0.470, R1 - 0.026), (0.470, R2 + 0.023))       # compound id -> confirm
arr(ax, (0.625, R2), (0.685, R2))                       # confirm -> cards
arr(ax, (0.145, R2 - 0.021), (0.145, R3 + 0.016))       # dispatcher -> chips

# Compound cards and the activated skills both become planner inputs: merge them
# into one labelled hand-off rather than two arrows landing at arbitrary points.
MERGE_Y = 0.722
CARD_X = 0.895                   # right of the last skill chip, so the line crosses no chip
ax.plot([CARD_X, CARD_X], [R2 - 0.023, MERGE_Y], color=ABLUE, lw=1.3, zorder=3)
ax.plot([0.320, 0.320], [R3 - 0.014, MERGE_Y], color=ABLUE, lw=1.3, zorder=3)
ax.plot([0.320, CARD_X], [MERGE_Y, MERGE_Y], color=ABLUE, lw=1.3, zorder=3)
arr(ax, (0.560, MERGE_Y), (0.560, P2_Y + P2_H - 0.004), c=ABLUE, lw=1.3)
txt(ax, 0.572, 0.697, 'compound cards + active skills',
    fs=6.6, c=ABLUE, ha='left', italic=True)

# ══════════════════════════════════════════════════════════════════ PHASE 2 ═══
PLAN_TOP = P2_Y + P2_H - LDY - 0.024      # first usable row under the label

INPUTS = ['Orchestration\nSystem Prompt', 'Active Skill\nModules',
          'Geometry State\nSummary', 'Compound Cards']
iy = PLAN_TOP - 0.012
for s in INPUTS:
    rbox(ax, 0.046, iy - 0.018, 0.175, 0.036, r=0.005)
    txt(ax, 0.1335, iy, s, fs=7.2)
    arr(ax, (0.221, iy), (0.288, 0.516), rad=0.0, lw=1.0)
    iy -= 0.043

# taller than its neighbours: three lines (title, role, measured tokens)
rbox(ax, 0.395 - 0.1025, 0.481, 0.205, 0.072, ec=E_LLM, lw=1.5, r=0.007)
txt(ax, 0.395, 0.534, 'Planner LLM', fs=8.6, bold=True)
txt(ax, 0.395, 0.5155, 'single bounded call', fs=6.9, c=GRAY, italic=True)
txt(ax, 0.395, 0.497, '~10k tokens', fs=7.0, c=E_LLM, italic=True)

box2(ax, 0.630, 0.516, 0.175, 0.054, 'JSON Plan', 'nodes[ ] · settings', fs1=8.0, fs2=6.5)

# Plan validator -- the only route from planning into execution.
VX, VY, VW, VH = 0.745, 0.480, 0.205, 0.078
rbox(ax, VX, VY, VW, VH, ec=E_PLAN, lw=1.4)
txt(ax, VX + VW / 2, VY + VH - 0.016, 'Plan Validator', fs=8.2, bold=True)
txt(ax, VX + VW / 2, VY + VH - 0.030, 'deterministic', fs=6.5, c=GRAY, italic=True)
for j, chk in enumerate(['structure', 'dependency graph', 'tool argument schemas']):
    txt(ax, VX + 0.014, VY + VH - 0.044 - j * 0.0135, '· ' + chk,
        fs=6.5, c=DARK, ha='left')

arr(ax, (0.4975, 0.516), (0.5425, 0.516))
arr(ax, (0.7175, 0.516), (VX, 0.516))

# bounded-retry loop: validator -> planner, routed BELOW the row so it does not
# cut through the JSON Workflow Plan box.
RETRY_Y = 0.462   # below the taller Planner box (bottom 0.481)
poly(ax, [(VX + 0.030, VY), (VX + 0.030, RETRY_Y),
          (0.470, RETRY_Y), (0.470, 0.481)], c=E_LLM, lw=1.2)   # turn up right of the legend chips
txt(ax, 0.598, RETRY_Y + 0.008, 'failed checks — bounded retry',
    fs=7.0, c=E_LLM, italic=True)
txt(ax, 0.598, RETRY_Y - 0.008, 'no compute dispatched', fs=6.5, c=GRAY, italic=True)

# valid -> execution
arr(ax, (VX + VW / 2, VY), (VX + VW / 2, P3_Y + P3_H - 0.004), c=E_EXEC, lw=1.5)
txt(ax, VX + VW / 2 + 0.012, 0.456, 'valid', fs=7.4, c=E_EXEC, ha='left', bold=True)

# plan node-kind key (schema legend, not nodes of this workflow) -- inside panel
txt(ax, 0.046, 0.4525, 'plan node kinds:', fs=6.6, c=GRAY, ha='left', italic=True)
for j, (nm, fc, ec) in enumerate([('tool', C_TOOL, E_EXEC),
                                   ('calc_expr', C_CALC, ABLUE),
                                   ('llm', C_LLMN, E_LLM)]):
    x = 0.195 + j * 0.090
    rbox(ax, x, 0.4445, 0.082, 0.017, fc=fc, ec=ec, lw=0.9, r=0.004)
    txt(ax, x + 0.041, 0.4530, nm, fs=6.5)

# ══════════════════════════════════════════════════════════════════ PHASE 3 ═══
EXEC_TOP = P3_Y + P3_H - LDY - 0.020
txt(ax, PX + LDX, EXEC_TOP, 'DAG execution of the validated plan',
    fs=7.2, c=GRAY, ha='left', italic=True)

# The executed pKa plan, per target compound: OPT then FREQ for the acid (HA) and
# its conjugate base (A-), then one deterministic compute_pka_calibrated node that
# also reads the reference acid's free energies. No llm or calc_expr node occurs
# in this workflow (checked against the logged plans).
NW, NH = 0.150, 0.046
YA, YB = 0.320, 0.252            # HA / A- rows for one target compound
YC = 0.172                       # reference acid row

# remote ORCA jobs are grouped in one dashed outline with a single MCP connection
GX0, GX1, GY0, GY1 = 0.036, 0.384, 0.126, 0.352
rbox(ax, GX0, GY0, GX1 - GX0, GY1 - GY0, fc='none', ec=E_EXEC, lw=1.0, ls='--',
     r=0.006, z=1.5)
txt(ax, GX0 + 0.010, GY0 + 0.009, 'remote ORCA jobs, run in parallel',
    fs=6.3, c=E_EXEC, ha='left', italic=True)

box2(ax, 0.118, YA, NW, NH, 'OPT', 'HA (compound i)', fc=C_TOOL, ec=E_EXEC)
box2(ax, 0.118, YB, NW, NH, 'OPT', r'A$^{-}$ (compound i)', fc=C_TOOL, ec=E_EXEC)
box2(ax, 0.300, YA, NW, NH, 'FREQ', 'HA (compound i)', fc=C_TOOL, ec=E_EXEC)
box2(ax, 0.300, YB, NW, NH, 'FREQ', r'A$^{-}$ (compound i)', fc=C_TOOL, ec=E_EXEC)
arr(ax, (0.193, YA), (0.225, YA))
arr(ax, (0.193, YB), (0.225, YB))
txt(ax, 0.209, YB - 0.034, r'A$^{-}$ geometry from the client-side deprotonation tool',
    fs=6.3, c=GRAY, italic=True)

box2(ax, 0.209, YC, 0.330, NH, 'OPT + FREQ', r'HA and A$^{-}$ of the reference acid (ethanal)',
     fc=C_TOOL, ec=E_EXEC)

# deterministic pKa tool: one node per target compound
PKX, PKY, PKW, PKH = 0.4375, 0.243, 0.165, 0.086
rbox(ax, PKX, PKY, PKW, PKH, fc=C_TOOL, ec=E_EXEC)
txt(ax, PKX + PKW / 2, PKY + PKH - 0.020, 'compute_pKa', fs=8.0, bold=True)
txt(ax, PKX + PKW / 2, PKY + PKH - 0.040, 'deterministic', fs=6.6, c=GRAY,
    italic=True)
txt(ax, PKX + PKW / 2, PKY + 0.014, '0 tokens', fs=6.6, c=E_EXEC, italic=True)
arr(ax, (0.375, YA), (PKX, YA))                       # FREQ HA -> compute_pKa
arr(ax, (0.375, YB), (PKX, YB))                       # FREQ A- -> compute_pKa
poly(ax, [(0.374, YC), (PKX + PKW / 2, YC), (PKX + PKW / 2, PKY)])   # reference

# right column: MCP, artifact store, reporter, report
MX, MW = 0.735, 0.222
rbox(ax, MX, 0.234, MW, 0.138, fc='#FFFFFF', ec=E_EXEC, lw=1.2, ls='--')
txt(ax, MX + MW / 2, 0.356, 'MCP over SSH', fs=7.8, bold=True)
txt(ax, MX + MW / 2, 0.342, r'$\rightarrow$ Remote Pod', fs=7.8, bold=True)
for j, nm in enumerate(['ORCA 6.1.1', 'OPI', 'SOLVATOR', 'xTB']):
    rbox(ax, MX + 0.028, 0.322 - j * 0.0205 - 0.017, MW - 0.056, 0.017,
         fc=C_TOOL, ec=E_EXEC, lw=0.7, r=0.003)
    txt(ax, MX + MW / 2, 0.3135 - j * 0.0205, nm, fs=6.3)

# the grouped remote jobs reach the pod over MCP; routed above compute_pKa
MCP_Y = 0.341
arr(ax, (GX1, MCP_Y), (MX, MCP_Y), c=E_EXEC, lw=0.9, ls='--')
txt(ax, 0.668, MCP_Y + 0.008, 'remote tool calls', fs=6.5, c=E_EXEC, italic=True)

box2(ax, MX + MW / 2, 0.180, MW, 0.064, 'Artifact Store', 'G_*_eh · pka_*', fs1=8.2, fs2=6.6)
arr(ax, (MX + MW / 2, 0.234), (MX + MW / 2, 0.212), c=E_EXEC)      # results
txt(ax, MX + MW / 2 + 0.008, 0.223, 'results', fs=6.5, c=E_EXEC, ha='left', italic=True)
arr(ax, (PKX + PKW, PKY + 0.020), (MX, 0.193))                     # pKa values

rbox(ax, MX, 0.098 - 0.029, MW, 0.058, ec=E_LLM, lw=1.5, r=0.007)
txt(ax, MX + MW / 2, 0.098 + 0.010, 'Reporter LLM', fs=8.2, bold=True)
txt(ax, MX + MW / 2, 0.098 - 0.013, '~4k tokens', fs=6.8, c=E_LLM, italic=True)
arr(ax, (MX + MW / 2, 0.148), (MX + MW / 2, 0.127))

box2(ax, MX + MW / 2, 0.042, MW, 0.036, 'Markdown Report', None, fs1=8.0)
arr(ax, (MX + MW / 2, 0.069), (MX + MW / 2, 0.060))

# legend -- two categories, every box in the figure belongs to one
LX, LY, LW, LH = 0.046, 0.036, 0.372, 0.070
rbox(ax, LX, LY, LW, LH, fc='#FFFFFF', ec='#B0B0B0', lw=0.9)
rbox(ax, LX + 0.016, LY + 0.043, 0.040, 0.016, fc='#FFFFFF', ec=E_LLM, lw=1.5, r=0.004)
txt(ax, LX + 0.066, LY + 0.051, 'red border = LLM call', fs=7.0, ha='left')
rbox(ax, LX + 0.016, LY + 0.020, 0.040, 0.016, fc='#FFFFFF', ec=EDGE, lw=1.0, r=0.004)
txt(ax, LX + 0.066, LY + 0.028, 'plain border = deterministic code', fs=7.0, ha='left')
txt(ax, LX + 0.016, LY + 0.008, 'solid = data flow    dashed = remote tool call',
    fs=6.3, c=GRAY, ha='left', italic=True)

if os.environ.get('FIG_CHECK'):
    from figure_layout_check import check
    check(fig, ax, PRINT_SCALE)

out = os.path.join(sys.argv[1] if len(sys.argv) > 1 else '.', 'workflow_drawn')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight', facecolor=BG)
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor=BG)
print(f'Saved:\n  {out}.png\n  {out}.pdf')
