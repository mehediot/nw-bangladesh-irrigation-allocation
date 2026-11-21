#!/usr/bin/env python3
"""Stage 8 - figures of the paper, drawn from the outputs of stages 1-7.

No numbers are computed here except simple shares for plotting. Each figure is saved as PNG (600 dpi),
TIFF (600 dpi, LZW) and PDF (vector) in generated/figures/, 190 mm wide, named by its number in the paper.

Main text
  Fig. 1  study area and district context (five maps)
  Fig. 2  framework: stages of the analysis, with small charts drawn from the pipeline's own outputs
  Fig. 3  feasibility: Boro share of water demand against the budget ratio, smallest Boro cut
  Fig. 4  plans: land use incl. fallow, response to the budget, energy-net-return frontier
  Fig. 5  budget needed under Gamma-robust water requirements
  Fig. 6  AWD adoption and the fallow; model value of water against irrigation charges
  Fig. 7  six-crop variant: fallow and land use as lentil and mustard expand
  Fig. 8  one-factor-at-a-time sensitivities at budget 0.80
Supporting Information
  Fig. S1 crop data: yield trends, cropped area, yield-predictor correlations
  Fig. S2 SHAP values of the hindcast GBT (descriptive)
  Fig. S3 RAWSI components, score and rank range over 20 variants
  Fig. S4 decision-time forecasts: observed vs forecast, skill, error distribution
  Fig. S5 potato price and the net-return plan
  Fig. S6 decision value of the forecasts, and maps of dual values

Use --only Fig3 FigS4 ... to draw selected figures (the notebook draws each one in its own section).

Colour rules: crops always Boro = blue, wheat = orange, potato = green, maize = yellow; forecast models
violet / yellow / pink; ordered quantities use one hue from light to dark. Hatching and marker shapes
repeat every colour distinction so the figures survive greyscale printing.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patheffects import withStroke

cli = argparse.ArgumentParser(description="Stage 8 - figures")
cli.add_argument("--input-dir", type=Path, required=True)
cli.add_argument("--generated-dir", type=Path, required=True)
cli.add_argument("--output-dir", type=Path, required=True)
cli.add_argument("--only", nargs="+", metavar="FIG", help="figures to draw, e.g. Fig1 FigS4 (default: all)")
args = cli.parse_args()
I, G, O = args.input_dir, args.generated_dir, args.output_dir
O.mkdir(parents=True, exist_ok=True)
X, T, D, R, F = G / "extensions", G / "tables", G / "decision", G / "hindcast", G / "forecasts"

INK, MUTED, GRID, LIGHT = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
CROPS = ["Boro Rice", "Wheat", "Potato"]  # yield models
CROPS_A = CROPS + ["Maize"]  # allocation model
CROP_LABEL = {"Boro Rice": "Boro rice", "Wheat": "Wheat", "Potato": "Potato", "Maize": "Maize"}
CROP_COL = {"Boro Rice": "#2a78d6", "Wheat": "#eb6834", "Potato": "#1baf7a", "Maize": "#eda100"}
CROP_HATCH = {"Boro Rice": "", "Wheat": "////", "Potato": "....", "Maize": "xxxx"}
LAND_COLS = [
    "boro_pct_of_land",
    "wheat_pct_of_land",
    "potato_pct_of_land",
    "maize_pct_of_land",
    "fallow_pct_of_land",
]
MODEL_COL = {"Elastic Net": "#6250d6", "Persistence": "#eda100", "Climatology": "#e87ba4"}
MODEL_HATCH = {"Elastic Net": "", "Persistence": "////", "Climatology": "...."}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUES)
PURPLES = LinearSegmentedColormap.from_list("purple_seq", ["#efedfb", "#c9c2f2", "#9b8fe6", "#6250d6", "#3b2d8f"])
W = 7.48  # inches = 190 mm

plt.rcParams.update(
    {
        "font.family": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "hatch.linewidth": 0.5,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def fmt_minus(v, nd=2):
    """Number as text with a true minus sign."""
    return f"{v:.{nd}f}".replace("-", "\u2212")


MIN_PT = 6.5  # smallest text at print size (190 mm wide)


def save(fig, name):
    """Raise any text below MIN_PT, then write PNG, TIFF and PDF."""
    for t in fig.findobj(matplotlib.text.Text):  # no text below MIN_PT anywhere in the figure
        if t.get_text() and t.get_fontsize() < MIN_PT:
            t.set_fontsize(MIN_PT)
    fig.savefig(O / f"{name}.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        O / f"{name}.tif", bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"}
    )
    fig.savefig(O / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel(ax, letter, title=""):
    """Bold panel label, e.g. '(a) Title'."""
    ax.set_title(f"({letter}) {title}" if title else f"({letter})", loc="left", fontweight="bold", pad=5)


# ---------------------------------------------------------------------------------------------------
# Map data and helpers (Fig. 1 and Fig. S6)
# ---------------------------------------------------------------------------------------------------
geo = json.loads((I / "bgd_districts_geoboundaries_adm2_simplified.geojson").read_text())
RENAME = {"Bogra": "Bogura", "Nawabganj": "Chapainawabganj"}
shapes = {}
for f in geo["features"]:
    name = RENAME.get(f["properties"]["shapeName"], f["properties"]["shapeName"])
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    shapes[name] = [np.asarray(p[0])[:, :2] for p in polys]  # outer rings
study = set(pd.read_csv(I / "analysis_panel_full.csv").district)  # the 16 study districts
assert study <= set(shapes) and len(shapes) == 64


def centroid(rings):
    r = max(rings, key=len)
    x, y = r[:, 0], r[:, 1]
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    return np.array([((x[:-1] + x[1:]) * a).sum(), ((y[:-1] + y[1:]) * a).sum()]) / (3 * a.sum())


cen = {d: centroid(shapes[d]) for d in study}
ORANGES = LinearSegmentedColormap.from_list(
    "orange_seq", ["#fdebe0", "#f7b996", "#ef8250", "#d4551f", "#9c3a10", "#6b2608"]
)
GREENS = LinearSegmentedColormap.from_list(
    "green_seq", ["#dff4ea", "#a4dfc3", "#5fc79a", "#1baf7a", "#11805a", "#0a5a3f"]
)
X0, X1, Y0, Y1 = 87.95, 90.05, 23.75, 26.70
SHIFT = {
    "Lalmonirhat": (0.30, 0.17),
    "Nilphamari": (-0.06, 0.0),
    "Kurigram": (0.03, -0.13),
    "Chapainawabganj": (0.10, -0.02),
    "Joypurhat": (0.0, 0.03),
    "Naogaon": (-0.05, -0.03),
    "Rangpur": (0.0, -0.07),
    "Thakurgaon": (-0.03, 0.0),
    "Rajshahi": (-0.03, 0.0),
}
halo = [withStroke(linewidth=1.8, foreground="white")]


def choropleth(ax, values, cmap, norm, extent=(X0, X1, Y0, Y1), lw=0.6):
    for name, rings in shapes.items():
        if name in study:
            ax.add_collection(
                PolyCollection(
                    rings, facecolors=cmap(norm(values[name])), edgecolors="white", linewidths=lw, zorder=2
                )
            )
        else:
            ax.add_collection(
                PolyCollection(rings, facecolors="#ecebe7", edgecolors="white", linewidths=0.3, zorder=1)
            )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(25.0)))


def small_map(ax, values, cmap, vmin, vmax, title, label, fmt="{:.2f}"):
    norm = Normalize(vmin, vmax)
    choropleth(ax, values, cmap, norm, lw=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for d in sorted(study):
        ax.text(
            cen[d][0],
            cen[d][1],
            fmt.format(values[d]),
            ha="center",
            va="center",
            fontsize=5,
            color=INK,
            path_effects=[withStroke(linewidth=1.3, foreground="white")],
            zorder=5,
        )
    ax.set_title(title, loc="left", fontweight="bold", pad=3, fontsize=7.5)
    cax = ax.inset_axes([0.08, -0.07, 0.84, 0.035])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=6, pad=1.5)
    cb.set_label(label, fontsize=6.3, labelpad=1.5)


def fig1():
    """Fig. 1 - study area and district context."""
    boro = pd.read_csv(X / "ext_10_district_boro_share.csv")
    cells = pd.read_csv(X / "ext_01_climate_cells.csv")
    share = boro.set_index("district").boro_water_share
    t3m = pd.read_csv(T / "TableS4_rawsi.csv").set_index("district")
    e6m = pd.read_csv(X / "ext_06_pm_deficit_district.csv").set_index("district")
    base_area = pd.read_csv(R / "09_baseline_coefficients.csv").pivot(
        index="district", columns="crop", values="area0"
    )
    other_share = 100 * (base_area["Wheat"] + base_area["Potato"]) / base_area.sum(axis=1)
    fig = plt.figure(figsize=(W, 6.3))
    axb = fig.add_axes([0.055, 0.10, 0.46, 0.84])
    norm = Normalize(0.5, 1.0)  # Boro share runs from 0.53 (Panchagarh) to 0.93
    choropleth(axb, share, SEQ, norm, extent=(87.42, X1, Y0, Y1))
    axb.set_xticks([88.0, 88.5, 89.0, 89.5, 90.0])
    axb.set_xticklabels([f"{v:.1f}°E" for v in axb.get_xticks()])
    axb.set_yticks([24.0, 24.5, 25.0, 25.5, 26.0, 26.5])
    axb.set_yticklabels([f"{v:.1f}°N" for v in axb.get_yticks()])
    axb.tick_params(length=2)
    for cell, grp in cells.groupby("climate_cell"):  # districts sharing one climate series
        ds = grp.district.tolist()
        for a, b in zip(ds[:-1], ds[1:]):
            axb.plot(
                [cen[a][0], cen[b][0]],
                [cen[a][1], cen[b][1]],
                color=MUTED,
                lw=0.8,
                ls=(0, (3, 2)),
                marker="o",
                ms=2.2,
                zorder=4,
            )
    for d in sorted(study):
        lx, ly = cen[d] + np.array(SHIFT.get(d, (0, 0)))
        if np.hypot(*SHIFT.get(d, (0, 0))) > 0.2:
            axb.plot([cen[d][0], lx - 0.05], [cen[d][1], ly - 0.03], color=INK, lw=0.5, zorder=5)
        axb.text(
            lx,
            ly + 0.035,
            d,
            ha="center",
            va="bottom",
            fontsize=6.3,
            zorder=6,
            color=INK,
            fontweight="bold",
            path_effects=halo,
        )
        axb.text(
            lx,
            ly + 0.02,
            f"{share[d]:.2f}",
            ha="center",
            va="top",
            fontsize=6.2,
            zorder=6,
            color=INK,
            path_effects=halo,
        )
    km = 50 / (111.32 * np.cos(np.radians(24.0)))
    axb.plot([89.35, 89.35 + km], [23.83, 23.83], color=INK, lw=1.5, zorder=6)
    axb.text(89.35 + km / 2, 23.86, "50 km", ha="center", va="bottom", fontsize=6.5)
    axb.annotate(
        "N",
        xy=(89.95, 26.6),
        xytext=(89.95, 26.38),
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1),
    )
    ins = axb.inset_axes([0.0, 0.0, 0.215, 0.30])  # locator
    for name, rings in shapes.items():
        ins.add_collection(
            PolyCollection(
                rings,
                facecolors="#6da7ec" if name in study else "#d9d8d3",
                edgecolors="white",
                linewidths=0.2,
            )
        )
    ins.set_xlim(87.9, 92.8)
    ins.set_ylim(20.5, 26.75)
    ins.set_aspect(1 / np.cos(np.radians(23.7)))
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.5)
    ins.set_facecolor("white")
    ins.text(90.9, 20.9, "Bangladesh", fontsize=5.4, color=MUTED, ha="center")
    cax = axb.inset_axes([0.36, -0.105, 0.60, 0.022])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=6.5)
    cb.set_label("Boro share of modelled irrigation demand", fontsize=6.8, labelpad=2)
    axb.plot(
        [], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), marker="o", ms=2.2, label="Same NASA POWER climate series"
    )
    axb.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), fontsize=6.3, handlelength=2.6)
    panel(axb, "a", "Study districts and Boro dependence")
    spec = [
        ("b", t3m.RAWSI, ORANGES, 0.15, 0.85, "RAWSI scarcity index", "relative score (0–1)", "{:.2f}"),
        (
            "c",
            t3m.groundwater_component,
            ORANGES,
            0,
            1,
            "Groundwater-trend component",
            "normalised (0–1)",
            "{:.2f}",
        ),
        (
            "d",
            e6m.pm_deficit_boro_mm,
            GREENS,
            680,
            860,
            "FAO-56 Boro requirement",
            "mm, 2023–2025 mean",
            "{:.0f}",
        ),
        ("e", other_share, PURPLES, 0, 55, "Wheat + potato share of area", "% of three-crop area", "{:.0f}"),
    ]
    for k, (letter, vals, cmap, vmin, vmax, title, label, fmt) in enumerate(spec):
        ax = fig.add_axes([0.555 + 0.225 * (k % 2), 0.575 - 0.495 * (k // 2), 0.215, 0.375])
        small_map(ax, vals, cmap, vmin, vmax, f"({letter}) {title}", label, fmt)
    save(fig, "Fig1_study_area")


def fig2():
    """Fig. 2 - framework."""
    t3m = pd.read_csv(T / "TableS4_rawsi.csv").set_index("district")
    # Drawn in millimetres (1 unit = 1 mm at 190 mm width). The small charts inside the boxes are
    # drawn from the pipeline's own outputs, so the diagram cannot drift away from the results.
    from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
    from matplotlib.path import Path as MPath

    NAVY, BAND, CARD_EC, VIOLET = "#104281", "#f4f3ef", "#c9c8c2", MODEL_COL["Elastic Net"]
    ROWS = [
        ("1", "Data", 27),
        ("2", "Study\ndesign", 30),
        ("3", "Models", 39),
        ("4", "Optimisation", 40),
        ("5", "Decision\nsupport", 25),
    ]
    GAP = 5.5
    FH = sum(h for *_, h in ROWS) + GAP * (len(ROWS) - 1) + 1
    fig = plt.figure(figsize=(W, W * FH / 190))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 190)
    ax.set_ylim(0, FH)
    ax.axis("off")
    ytop, ROWY = FH - 0.5, {}
    for num, label, h in ROWS:  # stage tabs and bands
        y0 = ytop - h
        ROWY[num] = (y0, ytop)
        ax.add_patch(
            FancyBboxPatch((0.8, y0), 20, h, boxstyle="round,pad=0,rounding_size=1.6", fc=NAVY, ec="none")
        )
        ax.add_patch(Circle((10.8, ytop - 6), 3.3, fc="white", ec="none"))
        ax.text(10.8, ytop - 6.15, num, ha="center", va="center", fontsize=9, fontweight="bold", color=NAVY)
        ax.text(
            10.8,
            (y0 + ytop - 9.5) / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="white",
            linespacing=1.2,
        )
        ax.add_patch(
            FancyBboxPatch((23.5, y0), 166, h, boxstyle="round,pad=0,rounding_size=1.6", fc=BAND, ec="none")
        )
        ytop = y0 - GAP

    def card(x, y, w, h, title=None, fc="white", ec=CARD_EC):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1", fc=fc, ec=ec, lw=0.6)
        )
        if title:
            ax.text(x + 2.4, y + h - 2.2, title, ha="left", va="top", fontsize=7.3, fontweight="bold")

    def note(x, y, s, size=6.2, color=MUTED, **kw):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            color=color,
            va=kw.pop("va", "top"),
            ha=kw.pop("ha", "left"),
            linespacing=kw.pop("linespacing", 1.3),
            **kw,
        )

    def flow(p, q, label=None, side="right", lw=1.6):
        ax.add_patch(
            FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw, color=NAVY, shrinkA=0, shrinkB=0)
        )
        if label:
            ax.text(
                p[0] + (1.6 if side == "right" else -1.6),
                (p[1] + q[1]) / 2,
                label,
                ha="left" if side == "right" else "right",
                va="center",
                fontsize=6,
                color=NAVY,
                style="italic",
            )

    def thumb(x, y, w, h):
        a = ax.inset_axes([x, y, w, h], transform=ax.transData)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(CARD_EC)
            s.set_linewidth(0.5)
        a.set_facecolor("#fbfbf9")
        return a

    def tile(x, y, s=8.0, fc="#eaf1fb"):
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="none"))

    def drop(cx, cy, r, color):
        v = [
            (cx, cy + 1.9 * r),
            (cx + 0.35 * r, cy + 1.2 * r),
            (cx + r, cy + 0.55 * r),
            (cx + r, cy),
            (cx + r, cy - 0.55 * r),
            (cx + 0.55 * r, cy - r),
            (cx, cy - r),
            (cx - 0.55 * r, cy - r),
            (cx - r, cy - 0.55 * r),
            (cx - r, cy),
            (cx - r, cy + 0.55 * r),
            (cx - 0.35 * r, cy + 1.2 * r),
            (cx, cy + 1.9 * r),
        ]
        ax.add_patch(PathPatch(MPath(v, [MPath.MOVETO] + [MPath.CURVE4] * 12), fc=color, ec="none"))

    # ---- row 1: data ------------------------------------------------------------------------------
    y0, y1 = ROWY["1"]
    cw = 38.6
    DATA = [
        (
            "Crop statistics",
            "BBS: area, yield, production;\n16 districts, four crops,\n2015–2025; Boro as husked rice",
        ),
        (
            "Climate",
            "NASA POWER: rain, temperature,\nhumidity; FAO-56 crop water need\n11 distinct series for 16 districts",
        ),
        (
            "Groundwater trend",
            "Published records of the\nlong-term change in\ngroundwater level, by district",
        ),
        (
            "Crop water and value",
            "Net irrigation requirement (NIR)\nFood energy per kg\nPrices and production costs",
        ),
    ]
    for k, (head, body) in enumerate(DATA):
        x = 26 + k * (cw + 2.4)
        card(x, y0 + 2, cw, y1 - y0 - 4)
        tx, ty = x + 2.4, y1 - 12.8
        tile(tx, ty)
        ax.text(tx + 9.6, ty + 4.0, head, ha="left", va="center", fontsize=7.0, fontweight="bold")
        note(x + 2.4, ty - 1.6, body, size=5.9)
        if k == 0:
            for i, (c, hh) in enumerate(zip(CROPS, (5.6, 3.2, 4.4))):
                ax.add_patch(Rectangle((tx + 1.5 + i * 1.9, ty + 1.4), 1.5, hh, fc=CROP_COL[c], ec="none"))
        if k == 1:
            cx, cy = tx + 4.2, ty + 4.3
            ax.add_patch(Circle((cx, cy), 1.75, fc=MODEL_COL["Persistence"], ec="none"))
            for t in np.arange(8) * np.pi / 4:
                ax.plot(
                    [cx + 2.4 * np.cos(t), cx + 3.3 * np.cos(t)],
                    [cy + 2.4 * np.sin(t), cy + 3.3 * np.sin(t)],
                    color=MODEL_COL["Persistence"],
                    lw=1,
                    solid_capstyle="round",
                )
        if k == 2:
            ax.plot(
                [tx + 1.2, tx + 7.2], [ty + 6.6, ty + 6.6], color="#8a5a2b", lw=1.4, solid_capstyle="round"
            )
            ax.plot(
                [tx + 1.4, tx + 3.2, tx + 4.8, tx + 6.3],
                [ty + 5.2, ty + 4.5, ty + 3.2, ty + 2.0],
                color=CROP_COL["Boro Rice"],
                lw=1.3,
                ls=(0, (2.2, 1.2)),
            )
            ax.add_patch(
                Polygon(
                    [(tx + 6.3, ty + 1.2), (tx + 5.5, ty + 2.5), (tx + 7.1, ty + 2.5)],
                    fc=CROP_COL["Boro Rice"],
                    ec="none",
                )
            )
        if k == 3:
            drop(tx + 2.9, ty + 3.5, 1.55, CROP_COL["Boro Rice"])
            ax.add_patch(Circle((tx + 6.0, ty + 3.4), 1.7, fc=MODEL_COL["Persistence"], ec="none"))
            ax.text(
                tx + 6.0,
                ty + 3.3,
                "Tk",
                ha="center",
                va="center",
                fontsize=4.6,
                fontweight="bold",
                color="white",
            )

    # ---- row 2: study design ----------------------------------------------------------------------
    y0, y1 = ROWY["2"]
    card(26, y0 + 2, 98, y1 - y0 - 4, "Decision-time forecasting: only what is known when the plan is made")
    YEARS = list(range(2015, 2026))
    sq, x_t, y_t = 5.5, 47.5, y1 - 10.5
    for i, yr in enumerate(YEARS):
        ax.text(
            x_t + i * sq + sq / 2 - 0.3,
            y_t + 0.9,
            f"'{yr % 100}",
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=MUTED,
        )
    for r, target in enumerate((2023, 2024, 2025)):
        yy = y_t - (r + 1) * 4.0
        ax.text(x_t - 1.8, yy + 1.6, f"Forecast for {target}", ha="right", va="center", fontsize=6)
        for i, yr in enumerate(YEARS):
            if yr > target:
                continue
            kw = (
                dict(fc="#d5d0f7", ec="white")
                if yr <= target - 2
                else (
                    dict(fc=VIOLET, ec="white")
                    if yr == target
                    else dict(fc="white", ec=CARD_EC, hatch="/////")
                )
            )
            ax.add_patch(Rectangle((x_t + i * sq, yy), sq - 0.6, 3.2, lw=0.5, **kw))
    ly = y0 + 4.5
    for lx, kw, lab in [
        (29, dict(fc="#d5d0f7", ec="white"), "yields known when planning (to t − 2)"),
        (72, dict(fc="white", ec=CARD_EC, hatch="/////"), "not yet published (t − 1)"),
        (101, dict(fc=VIOLET, ec="white"), "target season t"),
    ]:
        ax.add_patch(Rectangle((lx, ly - 1.3), 3.6, 2.6, lw=0.5, **kw))
        ax.text(lx + 4.6, ly, lab, ha="left", va="center", fontsize=5.9, color=MUTED)
    card(130, y0 + 2, 57.5, y1 - y0 - 4, "Water accounting")
    note(
        132.4,
        y1 - 8.2,
        "Water demand = 10 × NIR × area  (m³)\nModelled irrigation budget = ρ × baseline demand",
        size=6.0,
        color=INK,
        linespacing=1.5,
    )
    for i, (lab, frac, colr) in enumerate(
        [("baseline demand", 1.0, BLUES[1]), ("budget, ρ = 0.80", 0.8, BLUES[4])]
    ):
        yy = y0 + 10.6 - i * 4.6
        ax.text(132.4, yy + 1.4, lab, ha="left", va="center", fontsize=5.8, color=MUTED)
        ax.add_patch(Rectangle((153.5, yy), 31, 2.8, fc="none", ec=CARD_EC, lw=0.5, ls=(0, (2, 1.5))))
        ax.add_patch(Rectangle((153.5, yy), 31 * frac, 2.8, fc=colr, ec="none"))

    # ---- row 3: models ----------------------------------------------------------------------------
    y0, y1 = ROWY["3"]
    ch = y1 - y0 - 4
    card(26, y0 + 2, 98, ch, "O1  Yield forecasting and explanation")
    note(28.4, y1 - 8.4, "Forecasts used by the optimiser", size=6.3, color=INK, fontweight="bold")
    for i, (m, mk, desc) in enumerate(
        [
            ("Elastic Net", "o", "lagged yields, trend"),
            ("Persistence", "D", "yield of season t − 2"),
            ("Climatology", "s", "district mean yield"),
        ]
    ):
        yy = y1 - 14.6 - i * 6.3
        ax.plot(30.2, yy, mk, ms=5, color=MODEL_COL[m], mec="white", mew=0.4)
        ax.text(32.8, yy + 0.1, m, ha="left", va="center", fontsize=6.4)
        note(32.8, yy - 1.5, desc, size=5.7)
    note(28.4, y0 + 4.9, "Also a district trend.  Test seasons 2023–2025", size=5.8)
    fcst = pd.read_csv(F / "decision_time_forecasts.csv")
    g = fcst[fcst.crop == "Boro Rice"]
    a = thumb(63.5, y0 + 7.6, 17, 17)
    lo, hi = g.yield_t_ha.min() - 0.12, g.yield_t_ha.max() + 0.12
    a.plot([lo, hi], [lo, hi], color=MUTED, lw=0.5, ls=(0, (3, 2)))
    a.plot(g.yield_t_ha, g.EN_cutoff2, "o", ms=1.9, color=VIOLET, mec="none")
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    note(72, y0 + 7.35, "observed", size=5.5, ha="center")
    ax.text(62.4, y0 + 16.1, "forecast", rotation=90, ha="center", va="center", fontsize=5.5, color=MUTED)
    ax.plot([82.6, 82.6], [y0 + 4.2, y1 - 7.4], color=CARD_EC, lw=0.6, ls=(0, (2, 2)))
    note(84.6, y1 - 8.4, "Explanation", size=6.3, color=INK, fontweight="bold")
    note(84.6, y1 - 12.2, "Elastic Net: exact additive\ncontributions of its inputs", size=5.8)
    shp = pd.read_csv(R / "04_gbt_shap_values.csv")
    s = shp[shp.crop == "Boro Rice"]
    cols = [
        "year_trend",
        "district_aggregate",
        "hybrid_area_share",
        "terminal_tmax_c",
        "season_tmin_c",
        "season_rain_mm",
        "monsoon_prev_rain_mm",
        "season_rh_pct",
    ]
    a = thumb(106.5, y0 + 4.4, 15.5, 14.4)
    rng2 = np.random.default_rng(1)
    for i, c in enumerate(cols):
        a.scatter(
            s[c],
            len(cols) - 1 - i + rng2.uniform(-0.25, 0.25, len(s)),
            s=1.3,
            color=VIOLET if i != 1 else "#8a8984",
            lw=0,
        )
    a.axvline(0, color=MUTED, lw=0.4)
    a.set_ylim(-0.7, len(cols) - 0.3)
    note(84.6, y0 + 13.6, "Hindcast boosting\nwith same-season\nclimate: SHAP,\ndescriptive only", size=5.6)

    card(130, y0 + 2, 57.5, ch, "O2  Relative scarcity index (RAWSI)")
    for i, lab in enumerate(["Groundwater trend", "Crop-mix water demand", "Climate deficit"]):
        yy = y1 - 13.2 - i * 5.3
        ax.add_patch(
            FancyBboxPatch(
                (132.4, yy - 1.9), 27.4, 3.8, boxstyle="round,pad=0,rounding_size=.9", fc="#fdebe0", ec="none"
            )
        )
        ax.text(146.1, yy, lab, ha="center", va="center", fontsize=5.8)
    note(146.1, y0 + 11.0, "scaled 0–1, averaged", size=5.8, ha="center")
    note(146.1, y0 + 7.6, "20 alternative constructions", size=5.8, ha="center")
    ax.plot([160.2, 160.2], [y1 - 13.2, y1 - 23.8], color=NAVY, lw=0.8, solid_capstyle="butt")
    for k_ in range(3):
        ax.plot([159.8, 160.2], [y1 - 13.2 - k_ * 5.3] * 2, color=NAVY, lw=0.8)
    ax.add_patch(
        FancyArrowPatch(
            (160.2, y1 - 18.5),
            (164, y1 - 18.5),
            arrowstyle="-|>",
            mutation_scale=6,
            lw=0.9,
            color=NAVY,
            shrinkA=0,
            shrinkB=0,
        )
    )
    a = ax.inset_axes([164.6, y0 + 3.4, 22.2, 28.2], transform=ax.transData)
    a.axis("off")
    nrm = Normalize(0.15, 0.85)
    for name in sorted(study):  # fixed order: a set's order changes between runs and moved edge pixels
        a.add_collection(
            PolyCollection(
                shapes[name], facecolors=ORANGES(nrm(t3m.RAWSI[name])), edgecolors="white", linewidths=0.25
            )
        )
    a.set_xlim(X0, X1)
    a.set_ylim(Y0, Y1)
    a.set_aspect(1 / np.cos(np.radians(25.0)))

    # ---- row 4: optimisation ----------------------------------------------------------------------
    y0, y1 = ROWY["4"]
    ch = y1 - y0 - 4
    card(26, y0 + 2, 98, ch, "O3  District crop-allocation LP  (16 districts)")
    FORM = [
        ("maximise", r"$\Sigma_c\ g_c\,x_c$"),
        ("land", r"$\Sigma_c\ x_c + u = A$"),
        ("water", r"$10\,\Sigma_c\ n_c\,x_c \leq \rho\,W_0$"),
        ("Boro floor", r"$x_{\mathrm{Boro}} \geq \phi\,a_{\mathrm{Boro}}$"),
        ("limits", r"$0 \leq x_c \leq U_c$"),
    ]
    for i, (lab, eq) in enumerate(FORM):
        yy = y1 - 10.4 - i * 4.55
        ax.text(
            28.4,
            yy,
            lab,
            ha="left",
            va="center",
            fontsize=6,
            color=MUTED,
            fontweight="bold" if i == 0 else "normal",
        )
        ax.text(41.5, yy, eq, ha="left", va="center", fontsize=6.6)
    note(28.4, y0 + 5.9, "x = crop area, u = fallow;  duals give shadow prices", size=5.7)
    note(66.5, y1 - 8.4, "Two objectives  (gain g per hectare)", size=6.3, color=INK, fontweight="bold")
    for i, (lab, eq) in enumerate(
        [
            ("Food energy", "yield × kcal/kg"),
            ("Net return", "revenue − cost"),
            ("Fallow plan", "free or minimal"),
        ]
    ):
        yy = y1 - 14.2 - i * 5.0
        ax.add_patch(
            FancyBboxPatch(
                (66.5, yy - 1.9), 17.6, 3.8, boxstyle="round,pad=0,rounding_size=.9", fc="#e2f4ec", ec="none"
            )
        )
        ax.text(75.3, yy, lab, ha="center", va="center", fontsize=5.8)
        note(85.4, yy, eq, size=5.8, va="center")
    rs2 = pd.read_csv(X / "ext_16_budget_response.csv")
    g = rs2[(rs2.plan == "least") & (rs2.objective == "food_energy")]
    a = thumb(108, y0 + 14.2, 14, 15.6)
    a.plot(
        g.budget_ratio, g.fallow_pct_of_land, "-o", color=CROP_COL["Boro Rice"], lw=0.9, ms=1.8, mec="none"
    )
    a.margins(0.16)
    note(115, y0 + 13.6, "budget", size=5.5, ha="center")
    ax.text(106.9, y0 + 22, "fallow %", rotation=90, ha="center", va="center", fontsize=5.5, color=MUTED)
    note(
        66.5,
        y0 + 12.6,
        "Budget ratio and Boro floor are varied;\nthe curve shows the fallow a cut needs",
        size=5.8,
    )

    card(130, y0 + 2, 53.5, ch, "O4  Γ-robust planning  (48 district-years)")
    note(
        132.4,
        y1 - 8.6,
        "Crop water need is uncertain: up to Γ of\nthe four crops need their upper NIR",
        size=5.9,
        color=INK,
    )
    note(
        132.4,
        y1 - 15.8,
        "Finds the smallest budget\nat which a plan fits every\nsuch case and still holds\na set share of food energy",
        size=5.8,
    )
    note(132.4, y0 + 8.4, "Result: the budget that holds\n80% or 90% of food energy", size=5.8)
    t6f = pd.read_csv(T / "Fig5_robust_budgets.csv")
    t6f = t6f[t6f.uncertainty_set.isin(["nominal", "spatial_upper"])]
    a = thumb(168.2, y0 + 12.4, 13.4, 13.6)
    for k, tg in enumerate((80, 90)):
        a.plot(
            range(5),
            t6f[f"budget_needed_to_hold_{tg}pct_median"],
            "-o",
            color=[BLUES[3], BLUES[6]][k],
            lw=0.9,
            ms=1.8,
            mec="none",
        )
    a.set_xlim(-0.35, 4.35)
    note(174.9, y0 + 11.8, "Γ = 0 … 4", size=5.5, ha="center")
    ax.text(
        166.9, y0 + 19.2, "budget needed", rotation=90, ha="center", va="center", fontsize=5.5, color=MUTED
    )

    # ---- row 5: decision support ------------------------------------------------------------------
    y0, y1 = ROWY["5"]
    tw = 30.3
    OUT = [
        ("Feasibility", "Which districts can keep\ntheir Boro area at\neach budget"),
        ("Trade-offs", "Food energy, net\nreturn and fallow under\nthe same budget"),
        ("Shadow prices", "Value of extra water;\ncost of holding\nthe Boro floor"),
        ("Forecast value", "Whether a better yield\nforecast changes\nthe plan"),
        ("Budget needed", "Budget that holds food\nenergy when crop water\nneed is uncertain"),
    ]
    for k, (head, body) in enumerate(OUT):
        x = 26 + k * (tw + 2.5)
        card(x, y0 + 7.2, tw, y1 - y0 - 9.2, fc="#fdf3df", ec="#ecd9ac")
        ax.text(x + tw / 2, y1 - 4.3, head, ha="center", va="top", fontsize=7, fontweight="bold")
        note(x + tw / 2, y1 - 8.0, body, size=5.8, ha="center")
    ax.text(
        106.75,
        y0 + 3.6,
        "Checks:  FAO-56 requirement in place of published NIR   ·   validation that leaves out whole climate cells   ·   five yield sources   ·   potato price Tk 6–34 per kg",
        ha="center",
        va="center",
        fontsize=5.7,
        color=MUTED,
        style="italic",
    )

    # ---- flows between the stages ----------------------------------------------------------------
    for x in (45.3, 86.3, 127.3, 168.3):
        flow((x, ROWY["1"][0] - 0.4), (x, ROWY["2"][1] + 0.4))
    flow((76.5, ROWY["2"][0] - 0.4), (76.5, ROWY["3"][1] + 0.4))
    flow((158.7, ROWY["2"][0] - 0.4), (158.7, ROWY["3"][1] + 0.4))
    flow((60, ROWY["3"][0] - 0.4), (60, ROWY["4"][1] + 0.4), "yield forecasts")
    flow(
        (127, ROWY["2"][0] - 0.4), (127, ROWY["4"][1] + 0.4)
    )  # the budget goes straight to the optimisation stage
    ax.text(
        127,
        (ROWY["3"][0] + ROWY["3"][1]) / 2,
        "modelled irrigation budget",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6,
        color=NAVY,
        style="italic",
        bbox=dict(fc=BAND, ec="none", pad=1.0),
    )
    flow((72.5, ROWY["4"][0] - 0.4), (72.5, ROWY["5"][1] + 0.4), "plans, shadow prices")
    flow((155.5, ROWY["4"][0] - 0.4), (155.5, ROWY["5"][1] + 0.4), "robust budgets")
    ax.add_patch(
        FancyArrowPatch(
            (186.6, ROWY["3"][0] + 2),
            (186.6, ROWY["5"][1] + 0.4),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.6,
            color=NAVY,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        186.6,
        (ROWY["4"][0] + ROWY["4"][1]) / 2,
        "district scarcity ranking",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6,
        color=NAVY,
        style="italic",
        bbox=dict(fc=BAND, ec="none", pad=1.2),
    )
    save(fig, "Fig2_framework")


def fig3():
    """Fig. 3 - feasibility."""
    boro = pd.read_csv(X / "ext_10_district_boro_share.csv")
    ORDER_SHARE = boro.sort_values("boro_water_share").district.tolist()  # low to high (bottom to top)
    e6 = pd.read_csv(X / "ext_06_pm_deficit_district.csv").set_index("district").loc[ORDER_SHARE]
    fig = plt.figure(figsize=(W, 3.9))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.15, 1], wspace=0.06, left=0.15, right=0.93, bottom=0.13, top=0.9
    )
    ax = fig.add_subplot(gs[0])
    yy = np.arange(len(e6))
    for rho in (0.7, 0.8, 0.9):
        ax.vlines(rho, -0.7, len(e6) - 0.5, color=MUTED, lw=0.7, ls=(0, (4, 2)))  # stops below its label
        ax.text(rho, len(e6) - 0.4, f"budget {rho:.2f}", ha="center", va="bottom", fontsize=MIN_PT, color=MUTED)
    ax.hlines(yy, e6.boro_share_PM_2023_2025, e6.boro_share_fixed_NIR, color=LIGHT, lw=1.6, zorder=1)
    ax.plot(
        e6.boro_share_fixed_NIR,
        yy,
        "o",
        ms=5,
        color=CROP_COL["Boro Rice"],
        mec="white",
        mew=0.6,
        zorder=3,
        label="Published NIR",
    )
    ax.plot(
        e6.boro_share_PM_2023_2025,
        yy,
        "D",
        ms=4,
        mfc="white",
        mec=CROP_COL["Boro Rice"],
        mew=1.1,
        zorder=3,
        label="FAO-56 requirement",
    )
    ax.set_yticks(yy)
    ax.set_yticklabels(e6.index)
    ax.set_ylim(-0.7, len(e6) + 0.35)
    ax.set_xlim(0.5, 1.0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Boro share of the district's modelled irrigation demand")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), handletextpad=0.3, fontsize=MIN_PT)  # empty corner, clear of the budget lines
    panel(ax, "a", "Full retention fails where share > budget")
    ax = fig.add_subplot(gs[1])
    RHO = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    loss = np.array([[100 * max(0.0, 1 - r / e6.boro_share_fixed_NIR[d]) for r in RHO] for d in e6.index])[
        ::-1
    ]
    cm_ = ORANGES.copy()
    cm_.set_bad("#f3f2ee")
    im = ax.imshow(np.ma.masked_where(loss <= 0, loss), cmap=cm_, vmin=0, vmax=40, aspect="auto")
    for i in range(loss.shape[0]):
        for k in range(loss.shape[1]):
            if loss[i, k] > 0:
                ax.text(
                    k,
                    i,
                    f"{loss[i, k]:.0f}" if loss[i, k] >= 0.5 else "<1",
                    ha="center",
                    va="center",
                    fontsize=MIN_PT,
                    color="white" if loss[i, k] > 22 else INK,
                )
    ax.set_xticks(range(len(RHO)))
    ax.set_xticklabels([f"{r:.2f}" for r in RHO], fontsize=6.3)
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(RHO)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(e6)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Budget ratio")
    ax.set_ylim(len(e6) - 0.5 + 0.35 + 0.2, -0.5 - 0.35) if False else None
    cax = fig.add_axes([0.945, 0.25, 0.012, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=6.3)
    cb.set_label("% of baseline Boro area", fontsize=6.5)
    ax.set_title("(b) Smallest possible cut in Boro area (%)", loc="left", fontweight="bold", pad=5)
    save(fig, "Fig3_feasibility")


def fig4():
    """Fig. 4 - plans, fallow, budget response, frontier."""
    gm = pd.read_csv(X / "ext_12_gross_margin_objective.csv")
    rs = pd.read_csv(X / "ext_16_budget_response.csv")
    fr = pd.read_csv(X / "ext_17_energy_margin_frontier.csv")
    FALLOW_COL = "#d9d7d0"
    fig = plt.figure(figsize=(W, 5.4))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.95, 1.05],
        hspace=0.46,
        wspace=0.38,
        left=0.17,
        right=0.985,
        bottom=0.085,
        top=0.94,
    )
    ax = fig.add_subplot(gs[0, :])
    OBJ = [
        ("free", "food_energy", "Food energy, free fallow"),
        ("free", "gross_margin_potato_mid", "Net return, free fallow"),
        ("least", "gross_margin_potato_mid", "Net return, least fallow"),
    ]
    rows = [(rho, p_, o, lab) for rho in (0.8, 1.0) for p_, o, lab in OBJ]
    # at budget 0.80 the least-fallow plan is the same under both objectives (Table 3)
    rows = [(rho, p_, o, "Least fallow, either objective" if (rho == 0.8 and p_ == "least") else lab)
            for rho, p_, o, lab in rows]
    yy = [5.4, 4.4, 3.4, 1.8, 0.8, -0.2]
    for y, (rho, p_, o, lab) in zip(yy, rows):
        r = gm[np.isclose(gm.budget_ratio, rho) & (gm.plan == p_) & (gm.objective == o)].iloc[0]
        left = 0
        for c, col in zip(CROPS_A + ["Fallow"], LAND_COLS):
            fc_, hc_ = (FALLOW_COL, None) if c == "Fallow" else (CROP_COL[c], CROP_HATCH[c])
            ax.barh(y, r[col], left=left, height=0.62, color=fc_, hatch=hc_, edgecolor="white", linewidth=1.2)
            if r[col] > 5:
                ax.text(
                    left + r[col] / 2,
                    y,
                    f"{r[col]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=INK if c in ("Fallow", "Maize") else "white",
                    fontweight="bold",
                    bbox=(
                        None if c == "Fallow" else dict(boxstyle="round,pad=0.12", fc=CROP_COL[c], ec="none")
                    ),
                )
            left += r[col]
        ax.text(
            101.5,
            y,
            f"{r.food_energy_trillion_kcal:.1f} tn kcal · Tk {r.gross_margin_billion_Tk_potato_mid:.0f} bn",
            va="center",
            fontsize=6.5,
        )
    ax.set_yticks(yy)
    ax.set_yticklabels([lab for _, _, _, lab in rows])
    ax.tick_params(axis="y", length=0)
    ax.text(-25, 6.15, "Budget 0.80", fontsize=7, fontweight="bold", ha="left")
    ax.text(-25, 2.55, "Budget 1.00", fontsize=7, fontweight="bold", ha="left")
    ax.set_xlim(0, 130)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.75, 6.5)
    ax.set_xlabel("Share of the dry-season land pool (%)")
    ax.xaxis.set_label_coords(0.385, -0.13)
    ax.spines["bottom"].set_bounds(0, 100)
    for c in CROPS_A:
        ax.barh(
            [np.nan], [np.nan], color=CROP_COL[c], hatch=CROP_HATCH[c], edgecolor="white", label=CROP_LABEL[c]
        )
    ax.barh([np.nan], [np.nan], color=FALLOW_COL, edgecolor="white", label="Fallow")
    ax.legend(loc="upper right", ncol=5, bbox_to_anchor=(1.0, 1.17), handlelength=1.4, columnspacing=1.2)
    panel(ax, "a", "Land use of each plan (80% Boro floor)")
    STY = {
        ("free", "food_energy"): ("Food energy, free fallow", INK, "-", "s"),
        ("free", "gross_margin"): ("Net return, free fallow", BLUES[3], "-", "^"),
        ("least", "gross_margin"): ("Net return, least fallow", BLUES[3], (0, (1.2, 1.2)), "o"),
    }
    for j, (col, yl, title) in enumerate(
        [
            ("fallow_pct_of_land", "Fallow (% of land)", "Fallow rises fast below 0.90"),
            ("food_energy_trillion_kcal", "Food energy (trillion kcal)", "Food energy falls with it"),
        ]
    ):
        ax = fig.add_subplot(gs[1, j])
        for (p_, o), (lab, colr, ls, mk) in STY.items():
            g = rs[(rs.plan == p_) & (rs.objective == o)]
            ax.plot(
                g.budget_ratio,
                g[col],
                color=colr,
                lw=1.5,
                ls=ls,
                marker=mk,
                ms=3.2,
                mec="white",
                mew=0.4,
                label=lab,
                zorder=3 if o == "food_energy" else 2,
            )
        ax.axvline(0.8, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xlabel("Budget ratio")
        ax.set_ylabel(yl)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
        if j == 0:
            ax.legend(loc="upper right", handlelength=2.6, fontsize=6.5, frameon=True, facecolor="white",
                      edgecolor="none", framealpha=0.9)
        panel(ax, "bc"[j], title)
    ax = fig.add_subplot(gs[1, 2])
    for p_, colr, mk, lab in [
        ("free", BLUES[4], "o", "Free fallow"),
        ("least", BLUES[6], "s", "Least fallow"),
    ]:
        g = (
            fr[np.isclose(fr.budget_ratio, 0.8) & (fr.plan == p_)]
            .drop_duplicates(["gross_margin_billion_Tk", "food_energy_trillion_kcal"])
            .sort_values("gross_margin_billion_Tk")
        )
        ax.plot(
            g.gross_margin_billion_Tk,
            g.food_energy_trillion_kcal,
            color=colr,
            lw=1,
            marker=mk,
            ms=3.5,
            mec="white",
            mew=0.4,
            label=lab,
        )
    g = (fr[np.isclose(fr.budget_ratio, 0.8) & (fr.plan == "free")]
         .drop_duplicates(["gross_margin_billion_Tk", "food_energy_trillion_kcal"])
         .sort_values("gross_margin_billion_Tk"))
    a0, a1 = g.iloc[0], g.iloc[-1]
    ax.annotate("food-energy\nplan", (a0.gross_margin_billion_Tk, a0.food_energy_trillion_kcal),
                xytext=(0, -6), va="top", textcoords="offset points", fontsize=6.5, color=BLUES[5])
    ax.annotate("net-return plan", (a1.gross_margin_billion_Tk, a1.food_energy_trillion_kcal),
                xytext=(-6, 9), textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=BLUES[5])
    lf = fr[np.isclose(fr.budget_ratio, 0.8) & (fr.plan == "least")].iloc[0]
    ax.annotate("least fallow", (lf.gross_margin_billion_Tk, lf.food_energy_trillion_kcal),
                xytext=(6, 0), textcoords="offset points", fontsize=6.5, color=BLUES[6], va="center")
    ax.text(0.04, 0.26, "line: free-fallow plans\nwith mixed objectives", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Net return (bn Tk)")
    ax.set_ylabel("Food energy (trillion kcal)")
    ax.grid(True)
    ax.set_axisbelow(True)
    panel(ax, "d", "Trade-off at budget 0.80")
    save(fig, "Fig4_plans")


def fig5():
    """Fig. 5 - Gamma."""
    t6 = pd.read_csv(T / "Fig5_robust_budgets.csv")
    assert len(t6) == 9
    nom = t6[t6.uncertainty_set == "nominal"].iloc[0]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(W, 3.35),
        sharey=True,
        gridspec_kw=dict(wspace=0.1, left=0.075, right=0.94, bottom=0.235, top=0.83),
    )
    SETSTY = [
        ("temporal_cv", "Temporal NIR deviation (year-to-year CV)", "#d9622b", "o", -0.09),
        ("spatial_upper", "Spatial upper NIR (range across districts)", BLUES[6], "s", 0.09),
    ]
    for ax, tg in zip(axes, (80, 90)):
        for spec, lab, colr, mk, dx in SETSTY:
            g = pd.concat([nom.to_frame().T, t6[t6.uncertainty_set == spec]]).sort_values("gamma")
            gx = g.gamma.astype(float).to_numpy() + np.where(g.gamma.to_numpy() == 0, 0, dx)
            med, worst = g[f"budget_needed_to_hold_{tg}pct_median"].astype(float), g[
                f"budget_needed_to_hold_{tg}pct_worst_case"
            ].astype(float)
            ax.vlines(gx, med, worst, color=colr, lw=1.2)
            ax.plot(gx, med, "-", color=colr, lw=1.4, label=lab)
            ax.plot(gx, med, mk, ms=4.5, mfc="white", mec=colr, mew=1.1)
            ax.plot(gx, worst, mk, ms=4.5, color=colr, mec="white", mew=0.5)
            ax.text(
                gx[-1] + 0.12,
                worst.iloc[-1],
                f"{worst.iloc[-1]:.3f}",
                ha="left",
                va="center",
                fontsize=MIN_PT,
                color=INK,
            )
        ax.axhline(0.8, color=MUTED, lw=0.7, ls=(0, (4, 2)))
        ax.text(4.65, 0.797, "budget 0.80 (a 20% cut)", ha="right", va="top", fontsize=MIN_PT, color=MUTED)
        ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (1, 1.5)))
        if tg == 80:
            ax.text(-0.25, 0.997, "baseline demand (no cut)", ha="left", va="top", fontsize=MIN_PT, color=MUTED)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"Γ = {k}" for k in range(5)])
        ax.set_xlim(-0.35, 4.7)
        ax.set_xlabel("Crops whose NIR may take its high value")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        panel(ax, "ab"[tg == 90], f"Holding {tg}% of the food energy of the baseline areas")
        if tg == 90:
            sec = ax.secondary_yaxis("right", functions=(lambda r: 100 * (1 - r), lambda c: 1 - c / 100))
            sec.set_yticks([0, 5, 10, 15, 20])
            sec.set_yticklabels(["0%", "5%", "10%", "15%", "20%"])
            sec.tick_params(length=2, labelsize=MIN_PT)
            sec.set_ylabel("Cut in the budget", fontsize=6.8)
    sp = t6[(t6.uncertainty_set == "spatial_upper") & (t6.gamma == 1)].iloc[0]
    n_inf = 48 - int(sp["feasible_cases_of_48_at_budget_0.80"])
    axes[0].annotate(f"spatial set: {n_inf} of 48 cases cannot\nkeep 80% of their Boro at 0.80",
                     xy=(1.09, float(sp.budget_needed_to_hold_80pct_worst_case)), xytext=(1.45, 0.965),
                     fontsize=MIN_PT, color=BLUES[6], ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=BLUES[6], lw=0.6))
    top = max(t6[f"budget_needed_to_hold_{tg}pct_worst_case"].max() for tg in (80, 90))
    axes[0].set_ylabel("Budget ratio needed (48 district-years)")
    axes[0].set_ylim(0.76, max(1.01, top + 0.02))
    axes[0].plot([], [], "o", ms=4.5, mfc="white", mec=MUTED, mew=1.1, label="Median case")
    axes[0].plot([], [], "o", ms=4.5, color=MUTED, label="Worst case")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(
        hd,
        lb,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.0),
        ncol=4,
        fontsize=MIN_PT,
        handlelength=1.6,
        columnspacing=1.6,
    )  # below the panels: clear of the reference lines
    fig.suptitle("Uncertain crop water needs shrink the safe cut: 7–10% with the spatial set, "
                 "13–16% with the temporal set (worst cases, Γ = 1–4)", x=0.075, ha="left", fontsize=8,
                 fontweight="bold", y=0.985)
    save(fig, "Fig5_robust_budgets")


def fig6():
    """Fig. 6 - AWD and the value of water."""
    awdc = pd.read_csv(X / "ext_26_awd_fallow_curve.csv")
    wv = pd.read_csv(X / "ext_15_shadow_prices.csv")
    wv = wv[
        (wv.objective == "gross_margin") & (wv.plan == "free") & np.isclose(wv.budget_ratio, 0.8)
    ].sort_values("water_value_per_1000_m3")
    chg = pd.read_csv(X / "ext_27_water_value_vs_charges.csv")
    chg = chg[np.isclose(chg.budget_ratio, 0.8) & (chg.plan == "free")].iloc[0]
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(W, 3.4),
        gridspec_kw=dict(wspace=0.42, left=0.07, right=0.985, bottom=0.13, top=0.9, width_ratios=[1, 1.05]),
    )
    SAV = sorted(awdc.water_saving.unique())
    for k, sv in enumerate(SAV):
        g = awdc[np.isclose(awdc.water_saving, sv) & np.isclose(awdc.budget_ratio, 0.8)]
        colr = BLUES[2 + k + (k > 1)]
        ax.plot(
            g.adoption_share,
            g.min_fallow_pct,
            color=colr,
            lw=1.5,
            marker="osD^"[k],
            ms=3,
            markevery=4,
            mfc="white",
            mew=0.9,
            label=f"{100 * sv:.0f}%",
        )
    ax.legend(title="Water saved by AWD", title_fontsize=6.5, loc="upper right", handlelength=1.8)
    g0 = awdc[np.isclose(awdc.budget_ratio, 0.8) & np.isclose(awdc.adoption_share, 0)].min_fallow_pct.iloc[0]
    ax.annotate(
        f"conventional\nirrigation: {g0:.1f}%",
        xy=(0, g0),
        xytext=(0.1, g0 + 0.5),
        fontsize=MIN_PT,
        color=MUTED,
        va="bottom",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 21)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.25)])
    ax.set_xlabel("Share of the Boro area under AWD")
    ax.set_ylabel("Smallest fallow share (% of land pool)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    h = awdc[np.isclose(awdc.water_saving, 0.2) & np.isclose(awdc.budget_ratio, 0.8) & np.isclose(awdc.adoption_share, 0.5)]
    hv = float(h.min_fallow_pct.iloc[0])
    ax.plot([0.5], [hv], "o", ms=6.5, mfc="none", mec=INK, mew=0.9, zorder=5,
            label=f"half the area, 20% saving:\nfallow {hv:.1f}% (half)")
    ax.legend(title="Water saved by AWD", title_fontsize=6.5, loc="upper right", handlelength=1.8, fontsize=MIN_PT)
    panel(ax, "a", "AWD shrinks the fallow at budget 0.80")
    y = np.arange(len(wv))
    bands = [
        ("bmda_dtw_hourly_charge", "#cde2fb"),
        ("boro_irrigation_cost_per_ha_2016_17", "#ececea"),
    ]
    for key, colr in bands[::-1]:
        lo, hi = chg[f"{key}_Tk_per_m3_low"], chg[f"{key}_Tk_per_m3_high"]
        ax2.axvspan(lo, hi, color=colr, lw=0, zorder=0)
    bl, bh = chg["bmda_dtw_hourly_charge_Tk_per_m3_low"], chg["bmda_dtw_hourly_charge_Tk_per_m3_high"]
    cl, ch = chg["boro_irrigation_cost_per_ha_2016_17_Tk_per_m3_low"], chg["boro_irrigation_cost_per_ha_2016_17_Tk_per_m3_high"]
    ax2.text(np.sqrt(bl * bh), len(wv) - 0.2, f"pumping\ncharge\nTk {bl:.2f}–\n{bh:.2f}", ha="center", va="top",
             fontsize=MIN_PT, color=BLUES[5])
    ax2.text(np.sqrt(1.15 * ch), len(wv) - 0.2, f"surveyed\nirrigation\ncost, at most\nTk {cl:.2f}–{ch:.2f}",
             ha="center", va="top", fontsize=MIN_PT, color=MUTED)
    ax2.hlines(y, bh, wv.water_value_per_1000_m3, color=LIGHT, lw=0.8, zorder=1)
    ax2.plot(wv.water_value_per_1000_m3, y, "o", ms=4.3, color=BLUES[5], mec="white", mew=0.5, zorder=3)
    for yy_, v in zip(y, wv.water_value_per_1000_m3):
        ax2.text(v * 1.12, yy_, f"{v:.0f}", va="center", ha="left", fontsize=MIN_PT, color=INK)
    med = chg.water_value_Tk_per_m3_median
    ax2.axvline(med, color=BLUES[5], lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax2.text(med * 0.94, -0.95, f"median Tk {med:.1f}", ha="right", va="center", fontsize=MIN_PT, color=BLUES[5])
    ax2.set_xscale("log")
    ax2.set_xlim(0.4, 160)
    ax2.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    ax2.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax2.set_yticks(y)
    ax2.set_yticklabels(wv.district, fontsize=MIN_PT)
    ax2.set_ylim(-1.5, len(wv) - 0.2 + 2.6)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("Tk per m³ (log scale)")
    ax2.xaxis.grid(True, which="major")
    ax2.set_axisbelow(True)
    panel(ax2, "b", "Value of water in the model vs what farmers pay")
    save(fig, "Fig6_awd_water_value")


def fig7():
    """Fig. 7 - low-water crops (six-crop variant)."""
    lw = pd.read_csv(X / "ext_29_low_water_crops.csv")
    lw8 = lw[np.isclose(lw.budget_ratio, 0.8) & (lw.plan == "least") & (lw.objective == "food_energy")]
    KS = sorted(lw8.low_water_crop_limit_x_baseline.unique())
    xk = np.arange(len(KS))
    KLAB = ["observed\nmaximum"] + [f"{k:g}×\nbaseline" for k in KS[1:]]
    C6 = CROPS_A + ["Lentil", "Mustard"]
    COL6 = {**CROP_COL, "Lentil": "#e87ba4", "Mustard": "#6250d6"}
    HATCH6 = {**CROP_HATCH, "Lentil": "\\\\\\\\", "Mustard": "++"}
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(W, 3.0),
        gridspec_kw=dict(wspace=0.46, left=0.075, right=0.985, bottom=0.2, top=0.88, width_ratios=[1, 1.02]),
    )
    for name, lab, colr, mk in (
        ("none", "Conventional irrigation", BLUES[5], "o"),
        ("awd_half_boro_20pct_saving", "AWD on half the Boro area (20% saving)", BLUES[2], "s"),
    ):
        g = lw8[lw8.awd == name].sort_values("low_water_crop_limit_x_baseline")
        ax.plot(
            xk,
            g.min_fallow_thousand_ha,
            color=colr,
            lw=1.6,
            marker=mk,
            ms=4.2,
            mfc="white",
            mew=1.1,
            label=lab,
        )
        for xi, v in zip(xk, g.min_fallow_thousand_ha):
            ax.text(xi, v + 16, f"{v:.0f}", ha="center", va="bottom", fontsize=6.1, color=INK)
    four = lw8.four_crop_model_min_fallow_thousand_ha.iloc[0]
    ax.axhline(four, color=MUTED, lw=0.7, ls=(0, (4, 2)))
    ax.text(
        len(KS) - 0.55,
        four - 10,
        f"four-crop model: {four:.0f}",
        ha="right",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    ax.set_xticks(xk)
    ax.set_xticklabels(KLAB, fontsize=6.4)
    ax.set_xlim(-0.45, len(KS) - 0.5)
    ax.set_ylim(0, 600)
    ax.set_xlabel("Largest lentil and mustard area allowed in a district")
    ax.set_ylabel("Smallest fallow (thousand ha)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.3, handlelength=1.8)
    panel(ax, "a", "Fallow falls only far beyond 2015–2025 areas")
    g = lw8[lw8.awd == "none"].sort_values("low_water_crop_limit_x_baseline")
    left = np.zeros(len(g))
    for c in C6 + ["Fallow"]:
        v = 100 * g[f"{c.split()[0].lower()}_thousand_ha"].to_numpy() / g.land_pool_thousand_ha.to_numpy()
        fallow = c == "Fallow"
        ax2.barh(
            xk,
            v - 0.35,
            left=left + 0.175,
            height=0.66,
            color="#d9d7d0" if fallow else COL6[c],  # same fallow colour as Fig. 4
            edgecolor="white",
            hatch="" if fallow else HATCH6[c],
            lw=0,
            label="Fallow" if fallow else CROP_LABEL.get(c, c),
        )
        for xi, (l, vi) in enumerate(zip(left, v)):
            if vi >= 4.5:
                ax2.text(
                    l + vi / 2,
                    xi,
                    f"{vi:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=INK,
                    path_effects=[withStroke(linewidth=1.6, foreground="white")],
                )
        left += v
    ax2.set_yticks(xk)
    ax2.set_yticklabels([l.replace("\n", " ") for l in KLAB], fontsize=6.4)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of the six-crop land pool (%)")
    ax2.spines["left"].set_visible(False)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=7,
        fontsize=6.1,
        handlelength=1.1,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    panel(ax2, "b", "Mustard takes the released land (least-fallow plan)")
    save(fig, "Fig7_low_water_crops")

    print("FIGURES COMPLETE", len(list(O.glob("*.png"))))


def fig8():
    """Fig. 8 - one-at-a-time sensitivity, as a grid of ranges grouped by kind of input."""
    tor = pd.read_csv(X / "ext_28_sensitivity_tornado.csv")
    OUTC = [
        ("min_fallow_pct", "Smallest fallow", "percentage points", "{:.1f}% of land"),
        ("food_energy_trillion_kcal", "Food energy, food-energy plan", "% change", "{:.1f} trillion kcal"),
        ("gross_margin_billion_Tk", "Net return, net-return plan", "% change", "Tk {:.1f} billion"),
    ]
    GROUPS = [
        ("Water, land and policy", "#2a78d6", [
            ("Boro floor: 0.75 / 0.85 of baseline", "Boro floor 0.75 / 0.85"),
            ("Water coefficients: FAO-56, single years 2015-2025", "FAO-56 requirement, single years"),
            ("Crop limits: 1.5 x baseline area", "Crop limits 1.5× baseline area"),
        ]),
        ("Yields", "#6250d6", [
            ("Yields: persistence, climatology, observed, hindcast", "Four yield sources"),
        ]),
        ("Prices and costs", "#c2410c", [
            ("Potato price: low / high", "Potato price low / high"),
            ("Boro and wheat at procurement price", "Boro and wheat at procurement price"),
            ("Maize price: low / high", "Maize price low / high"),
            ("Maize cost: -20% / +20%", "Maize cost −20% / +20%"),
            ("Wheat cost: -20% / +20%", "Wheat cost −20% / +20%"),
        ]),
        ("Management option (not an uncertainty)", "#0d366b", [
            ("AWD on half the Boro area, 15-30% saving", "AWD on half the Boro area"),
            ("AWD on all Boro area, 15-30% saving", "AWD on all the Boro area"),
        ]),
    ]
    known = {f for _, _, rows in GROUPS for f, _ in rows}
    assert known == set(tor.factor), set(tor.factor) ^ known
    # rows top to bottom, with a header row for each group
    layout, y = [], 0.0
    for g, col, rows in GROUPS:
        layout.append(("head", g, col, y)); y += 1.0
        for f, lab in rows:
            layout.append(("row", f, lab, col, y)); y += 1.0
        y += 0.35
    ymax = y - 0.35
    fig, axes = plt.subplots(1, 3, figsize=(W, 4.3), sharey=True,
                             gridspec_kw=dict(wspace=0.10, left=0.27, right=0.985, bottom=0.10, top=0.84))
    for k, (ax, (oc, olab, unit, mfmt)) in enumerate(zip(axes, OUTC)):
        t = tor[tor.outcome == oc].set_index("factor")
        span = max(t[["low_change", "high_change"]].abs().max().max(), 1e-9)
        ax.set_xlim(-1.18 * span, 1.18 * span)
        for item in layout:
            if item[0] == "head":
                continue
            _, f, lab, col, yy = item
            r = t.loc[f]
            lo, hi = sorted((r.low_change, r.high_change))
            if r.settings == 1:
                lo, hi = sorted((0.0, r.low_change))
            if abs(lo) < 1e-9 and abs(hi) < 1e-9:
                ax.plot(0, yy, "o", ms=2.6, color=LIGHT, zorder=3)
                continue
            if r.settings == 1:  # one alternative: a thin stem from the main case to its value
                ax.hlines(yy, lo, hi, color=col, lw=1.0, zorder=2)
                v = r.low_change
                ax.plot(v, yy, "o", ms=5.2, color=col, mec="white", mew=0.6, zorder=4)
                ends = [v]
            else:
                if lo > 0 or hi < 0:  # range away from zero: dotted link back to the main case
                    ax.hlines(yy, min(0, lo), max(0, hi), color=MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
                ax.hlines(yy, lo, hi, color=col, lw=4.2, alpha=0.9, zorder=2, capstyle="round")
                ends = [lo, hi]
            if hi - lo >= 0.12 * span or r.settings == 1:
                txt = " / ".join(fmt_minus(e, 1 if abs(e) < 10 else 0).replace("-", "−") if e < 0 else
                                 ("+" + (f"{e:.1f}" if abs(e) < 10 else f"{e:.0f}")) for e in ends)
                xc = ends[0] if len(ends) == 1 else (lo + hi) / 2
                xc = max(min(xc, 0.82 * span), -0.82 * span)
                ax.text(xc, yy - 0.42, txt, ha="center", va="center", fontsize=MIN_PT, color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.02", fc="white", ec="none"))
        for item in layout:
            if item[0] == "head":
                ax.axhline(item[3] - 0.55, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=INK, lw=0.7, zorder=1)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(unit, fontsize=6.8, labelpad=2)
        m = t.iloc[0].main_value
        ax.set_title(f"{olab}\nmain case {mfmt.format(m)}", fontsize=7, loc="left", pad=4)
    rows = [it for it in layout if it[0] == "row"]
    axes[0].set_yticks([it[4] for it in rows])
    axes[0].set_yticklabels([it[2] for it in rows], fontsize=MIN_PT)
    from matplotlib.transforms import blended_transform_factory as _btf
    tr = _btf(fig.transFigure, axes[0].transData)
    for it in layout:
        if it[0] == "head":
            axes[0].text(0.012, it[3], it[1], transform=tr, ha="left", va="center", fontsize=6.8,
                         fontweight="bold", color=it[2])
    for ax in axes:
        ax.set_ylim(ymax - 0.4, -0.7)
    fig.suptitle("Budget 0.80: water, land and policy settings move the fallow; prices move only the net return",
                 x=0.012, ha="left", fontsize=8, fontweight="bold", y=0.985)
    fig.text(0.012, 0.925, "Change from the main case when one input takes its alternative values. "
             "Bars span the range over the alternatives; a dot is a single alternative; a grey dot means no change.",
             fontsize=MIN_PT, color=MUTED, ha="left")
    save(fig, "Fig8_sensitivity")


def figs1():
    """Fig. S1 - data overview."""
    ya = pd.read_csv(X / "ext_18_regional_yield_area_by_year.csv")
    cr = pd.read_csv(X / "ext_19_yield_climate_correlation.csv")
    DIV = LinearSegmentedColormap.from_list("div", ["#1f5fb0", "#8fb8ec", "#ecebe7", "#f2a583", "#c2410c"])
    fig = plt.figure(figsize=(W, 4.7))
    gs = fig.add_gridspec(
        2, 6, height_ratios=[1, 1.15], hspace=0.5, wspace=1.1, left=0.07, right=0.985, bottom=0.1, top=0.93
    )
    for j, c in enumerate(CROPS):
        ax = fig.add_subplot(gs[0, 2 * j : 2 * j + 2])
        g = ya[ya.crop == c]
        ax.fill_between(
            g.year_end, g.district_yield_min, g.district_yield_max, color=CROP_COL[c], alpha=0.18, lw=0
        )
        ax.plot(
            g.year_end,
            g.regional_yield_t_ha,
            color=CROP_COL[c],
            lw=1.8,
            marker="osv^"[j % 3] if False else ["o", "s", "^"][j],
            ms=3.5,
            mec="white",
            mew=0.5,
        )
        ax.set_xticks([2015, 2017, 2019, 2021, 2023, 2025])
        ax.set_xticklabels(["'15", "'17", "'19", "'21", "'23", "'25"])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(f'({"abc"[j]}) {CROP_LABEL[c]} yield', loc="left", fontweight="bold", pad=5)
        if j == 0:
            ax.set_ylabel("Yield (t/ha)")
        ax.axvspan(2022.5, 2025.5, color="#f3f2ee", zorder=0)
        ax.text(2024, ax.get_ylim()[0], "test", ha="center", va="bottom", fontsize=6, color=MUTED)
    gsb = gs[1, :].subgridspec(1, 3, width_ratios=[1, 0.34, 0.98], wspace=0)
    ax = fig.add_subplot(gsb[0, 0])
    yrs = sorted(ya.year_end.unique())
    bottom = np.zeros(len(yrs))
    for c in CROPS:
        v = ya[ya.crop == c].sort_values("year_end").regional_area_thousand_ha.to_numpy()
        ax.bar(
            yrs,
            v,
            bottom=bottom,
            width=0.72,
            color=CROP_COL[c],
            hatch=CROP_HATCH[c],
            edgecolor="white",
            linewidth=0.8,
            label=CROP_LABEL[c],
        )
        bottom += v
    ax.set_xticks(yrs)
    ax.set_xticklabels([f"'{str(y)[2:]}" for y in yrs])
    ax.set_ylabel("Area (thousand ha)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.legend(loc="upper left", ncol=3, handlelength=1.3, columnspacing=1.1)
    panel(ax, "d", "Cropped area of the three crops, 16 districts")
    ax = fig.add_subplot(gsb[0, 2])
    VAR = [
        ("year_trend", "Year trend"),
        ("terminal_tmax_c", "Terminal max. temp."),
        ("season_tmin_c", "Season min. temp."),
        ("season_rh_pct", "Relative humidity"),
        ("season_rain_mm", "Season rainfall"),
        ("monsoon_prev_rain_mm", "Previous monsoon rain"),
    ]
    M = np.array(
        [
            [
                cr[(cr.crop == c) & (cr.variable == v)][col].iloc[0]
                for col in ("pearson_r", "pearson_r_within_district")
                for c in CROPS
            ]
            for v, _ in VAR
        ]
    )
    ax.imshow(M, cmap=DIV, vmin=-0.7, vmax=0.7, aspect="auto")
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            ax.text(
                k,
                i,
                f"{M[i, k]:+.2f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if abs(M[i, k]) > 0.5 else INK,
            )
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Boro", "Wheat", "Potato"] * 2)
    ax.set_yticks(range(len(VAR)))
    ax.set_yticklabels([l for _, l in VAR])
    ax.tick_params(length=0)
    ax.axvline(2.5, color="white", lw=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1, -0.72, "Pooled", ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.text(4, -0.72, "Within district", ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.set_title("(e) Correlation of yield with predictors (r)", loc="left", fontweight="bold", pad=14)
    save(fig, "FigS1_crop_data")


def figs2():
    """Fig. S2 - SHAP (hindcast GBT, descriptive)."""
    shap = pd.read_csv(R / "04_gbt_shap_values.csv")
    pan = pd.read_csv(I / "analysis_panel_full.csv")
    pan["year_trend"] = pan.year_end - pan.year_end.min()
    pan["wheat_postbreak"] = (pan.year_end >= 2018).astype(int)
    FEATS = [
        ("year_trend", "Year trend"),
        ("district_aggregate", "District effects"),
        ("structure", "Variety / structure term"),
        ("terminal_tmax_c", "Terminal max. temp."),
        ("season_tmin_c", "Season min. temp."),
        ("season_rain_mm", "Season rainfall"),
        ("monsoon_prev_rain_mm", "Previous monsoon rain"),
        ("season_rh_pct", "Relative humidity"),
    ]
    STRUCT = {"Boro Rice": "hybrid_area_share", "Wheat": "wheat_postbreak", "Potato": "hyv_area_share"}
    VAL = LinearSegmentedColormap.from_list("val", ["#1f5fb0", "#8fb8ec", "#d9d8d3", "#f2a583", "#c2410c"])
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(W, 3.7))
    gs = fig.add_gridspec(1, 3, wspace=0.08, left=0.17, right=0.93, bottom=0.15, top=0.9)
    for j, c in enumerate(CROPS):
        ax = fig.add_subplot(gs[j])
        s = shap[shap.crop == c].rename(columns={STRUCT[c]: "structure"})
        v = (
            pan[pan.crop == c]
            .rename(columns={STRUCT[c]: "structure"})
            .merge(
                s[["district", "year_end"]], on=["district", "year_end"], how="right", validate="one_to_one"
            )
        )
        tot = s[[f for f, _ in FEATS]].abs().mean()
        pct = 100 * tot / tot.sum()
        for i, (f, lab) in enumerate(FEATS):
            y = len(FEATS) - 1 - i + rng.uniform(-0.28, 0.28, len(s))
            if f == "district_aggregate":
                ax.scatter(s[f], y, s=5, color="#8a8984", lw=0, alpha=0.8, zorder=2)
            else:
                z = v[f].astype(float).to_numpy()
                span = z.max() - z.min()
                ax.scatter(
                    s[f],
                    y,
                    s=5,
                    c=(z - z.min()) / span if span > 0 else np.full(len(z), 0.5),
                    cmap=VAL,
                    vmin=0,
                    vmax=1,
                    lw=0,
                    zorder=2,
                )
            ax.text(
                1.0,
                len(FEATS) - 1 - i,
                f"{pct[f]:.0f}%",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.2,
                color=MUTED,
            )
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        ax.set_ylim(-0.6, len(FEATS) - 0.4)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        lim = np.abs(s[[f for f, _ in FEATS]].to_numpy()).max() * 1.08
        ax.set_xlim(-lim, lim * 1.25)
        ax.set_yticks(range(len(FEATS)))
        ax.set_yticklabels([l for _, l in FEATS][::-1] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_title(f'({"abc"[j]}) {CROP_LABEL[c]}', loc="left", fontweight="bold", pad=5)
        if j == 1:
            ax.set_xlabel(
                "SHAP value: contribution to predicted yield (t/ha); right-hand number = share of mean |SHAP|"
            )
    cax = fig.add_axes([0.945, 0.3, 0.012, 0.45])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=VAL), cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.3)
    cb.set_label("Predictor value", fontsize=6.5, labelpad=-6)
    save(fig, "FigS2_shap")


def figs3():
    """Fig. S3 - RAWSI."""
    t3 = pd.read_csv(T / "TableS4_rawsi.csv").sort_values("rank")
    yy = np.arange(len(t3))[::-1]
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(W, 3.5),
        sharey=True,
        gridspec_kw=dict(
            width_ratios=[1, 1, 1, 1.15, 1.5], wspace=0.1, left=0.15, right=0.99, bottom=0.14, top=0.9
        ),
    )
    for ax, (col, title, colr) in zip(
        axes[:4],
        [
            ("groundwater_component", "Groundwater trend", BLUES[2]),
            ("demand_component", "Crop-mix demand", BLUES[2]),
            ("deficit_component", "Climate deficit", BLUES[2]),
            ("RAWSI", "RAWSI (mean of three)", BLUES[4]),
        ],
    ):
        ax.barh(yy, t3[col], height=0.62, color=colr, edgecolor="white", linewidth=0.4)
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=7.5, pad=4)
    for yi, v in zip(yy, t3.RAWSI):
        axes[3].text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=6.2)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([f"{r}  {d}" for r, d in zip(t3["rank"], t3.district)])
    axes[1].set_xlabel("Normalised component (0–1)")
    ax = axes[4]
    ax.hlines(yy, t3.best_rank_20_variants, t3.worst_rank_20_variants, color=MUTED, lw=1.6)
    ax.plot(t3["rank"], yy, "o", ms=4.5, color=BLUES[5], mec="white", mew=0.6, zorder=3, label="Primary rank")
    ax.plot([], [], color=MUTED, lw=1.6, label="Range over 20 variants")
    ax.set_xlim(0.3, 16.7)
    ax.set_xticks([1, 5, 10, 16])
    ax.axvline(5.5, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.text(5.7, yy[0] + 0.1, "top 5", ha="left", va="center", fontsize=6.2, color=MUTED)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Rank stability", fontsize=7.5, pad=4)
    ax.set_xlabel("Rank (1 = most scarce)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 0.02),
        handlelength=1.5,
        fontsize=6.0,
        framealpha=0.9,
        edgecolor="none",
    )
    save(fig, "FigS3_rawsi")


def figs4():
    """Fig. S4 - forecast evaluation."""
    t2 = pd.read_csv(T / "Table2_forecast_accuracy.csv")
    fc = pd.read_csv(F / "decision_time_forecasts.csv")
    MCOL = {
        "Elastic Net": "EN_cutoff2",
        "Persistence": "Persistence_cutoff2",
        "Climatology": "Climatology_cutoff2",
    }
    MMARK = {"Elastic Net": "o", "Persistence": "D", "Climatology": "s"}
    fig = plt.figure(figsize=(W, 5.3))
    gs = fig.add_gridspec(
        2, 6, height_ratios=[1.12, 1], hspace=0.42, wspace=1.2, left=0.07, right=0.985, bottom=0.085, top=0.94
    )
    for j, c in enumerate(CROPS):
        ax = fig.add_subplot(gs[0, 2 * j : 2 * j + 2])
        g = fc[fc.crop == c]
        lo, hi = min(g.yield_t_ha.min(), g.EN_cutoff2.min(), g.Persistence_cutoff2.min()), max(
            g.yield_t_ha.max(), g.EN_cutoff2.max(), g.Persistence_cutoff2.max()
        )
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.plot(
            g.yield_t_ha,
            g.Persistence_cutoff2,
            "D",
            ms=3.4,
            mfc="white",
            mec=MODEL_COL["Persistence"],
            mew=0.9,
            zorder=2,
            label="Persistence",
        )
        ax.plot(
            g.yield_t_ha,
            g.EN_cutoff2,
            "o",
            ms=3.8,
            color=MODEL_COL["Elastic Net"],
            mec="white",
            mew=0.4,
            zorder=3,
            label="Elastic Net",
        )
        r = {m: t2[(t2.crop == c) & (t2.model == m)].iloc[0] for m in ("Elastic Net", "Persistence")}
        ax.text(
            0.04,
            0.97,
            f"Elastic Net  RMSE {r['Elastic Net'].RMSE_t_ha:.2f}, R² {fmt_minus(r['Elastic Net'].R2)}\nPersistence  RMSE {r['Persistence'].RMSE_t_ha:.2f}, R² {fmt_minus(r['Persistence'].R2)}",
            transform=ax.transAxes,
            va="top",
            fontsize=6.2,
            linespacing=1.4,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi + 0.17 * (hi - lo))
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_axisbelow(True)  # headroom so that the note covers no point
        ax.set_xlabel("Observed yield (t/ha)")
        ax.set_title(f'({"abc"[j]}) {CROP_LABEL[c]}', loc="left", fontweight="bold", pad=5)
        if j == 0:
            ax.set_ylabel("Forecast yield (t/ha)")
        if j == 2:
            ax.legend(loc="lower right", handletextpad=0.2, fontsize=6.4)
    ax = fig.add_subplot(gs[1, 0:3])
    xs = np.arange(3)
    wbar = 0.36
    for k, m in enumerate(["Elastic Net", "Persistence"]):
        v = [t2[(t2.crop == c) & (t2.model == m)].skill_vs_climatology.iloc[0] for c in CROPS]
        ax.bar(
            xs + (k - 0.5) * (wbar + 0.03),
            v,
            wbar,
            color=MODEL_COL[m],
            hatch=MODEL_HATCH[m],
            edgecolor="white",
            linewidth=0.6,
            label=m,
        )
        for xi, vi in zip(xs + (k - 0.5) * (wbar + 0.03), v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([CROP_LABEL[c] for c in CROPS])
    ax.set_ylim(0, 0.64)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylabel("Skill vs climatology (0 = no better)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=1, bbox_to_anchor=(-0.02, 1.02))
    panel(ax, "d", "Skill, 2023–2025 (48 forecasts per crop)")
    ax = fig.add_subplot(gs[1, 3:6])
    pos = []
    data = []
    cols = []
    for j, c in enumerate(CROPS):
        g = fc[fc.crop == c]
        for k, m in enumerate(MCOL):
            pos.append(j * 4 + k)
            data.append((100 * (g[MCOL[m]] - g.yield_t_ha) / g.yield_t_ha).to_numpy())
            cols.append(MODEL_COL[m])
    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", ms=2, mfc=MUTED, mec="none"),
        medianprops=dict(color=INK, lw=1),
        whiskerprops=dict(color=MUTED, lw=0.7),
        capprops=dict(color=MUTED, lw=0.7),
        boxprops=dict(linewidth=0.5, edgecolor="white"),
    )
    for patch, colr, k in zip(bp["boxes"], cols, range(len(cols))):
        patch.set_facecolor(colr)
        patch.set_hatch(list(MODEL_HATCH.values())[k % 3])
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks([1, 5, 9])
    ax.set_xticklabels([CROP_LABEL[c] for c in CROPS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Forecast error (% of observed yield)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    for m in MCOL:
        ax.bar([np.nan], [np.nan], color=MODEL_COL[m], hatch=MODEL_HATCH[m], edgecolor="white", label=m)
    ax.legend(loc="upper left", ncol=3, handlelength=1.3, columnspacing=1, bbox_to_anchor=(-0.02, 1.02))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    panel(ax, "e", "Error distribution by model")
    save(fig, "FigS4_forecasts")


def figs5():
    """Fig. S5 - potato price risk."""
    gm = pd.read_csv(X / "ext_12_gross_margin_objective.csv")
    sw = pd.read_csv(X / "ext_14_potato_price_sweep.csv")
    sw = sw[sw.plan == "free"]
    econ = pd.read_csv(I / "crop_economics_scenarios.csv").set_index(["crop", "item"]).value
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(W, 2.9), gridspec_kw=dict(wspace=0.26, left=0.075, right=0.985, bottom=0.19, top=0.88)
    )
    for c, col, mk in zip(CROPS_A + ["Fallow"], LAND_COLS, ["o", "s", "^", "v", "D"]):
        colr = "#8a8984" if c == "Fallow" else CROP_COL[c]
        ax.plot(sw.potato_price_Tk_per_kg, sw[col], color=colr, lw=1.6, marker=mk, ms=2.6, markevery=4)
        ax.text(
            34.4, sw[col].iloc[-1], "Fallow" if c == "Fallow" else CROP_LABEL[c], va="center", fontsize=6.6
        )
    for key, lab in [("price_low", "low"), ("price_mid", "mid"), ("price_high", "high")]:
        p = float(econ[("Potato", key)])
        ax.axvline(p, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.text(p + 0.3, 62, lab, ha="left", va="center", fontsize=MIN_PT, color=MUTED)
    ax.set_xlim(6, 39.5)
    ax.set_xticks([6, 10, 14, 18, 22, 26, 30, 34])
    ax.set_ylim(0, 65)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_bounds(6, 34)
    ax.set_xlabel("Potato harvest-time price (Tk per kg)")
    ax.set_ylabel("Share of the land pool (%)")
    ax.xaxis.set_label_coords(0.42, -0.15)
    panel(ax, "a", "Net-return plan (free fallow) vs potato price, budget 0.80")
    PLANS = [
        ("gross_margin_potato_mid", "Net-return plan", "o"),
        ("food_energy", "Food-energy plan", "s"),
        ("gross_margin_potato_low", "No-potato plan", "^"),
    ]
    xs = np.arange(3)
    for k, (o, lab, mk) in enumerate(PLANS):
        r = gm[np.isclose(gm.budget_ratio, 0.8) & (gm.plan == "free") & (gm.objective == o)].iloc[0]
        v = [r[f"gross_margin_billion_Tk_potato_{s}"] for s in ("low", "mid", "high")]
        ax2.plot(
            xs,
            v,
            color=[INK, MUTED, BLUES[4]][k],
            lw=1.2,
            marker=mk,
            ms=4.5,
            mfc=["white", MUTED, BLUES[4]][k],
            mew=1,
            label=lab,
        )
        for xi, vi in zip(xs, v):
            if xi != 1:
                ax2.text(
                    xi + (0.1 if xi == 0 else -0.1),
                    vi + (4 if k == 2 else -5 if k == 0 else 3),
                    fmt_minus(vi, 0),
                    ha="left" if xi == 0 else "right",
                    va="center",
                    fontsize=6.3,
                )
    ax2.axhline(0, color=MUTED, lw=0.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [
            f'Low\nTk {econ[("Potato","price_low")]:g}',
            f'Mid\nTk {econ[("Potato","price_mid")]:g}',
            f'High\nTk {econ[("Potato","price_high")]:g}',
        ]
    )
    ax2.set_xlim(-0.3, 2.3)
    ax2.set_ylabel("Regional net return (billion Tk)")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", handlelength=1.6)
    panel(ax2, "b", "What each plan earns at three potato prices")
    save(fig, "FigS5_potato_price")


def figs6():
    """Fig. S6 - decision value + shadow price maps."""
    t7 = pd.read_csv(T / "Table4_decision_value.csv")
    t7 = t7[t7.plan == "free"]
    sh = pd.read_csv(X / "ext_15_shadow_prices.csv")
    sh = sh[
        (sh.objective == "gross_margin") & (sh.plan == "free") & np.isclose(sh.budget_ratio, 0.8)
    ].set_index("district")
    BCOL = {
        "Persistence": MODEL_COL["Persistence"],
        "Climatology": MODEL_COL["Climatology"],
        "District trend": "#008300",
    }
    BMK = {"Persistence": "o", "Climatology": "D", "District trend": "s"}
    fig = plt.figure(figsize=(W, 3.9))
    axa = None
    for k_obj, (objective, olab, unit, box) in enumerate(
        (
            ("food_energy", "Food energy", "pp of oracle", [0.165, 0.60, 0.23, 0.30]),
            ("gross_margin", "Net return", "Tk per ha", [0.165, 0.14, 0.23, 0.30]),
        )
    ):
        ax = fig.add_axes(box)
        axa = axa or ax
        ypos = []
        labels = []
        ax.axvline(0, color=MUTED, lw=0.6)
        for j, rho in enumerate((0.8, 0.9, 1.0)):
            y = -j
            for k, comp in enumerate(("Persistence", "Climatology", "District trend")):
                r = t7[
                    (t7.objective == objective) & np.isclose(t7.budget_ratio, rho) & (t7.EN_vs == comp)
                ].iloc[0]
                yy_ = y - (k - 1) * 0.24
                ax.hlines(yy_, r.bootstrap95_low, r.bootstrap95_high, color=BCOL[comp], lw=1.1)
                ax.plot(
                    r.mean_difference,
                    yy_,
                    BMK[comp],
                    ms=3.8,
                    color=BCOL[comp],
                    mec="white",
                    mew=0.5,
                    zorder=3,
                )
            labels.append(f"Budget {rho:.2f}")
            ypos.append(y)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"{olab}: EN minus benchmark ({unit})", fontsize=6.6)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_ylim(-2.6, 0.6)
        if k_obj == 0:
            for comp in ("Persistence", "Climatology", "District trend"):
                ax.plot([], [], BMK[comp], ms=3.8, color=BCOL[comp], label=f"vs {comp.lower()}")
    ax = axa
    ax.legend(
        loc="lower left",
        handletextpad=0.2,
        fontsize=5.9,
        ncol=3,
        bbox_to_anchor=(-0.62, 0.99),
        frameon=False,
        columnspacing=0.6,
    )
    ax.set_title("(a) Regret difference, free fallow", loc="left", fontweight="bold", pad=14, x=-0.62)
    for k, (col, cmap, vmax, title, label) in enumerate(
        [
            ("water_value_per_1000_m3", SEQ, 55, "Value of 1000 m³ more water", "thousand Tk per 1000 m³"),
            (
                "boro_floor_cost_per_ha",
                ORANGES,
                350,
                "Cost of one more hectare of Boro",
                "thousand Tk per hectare",
            ),
        ]
    ):
        axm = fig.add_axes([0.435 + 0.285 * k, 0.2, 0.27, 0.7])
        vmax = max(
            vmax,
            (
                float(np.ceil(sh[col].max() / 1e3 / 50) * 50)
                if sh[col].max() > 1e3
                else float(np.ceil(sh[col].max() / 10) * 10)
            ),
        )
        small_map(axm, sh[col], cmap, 0, vmax, f'({"bc"[k]}) {title}', label + ", budget 0.80", "{:.0f}")
    save(fig, "FigS6_decision_value_dual_values")


FIGURES = {
    "Fig1": (fig1, "Fig1_study_area"),
    "Fig2": (fig2, "Fig2_framework"),
    "Fig3": (fig3, "Fig3_feasibility"),
    "Fig4": (fig4, "Fig4_plans"),
    "Fig5": (fig5, "Fig5_robust_budgets"),
    "Fig6": (fig6, "Fig6_awd_water_value"),
    "Fig7": (fig7, "Fig7_low_water_crops"),
    "Fig8": (fig8, "Fig8_sensitivity"),
    "FigS1": (figs1, "FigS1_crop_data"),
    "FigS2": (figs2, "FigS2_shap"),
    "FigS3": (figs3, "FigS3_rawsi"),
    "FigS4": (figs4, "FigS4_forecasts"),
    "FigS5": (figs5, "FigS5_potato_price"),
    "FigS6": (figs6, "FigS6_decision_value_dual_values"),
}

if __name__ == "__main__":
    wanted = args.only or list(FIGURES)
    unknown = set(wanted) - set(FIGURES)
    if unknown:
        raise SystemExit(f"Unknown figure(s): {sorted(unknown)}; choose from {list(FIGURES)}")
    for key in wanted:
        draw, file_name = FIGURES[key]
        draw()
        print("Figure written:", file_name, flush=True)
