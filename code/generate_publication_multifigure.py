"""Generate a publication-ready multi-panel figure from current analysis outputs.

Panels
------
(a) Full-extent baseline accessibility map
(b) RP100 hotspot zoom with structural-operational risk corridors
(c) Scenario comparison for weighted vs uniform accessibility metrics

The script is designed to run locally or in Google Colab with only pandas,
geopandas, and matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTED_DIR = ROOT / "outputs" / "pop_building_weighted_rp"
DEFAULT_RP_DIR = ROOT / "outputs" / "rp_fire_response"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a WRR-style multi-panel figure.")
    parser.add_argument("--weighted-dir", default=str(DEFAULT_WEIGHTED_DIR))
    parser.add_argument("--rp-dir", default=str(DEFAULT_RP_DIR))
    parser.add_argument(
        "--output-figure",
        default=str(DEFAULT_WEIGHTED_DIR / "figure_publication_multifigure.png"),
    )
    parser.add_argument(
        "--output-code",
        default=str(DEFAULT_WEIGHTED_DIR / "colab_publication_multifigure.py"),
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_station_summary(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def load_inputs(weighted_dir: Path, rp_dir: Path) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    weighted_points = gpd.read_file(weighted_dir / "weighted_demand_points.gpkg")
    corridors = gpd.read_file(weighted_dir / "priority_defense_corridors.gpkg")
    facilities = gpd.read_file(rp_dir / "fire_facilities.gpkg")
    sigungu = gpd.read_file(rp_dir / "seoul_sigungu.gpkg")
    baseline_nodes = pd.read_csv(rp_dir / "baseline_nodes.csv")
    rp_nodes = pd.read_csv(rp_dir / "rp_nodes.csv")
    table1 = pd.read_csv(weighted_dir / "table1_population_building_weighted.csv")
    gap = pd.read_csv(weighted_dir / "gap_analysis_population_building_weighted.csv")
    corr = pd.read_csv(weighted_dir / "structural_operational_correlation.csv")
    station = load_station_summary(weighted_dir / "fire_station_accessibility_summary.csv")

    return {
        "weighted_points": weighted_points,
        "corridors": corridors,
        "facilities": facilities,
        "sigungu": sigungu,
        "baseline_nodes": baseline_nodes,
        "rp_nodes": rp_nodes,
        "table1": table1,
        "gap": gap,
        "corr": corr,
        "station": station,
    }


def prepare_demand_results(
    weighted_points: gpd.GeoDataFrame,
    baseline_nodes: pd.DataFrame,
    rp_nodes: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    base = weighted_points.merge(
        baseline_nodes.rename(
            columns={"travel_time_min": "baseline_time_min", "isolated": "baseline_isolated"}
        ),
        on="node_id",
        how="left",
    )
    rp100 = rp_nodes[rp_nodes["scenario"] == "RP100"].copy()
    flood = weighted_points.merge(
        rp100[
            [
                "node_id",
                "baseline_time_min",
                "scenario_time_min",
                "scenario_isolated",
                "delay_min",
            ]
        ],
        on="node_id",
        how="left",
    )

    base = gpd.GeoDataFrame(base, geometry="geometry", crs=weighted_points.crs)
    flood = gpd.GeoDataFrame(flood, geometry="geometry", crs=weighted_points.crs)

    base["baseline_bin"] = pd.cut(
        base["baseline_time_min"],
        bins=[0, 3, 5, 10, np.inf],
        labels=["<=3 min", "3-5 min", "5-10 min", ">10 min"],
        include_lowest=True,
    )
    return base, flood


def clip_to_seoul(
    frame: gpd.GeoDataFrame,
    sigungu: gpd.GeoDataFrame,
    buffer_m: float = 200.0,
) -> gpd.GeoDataFrame:
    district = sigungu.to_crs(frame.crs)
    seoul_union = district.union_all()
    clipped = frame[frame.geometry.within(seoul_union.buffer(buffer_m))].copy()
    if clipped.empty:
        clipped = frame.copy()
    return gpd.clip(clipped, district)


def compute_hotspot_extent(rp100_corridors: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    focus = rp100_corridors.nlargest(max(100, int(len(rp100_corridors) * 0.005)), "risk_score").copy()
    mids = focus.geometry.interpolate(0.5, normalized=True)
    x = mids.x.to_numpy()
    y = mids.y.to_numpy()
    w = focus["risk_score"].to_numpy()
    if w.sum() == 0:
        cx, cy = float(np.mean(x)), float(np.mean(y))
    else:
        cx, cy = float(np.average(x, weights=w)), float(np.average(y, weights=w))
    half_width = 7000
    half_height = 5500
    return cx - half_width, cx + half_width, cy - half_height, cy + half_height


def draw_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )


def plot_full_extent(
    ax: plt.Axes,
    sigungu: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    baseline: gpd.GeoDataFrame,
) -> None:
    district = sigungu.to_crs(baseline.crs)
    facilities = facilities.to_crs(baseline.crs)
    baseline = clip_to_seoul(baseline, sigungu)
    district.boundary.plot(ax=ax, color="#9aa0a6", linewidth=0.4, zorder=1)

    palette = {
        "<=3 min": "#d7ecfa",
        "3-5 min": "#73add1",
        "5-10 min": "#2f6c99",
        ">10 min": "#163a5f",
    }
    for label, color in palette.items():
        part = baseline[baseline["baseline_bin"] == label]
        if not part.empty:
            part.plot(
                ax=ax,
                color=color,
                markersize=5 + 10 * part["demand_weight"].rank(pct=True),
                alpha=0.72,
                zorder=2,
            )

    facilities.plot(
        ax=ax,
        color="black",
        markersize=34,
        marker="^",
        alpha=0.95,
        zorder=3,
    )
    xmin, ymin, xmax, ymax = district.total_bounds
    pad_x = (xmax - xmin) * 0.02
    pad_y = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_title("Baseline accessibility across Seoul")
    ax.set_axis_off()

    handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in palette.items()]
    handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Fire facility"))
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        frameon=True,
        title="Weighted demand travel time",
        borderaxespad=0.4,
    )


def plot_hotspot(
    ax: plt.Axes,
    sigungu: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    flood: gpd.GeoDataFrame,
    rp100_corridors: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
) -> None:
    xmin, xmax, ymin, ymax = extent
    district = sigungu.to_crs(flood.crs)
    facilities = facilities.to_crs(flood.crs)
    district = district.cx[xmin:xmax, ymin:ymax]
    demand = flood.cx[xmin:xmax, ymin:ymax].copy()
    corridors = rp100_corridors.cx[xmin:xmax, ymin:ymax].copy()
    facilities = facilities.cx[xmin:xmax, ymin:ymax].copy()

    district.boundary.plot(ax=ax, color="#9aa0a6", linewidth=0.5, zorder=1)

    delay_vmax = max(0.5, float(demand["delay_min"].quantile(0.98))) if not demand.empty else 1.0
    risk_top = corridors[corridors["risk_score"] >= corridors["risk_score"].quantile(0.90)].copy() if not corridors.empty else corridors
    risk_vmax = max(0.1, float(risk_top["risk_score"].quantile(0.98))) if not risk_top.empty else 1.0

    if not demand.empty:
        demand["marker_size"] = 8 + 30 * demand["demand_weight"].rank(pct=True)
        demand.plot(
            ax=ax,
            column="delay_min",
            cmap="Blues",
            vmin=0.0,
            vmax=delay_vmax,
            markersize=demand["marker_size"],
            alpha=0.58,
            zorder=2,
        )

    if not corridors.empty:
        top = risk_top
        base = corridors[corridors["risk_score"] < corridors["risk_score"].quantile(0.90)].copy()
        if not base.empty:
            base.plot(ax=ax, color="#f4b183", linewidth=0.7, alpha=0.45, zorder=3)
        if not top.empty:
            top.plot(
                ax=ax,
                column="risk_score",
                cmap="inferno",
                vmin=0.0,
                vmax=risk_vmax,
                linewidth=1.8,
                alpha=0.95,
                zorder=4,
            )

    if not facilities.empty:
        facilities.plot(
            ax=ax,
            color="black",
            markersize=42,
            marker="^",
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
        label_col = "서ㆍ센터ID" if "서ㆍ센터ID" in facilities.columns else None
        name_col = "서ㆍ센터명" if "서ㆍ센터명" in facilities.columns else None
        if label_col:
            top_fac = facilities.copy()
            top_fac["dist_to_center"] = np.sqrt(
                (top_fac.geometry.x - (xmin + xmax) / 2.0) ** 2
                + (top_fac.geometry.y - (ymin + ymax) / 2.0) ** 2
            )
            selected = top_fac.nsmallest(3, "dist_to_center").copy()
            selected["label_num"] = np.arange(1, len(selected) + 1)
            x_offsets = [6, 6, -12]
            y_offsets = [6, -10, 8]
            for idx, row in enumerate(selected.itertuples(index=False)):
                ax.annotate(
                    str(int(row.label_num)),
                    (row.geometry.x, row.geometry.y),
                    xytext=(x_offsets[idx % len(x_offsets)], y_offsets[idx % len(y_offsets)]),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#111111",
                    bbox={"facecolor": "white", "edgecolor": "#666666", "alpha": 0.9, "pad": 0.6},
                )

    divider = make_axes_locatable(ax)
    cax_risk = divider.append_axes("right", size="3.2%", pad=0.04)
    cax_delay = divider.append_axes("right", size="3.2%", pad=0.78)

    sm_risk = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=colors.Normalize(vmin=0.0, vmax=risk_vmax),
    )
    sm_risk.set_array([])
    cb_risk = ax.figure.colorbar(sm_risk, cax=cax_risk)
    cb_risk.set_label("Priority corridor score", rotation=270, labelpad=16)

    sm_delay = plt.cm.ScalarMappable(
        cmap="Blues",
        norm=colors.Normalize(vmin=0.0, vmax=delay_vmax),
    )
    sm_delay.set_array([])
    cb_delay = ax.figure.colorbar(sm_delay, cax=cax_delay)
    cb_delay.set_label("RP 100 delay (min)", rotation=270, labelpad=16)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("RP100 hotspot: delay and priority corridors")
    ax.set_axis_off()


def plot_scenario_comparison(
    container_ax: plt.Axes,
    table1: pd.DataFrame,
    gap: pd.DataFrame,
    corr: pd.DataFrame,
) -> None:
    container_ax.set_axis_off()
    sub = GridSpec(
        2,
        1,
        figure=container_ax.figure,
        left=container_ax.get_position().x0,
        right=container_ax.get_position().x1,
        bottom=container_ax.get_position().y0,
        top=container_ax.get_position().y1,
        hspace=0.28,
    )
    ax1 = container_ax.figure.add_subplot(sub[0])
    ax2 = container_ax.figure.add_subplot(sub[1], sharex=ax1)

    table = table1.sort_values("return_period").copy()
    x = table["return_period"]

    ax1.plot(x, table["uniform_mean_delay_min"], color="#7f8c8d", marker="o", linewidth=1.8, label="Uniform delay")
    ax1.plot(x, table["weighted_mean_delay_min"], color="#c44536", marker="o", linewidth=2.2, label="Weighted delay")
    ax1.set_ylabel("Mean delay (min)")
    ax1.set_title("Scenario comparison: uniform vs population-weighted performance")
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.legend(loc="upper left", frameon=False, ncol=2)

    ax2.plot(x, table["uniform_coverage_5"], color="#7f8c8d", marker="s", linewidth=1.8, linestyle="--", label="Uniform 5-min coverage")
    ax2.plot(x, table["weighted_coverage_5"], color="#0b3954", marker="s", linewidth=2.2, label="Weighted 5-min coverage")
    ax2.set_xlabel("Return period")
    ax2.set_ylabel("5-min coverage")
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.legend(loc="lower left", frameon=False)

    rp100_gap = gap.sort_values("return_period").iloc[-1]
    rp100_corr = corr.sort_values("return_period").iloc[-1]
    note = (
        f"RP100 weighted-uniform delay gap: {rp100_gap['weighted_minus_uniform_delay_min']:.3f} min\n"
        f"High-vs-low demand delay: {rp100_gap['high_weight_delay_min']:.3f} vs "
        f"{rp100_gap['low_weight_delay_min']:.3f} min\n"
        f"Spearman rho (ΔIntegration, weighted delay): "
        f"{rp100_corr['corr_dintegration_weighted_delay']:.3f}"
    )
    ax2.text(
        0.995,
        1.08,
        note,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95, "pad": 4},
        clip_on=False,
    )


def build_colab_script(output_path: Path) -> None:
    script = """# Publication figure generator for Google Colab
!pip install geopandas matplotlib pandas pyogrio shapely fiona

from pathlib import Path
import sys

ROOT = Path('/content/New project/backend')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

!python /content/New\\ project/backend/scripts/generate_publication_multifigure.py \\
  --weighted-dir '/content/New project/backend/outputs/pop_building_weighted_rp' \\
  --rp-dir '/content/New project/backend/outputs/rp_fire_response' \\
  --output-figure '/content/New project/backend/outputs/pop_building_weighted_rp/figure_publication_multifigure.png'
"""
    output_path.write_text(script, encoding="utf-8")


def generate_figure(data: dict[str, pd.DataFrame | gpd.GeoDataFrame], output_figure: Path) -> None:
    setup_style()

    baseline, flood = prepare_demand_results(
        data["weighted_points"],
        data["baseline_nodes"],
        data["rp_nodes"],
    )
    rp100_corridors = data["corridors"][data["corridors"]["scenario"] == "RP100"].copy()
    extent = compute_hotspot_extent(rp100_corridors)

    fig = plt.figure(figsize=(15, 11), constrained_layout=False)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 0.95], width_ratios=[1, 1], hspace=0.18, wspace=0.14)

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    plot_full_extent(ax_a, data["sigungu"], data["facilities"], baseline)
    plot_hotspot(ax_b, data["sigungu"], data["facilities"], flood, rp100_corridors, extent)
    plot_scenario_comparison(ax_c, data["table1"], data["gap"], data["corr"])

    draw_panel_label(ax_a, "(a)")
    draw_panel_label(ax_b, "(b)")
    draw_panel_label(ax_c, "(c)")

    fig.suptitle(
        "Population-weighted fire accessibility under flood stress in Seoul",
        y=0.975,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.94, hspace=0.18, wspace=0.14)
    fig.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    weighted_dir = Path(args.weighted_dir)
    rp_dir = Path(args.rp_dir)
    output_figure = Path(args.output_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_code = Path(args.output_code)
    output_code.parent.mkdir(parents=True, exist_ok=True)

    data = load_inputs(weighted_dir, rp_dir)
    generate_figure(data, output_figure)
    build_colab_script(output_code)

    print(f"Saved figure to: {output_figure}")
    print(f"Saved Colab script to: {output_code}")


if __name__ == "__main__":
    main()
