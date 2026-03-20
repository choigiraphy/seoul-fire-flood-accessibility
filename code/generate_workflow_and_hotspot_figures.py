"""Generate a workflow diagram and a hotspot zoom figure for the CEUS manuscript."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "pop_building_weighted_rp"
RPDIR = ROOT / "outputs" / "rp_fire_response"


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#333333") -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + width / 2, y + height * 0.36, body, ha="center", va="center", fontsize=9)


def workflow_diagram(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.05, 0.63),
        0.22,
        0.22,
        "Input data",
        "OSM road network\nFire facilities\nFlood traces + RP scenarios\n250 m population grid\nBuilding dataset",
        "#dceaf7",
    )
    add_box(
        ax,
        (0.39, 0.63),
        0.22,
        0.22,
        "Network construction",
        "Baseline graph\nSegment representation\nFacility-node snapping\nDemand-node allocation",
        "#e4f3df",
    )
    add_box(
        ax,
        (0.73, 0.63),
        0.22,
        0.22,
        "Scenario weighting",
        "Flood susceptibility\nRP-based edge penalties\nTravel-time updates\nScenario graph generation",
        "#fde7c8",
    )
    add_box(
        ax,
        (0.05, 0.22),
        0.22,
        0.22,
        "Weighted demand",
        "Population-primary demand\nBuilding refinement\nVulnerability normalization\nFinal demand weights",
        "#efe1f7",
    )
    add_box(
        ax,
        (0.39, 0.22),
        0.22,
        0.22,
        "Hybrid analytics",
        "Nearest-facility travel time\nDelay and coverage\nChoice / Integration loss\nStrategic corridor scoring",
        "#f9d9dc",
    )
    add_box(
        ax,
        (0.73, 0.22),
        0.22,
        0.22,
        "Outputs",
        "Weighted accessibility maps\nHotspot corridors\nScenario comparison\nStation-level summaries",
        "#e8ecef",
    )

    arrow_kw = dict(arrowstyle="-|>", lw=1.5, color="#444444", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(0.39, 0.74), xytext=(0.27, 0.74), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.73, 0.74), xytext=(0.61, 0.74), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.39, 0.33), xytext=(0.27, 0.33), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.73, 0.33), xytext=(0.61, 0.33), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.50, 0.44), xytext=(0.50, 0.63), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.84, 0.44), xytext=(0.84, 0.63), arrowprops=arrow_kw)

    ax.text(
        0.50,
        0.51,
        "Integrated workflow for flood-stressed fire accessibility analysis",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    fig.suptitle("Workflow of the hybrid geospatial framework", y=0.97, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def hotspot_zoom(output_path: Path) -> None:
    corridors = gpd.read_file(OUTDIR / "priority_defense_corridors.gpkg")
    facilities = gpd.read_file(RPDIR / "fire_facilities.gpkg").to_crs(corridors.crs)
    sigungu = gpd.read_file(RPDIR / "seoul_sigungu.gpkg").to_crs(corridors.crs)
    demand = gpd.read_file(OUTDIR / "weighted_demand_points.gpkg").to_crs(corridors.crs)

    rp100 = corridors[corridors["scenario"] == "RP100"].copy()
    # Zoom to the most important RP100 priority-defense corridor cluster,
    # rather than to the broader multi-panel hotspot used elsewhere.
    top = rp100.nlargest(200, "risk_score").copy()
    xmin, ymin, xmax, ymax = top.total_bounds
    pad_x = (xmax - xmin) * 0.18
    pad_y = (ymax - ymin) * 0.18
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    district = sigungu.cx[xmin:xmax, ymin:ymax]
    demand_sub = demand.cx[xmin:xmax, ymin:ymax].copy()
    rp_sub = rp100.cx[xmin:xmax, ymin:ymax].copy()
    facilities_sub = facilities.cx[xmin:xmax, ymin:ymax].copy()
    background = rp_sub[rp_sub["risk_score"] < rp_sub["risk_score"].quantile(0.90)].copy()
    focus = rp_sub[rp_sub["risk_score"] >= rp_sub["risk_score"].quantile(0.90)].copy()

    fig, ax = plt.subplots(figsize=(9, 8))
    district.boundary.plot(ax=ax, color="#9aa0a6", linewidth=0.5, zorder=1)

    demand_vmax = max(1.0, float(demand_sub["demand_weight"].quantile(0.98))) if not demand_sub.empty else 1.0
    risk_vmax = max(0.1, float(focus["risk_score"].quantile(0.98))) if not focus.empty else 1.0

    if not demand_sub.empty:
        demand_sub["pt_size"] = 8 + 26 * demand_sub["demand_weight"].rank(pct=True)
        demand_sub.plot(
            ax=ax,
            column="demand_weight",
            cmap="Blues",
            vmin=0.0,
            vmax=demand_vmax,
            markersize=demand_sub["pt_size"],
            alpha=0.42,
            zorder=2,
        )

    if not background.empty:
        background.plot(ax=ax, color="#f4b183", linewidth=0.8, alpha=0.40, zorder=3)
    if not focus.empty:
        focus.plot(
            ax=ax,
            column="risk_score",
            cmap="inferno",
            vmin=0.0,
            vmax=risk_vmax,
            linewidth=2.0,
            alpha=0.95,
            zorder=4,
        )

    facilities_sub.plot(
        ax=ax,
        color="black",
        marker="^",
        markersize=44,
        edgecolor="white",
        linewidth=0.4,
        zorder=5,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    divider = make_axes_locatable(ax)
    cax_risk = divider.append_axes("right", size="3.2%", pad=0.04)
    cax_demand = divider.append_axes("right", size="3.2%", pad=0.42)

    sm_risk = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=colors.Normalize(vmin=0.0, vmax=risk_vmax),
    )
    sm_risk.set_array([])
    cb_risk = fig.colorbar(sm_risk, cax=cax_risk)
    cb_risk.set_label("Priority corridor score", rotation=270, labelpad=14)

    sm_demand = plt.cm.ScalarMappable(
        cmap="Blues",
        norm=colors.Normalize(vmin=0.0, vmax=demand_vmax),
    )
    sm_demand.set_array([])
    cb_demand = fig.colorbar(sm_demand, cax=cax_demand)
    cb_demand.set_label("Demand weight", rotation=270, labelpad=14)

    ax.set_title("RP100 priority-defense corridors: zoom into the highest-risk cluster")
    ax.set_axis_off()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def priority_corridor_result_zoom(output_path: Path) -> None:
    corridors = gpd.read_file(OUTDIR / "priority_defense_corridors.gpkg")
    rp100 = corridors[corridors["scenario"] == "RP100"].copy()

    # Use the visually meaningful high-score cluster from the original result map.
    focus_cluster = rp100[rp100["risk_score"] >= rp100["risk_score"].quantile(0.998)].copy()
    xmin, ymin, xmax, ymax = focus_cluster.total_bounds
    pad_x = (xmax - xmin) * 0.18
    pad_y = (ymax - ymin) * 0.22
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    background = rp100.cx[xmin:xmax, ymin:ymax].copy()
    focus = focus_cluster.cx[xmin:xmax, ymin:ymax].copy()
    risk_vmax = max(0.1, float(focus["risk_score"].quantile(0.98))) if not focus.empty else 1.0

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    background.plot(ax=ax, color="#111111", linewidth=0.8, alpha=0.65, zorder=1)
    if not focus.empty:
        focus.plot(
            ax=ax,
            column="risk_score",
            cmap="inferno",
            vmin=0.0,
            vmax=risk_vmax,
            linewidth=2.1,
            alpha=1.0,
            zorder=2,
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.6%", pad=0.05)
    sm = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=colors.Normalize(vmin=0.0, vmax=risk_vmax),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Priority defense score", rotation=270, labelpad=16)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("RP100 priority defense corridors: result-focused zoom")
    ax.set_axis_off()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    workflow_diagram(OUTDIR / "figure_workflow_diagram.png")
    hotspot_zoom(OUTDIR / "figure_priority_corridors_rp100_zoom.png")
    priority_corridor_result_zoom(OUTDIR / "figure_priority_corridors_rp100_result_zoom.png")
    print(OUTDIR / "figure_workflow_diagram.png")
    print(OUTDIR / "figure_priority_corridors_rp100_zoom.png")
    print(OUTDIR / "figure_priority_corridors_rp100_result_zoom.png")


if __name__ == "__main__":
    main()
