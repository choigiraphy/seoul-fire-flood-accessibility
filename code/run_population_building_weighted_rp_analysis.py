"""Population-primary, building-refined weighted-demand RP analysis for Seoul."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import pyogrio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.observed_fire_response import TARGET_CRS, load_graph
from research.sci_fire_response_pipeline import edge_geometries, load_fire_facilities, load_seoul_boundary


POP_GRID_BASE_X = 150000.0
POP_GRID_BASE_Y = 500000.0
POP_GRID_CELL_M = 250.0
POP_GRID_CRS = "EPSG:5186"


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fix_mojibake(value: object) -> object:
    if isinstance(value, str):
        try:
            return value.encode("latin1").decode("cp949")
        except Exception:
            return value
    return value


def fix_mojibake_series(series: pd.Series) -> pd.Series:
    unique_values = pd.Series(series.dropna().unique())
    mapping = {value: fix_mojibake(value) for value in unique_values}
    return series.map(mapping).fillna(series)


def parse_grid_code(code: str):
    code = str(code).strip()
    if len(code) < 10:
        return None
    numeric = code[2:]
    if len(numeric) != 8 or not numeric.isdigit():
        return None
    x_code = int(numeric[:4])
    y_code = int(numeric[4:])
    x0 = POP_GRID_BASE_X + x_code * 10.0
    y0 = POP_GRID_BASE_Y + y_code * 10.0
    return box(x0, y0, x0 + POP_GRID_CELL_M, y0 + POP_GRID_CELL_M)


def load_population_grid(pop_csv: Path, seoul_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = pd.read_csv(pop_csv, encoding="cp949")
    frame["생활인구합계"] = pd.to_numeric(frame["생활인구합계"], errors="coerce")
    frame = frame.dropna(subset=["생활인구합계", "250M격자"]).copy()
    grouped = (
        frame.groupby(["행정동코드", "250M격자"], as_index=False)["생활인구합계"]
        .mean()
        .rename(columns={"생활인구합계": "pop_count"})
    )
    grouped["geometry"] = grouped["250M격자"].map(parse_grid_code)
    grouped = grouped[grouped["geometry"].notna()].copy()
    grid = gpd.GeoDataFrame(grouped, geometry="geometry", crs=POP_GRID_CRS)
    seoul_geom = seoul_boundary.to_crs(POP_GRID_CRS).iloc[0].geometry
    grid = grid[grid.intersects(seoul_geom)].copy()
    grid["centroid"] = grid.geometry.centroid
    return grid


def load_buildings(building_shp: Path, seoul_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # pyogrio is materially faster than the default engine for this 695k-feature layer.
    gdf = pyogrio.read_dataframe(
        building_shp,
        columns=["A4", "A9", "A11", "A12", "A14", "A15", "A16", "A17", "A18", "A27"],
    )
    gdf = gdf.set_crs("EPSG:5186", allow_override=True)
    # The source layer is already Seoul (administrative code 11), so we avoid
    # an extra full-layer spatial clip here for performance.
    gdf = gdf.copy()
    for col in ["A4", "A9", "A11"]:
        gdf[col] = fix_mojibake_series(gdf[col])
    for col in ["A12", "A14", "A15", "A16", "A17", "A18", "A27"]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    gdf["usage_type"] = gdf["A9"].fillna("미상")
    gdf["structure_type"] = gdf["A11"].fillna("미상")
    gdf["footprint_area"] = gdf.geometry.area
    gdf["area_proxy"] = pd.concat(
        [
            gdf["A15"].clip(lower=0),
            gdf["A14"].clip(lower=0),
            gdf["A12"].clip(lower=0),
            gdf["footprint_area"].clip(lower=0),
        ],
        axis=1,
    ).max(axis=1)
    gdf["geometry"] = gdf.geometry.centroid
    return gdf


def usage_multiplier(series: pd.Series) -> pd.Series:
    mapping = {
        "의료시설": 1.80,
        "노유자시설": 1.70,
        "공동주택": 1.25,
        "단독주택": 1.10,
        "교육연구시설": 1.20,
        "업무시설": 1.00,
        "제1종근린생활시설": 0.95,
        "제2종근린생활시설": 0.95,
        "숙박시설": 1.10,
        "판매시설": 1.00,
        "종교시설": 0.90,
        "공장": 0.85,
        "문화및집회시설": 1.05,
        "위험물저장및처리시설": 1.40,
    }
    return series.map(mapping).fillna(0.85)


def dominant_value(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "미상"
    modes = series.mode()
    return str(modes.iloc[0]) if not modes.empty else str(series.iloc[0])


def structure_multiplier(series: pd.Series) -> pd.Series:
    mapping = {
        "일반목구조": 1.35,
        "벽돌구조": 1.20,
        "블록구조": 1.15,
        "경량철골구조": 1.10,
        "일반철골구조": 1.05,
        "기타조적구조": 1.15,
        "철근콘크리트구조": 1.00,
        "철골철근콘크리트구조": 0.95,
        "철골콘크리트구조": 0.98,
    }
    return series.map(mapping).fillna(1.00)


def build_refined_demand(pop_grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, data_root: Path) -> gpd.GeoDataFrame:
    graph = load_graph(data_root)
    work = buildings.copy()
    log_area = np.log1p(work["area_proxy"].clip(lower=0))
    amin = log_area.min()
    amax = log_area.max()
    work["area_norm"] = 0.0 if amax == amin else (log_area - amin) / (amax - amin)
    work["usage_mult"] = usage_multiplier(work["usage_type"])
    work["structure_mult"] = structure_multiplier(work["structure_type"])
    work["refine_score"] = (0.35 + 0.65 * work["area_norm"]) * work["usage_mult"] * work["structure_mult"]

    # Avoid an expensive full spatial join by assigning each building centroid to
    # its 250 m population cell directly from projected coordinates.
    x_code = np.floor((work.geometry.x - POP_GRID_BASE_X) / POP_GRID_CELL_M).astype(int) * 25
    y_code = np.floor((work.geometry.y - POP_GRID_BASE_Y) / POP_GRID_CELL_M).astype(int) * 25
    work["250M격자"] = ["다사" + f"{xv:04d}{yv:04d}" for xv, yv in zip(x_code, y_code)]
    join = work.merge(
        pop_grid[["250M격자", "행정동코드", "pop_count"]],
        on="250M격자",
        how="inner",
    )

    if join.empty:
        raise RuntimeError("No buildings matched the 250m population grid codes.")

    score_sum = join.groupby("250M격자")["refine_score"].sum().rename("score_sum")
    join = join.merge(score_sum, on="250M격자", how="left")
    join["weight_share"] = np.where(join["score_sum"] > 0, join["refine_score"] / join["score_sum"], 0.0)
    join["wx"] = join.geometry.x * join["weight_share"]
    join["wy"] = join.geometry.y * join["weight_share"]

    grid_refined = join.groupby(["250M격자", "행정동코드"], as_index=False).agg(
        pop_count=("pop_count", "first"),
        mean_refine_score=("refine_score", "mean"),
        max_refine_score=("refine_score", "max"),
        area_proxy=("area_proxy", "sum"),
        usage_type=("usage_type", dominant_value),
        structure_type=("structure_type", dominant_value),
        x=("wx", "sum"),
        y=("wy", "sum"),
    )
    grid_refined = gpd.GeoDataFrame(
        grid_refined,
        geometry=gpd.points_from_xy(grid_refined["x"], grid_refined["y"], crs=POP_GRID_CRS),
        crs=POP_GRID_CRS,
    )

    covered = set(grid_refined["250M격자"].unique())
    fallback = pop_grid[~pop_grid["250M격자"].isin(covered)].copy()
    if not fallback.empty:
        fallback["mean_refine_score"] = 1.0
        fallback["max_refine_score"] = 1.0
        fallback["area_proxy"] = POP_GRID_CELL_M * POP_GRID_CELL_M
        fallback["usage_type"] = "격자대표"
        fallback["structure_type"] = "미상"
        fallback["geometry"] = fallback["centroid"]
        fallback = fallback[["250M격자", "행정동코드", "pop_count", "mean_refine_score", "max_refine_score", "area_proxy", "usage_type", "structure_type", "geometry"]]
    else:
        fallback = gpd.GeoDataFrame(
            columns=["250M격자", "행정동코드", "pop_count", "mean_refine_score", "max_refine_score", "area_proxy", "usage_type", "structure_type", "geometry"],
            geometry="geometry",
            crs=TARGET_CRS,
        )

    demand = gpd.GeoDataFrame(pd.concat([grid_refined, fallback], ignore_index=True), geometry="geometry", crs=POP_GRID_CRS).to_crs(TARGET_CRS)
    demand["vulnerability_score"] = 0.6 * demand["mean_refine_score"].fillna(1.0) + 0.4 * demand["max_refine_score"].fillna(1.0)
    vmin = demand["vulnerability_score"].min()
    vmax = demand["vulnerability_score"].max()
    demand["vulnerability_norm"] = 0.0 if vmax == vmin else (demand["vulnerability_score"] - vmin) / (vmax - vmin)
    demand["allocated_pop"] = demand["pop_count"]
    demand["demand_weight"] = demand["pop_count"] * (0.70 + 0.30 * demand["vulnerability_norm"])

    x = demand.geometry.x.to_numpy()
    y = demand.geometry.y.to_numpy()
    node_ids = []
    chunk = 50000
    for start in range(0, len(demand), chunk):
        end = min(start + chunk, len(demand))
        node_ids.extend(ox.distance.nearest_nodes(graph, X=x[start:end], Y=y[start:end]))
    demand["node_id"] = pd.Series(node_ids, index=demand.index).astype(int)
    return demand


def weighted_coverage(frame: pd.DataFrame, time_col: str, threshold: int) -> float:
    total = frame["demand_weight"].sum()
    covered = frame.loc[frame[time_col].fillna(np.inf) <= threshold, "demand_weight"].sum()
    return float(covered / total) if total else 0.0


def weighted_mean(frame: pd.DataFrame, value_col: str) -> float:
    subset = frame[frame[value_col].notna()].copy()
    total = subset["demand_weight"].sum()
    if subset.empty or total == 0:
        return np.nan
    return float(np.average(subset[value_col], weights=subset["demand_weight"]))


def assign_edges(demand: gpd.GeoDataFrame, data_root: Path) -> pd.DataFrame:
    graph = load_graph(data_root)
    edges = edge_geometries(graph)
    mids = edges[["edge_id", "geometry"]].copy()
    mids["geometry"] = mids.geometry.interpolate(0.5, normalized=True)
    joined = gpd.sjoin_nearest(
        demand[["demand_weight", "geometry"]],
        gpd.GeoDataFrame(mids, geometry="geometry", crs=TARGET_CRS),
        how="left",
        distance_col="edge_distance_m",
    )
    return joined.groupby("edge_id", as_index=False)["demand_weight"].sum().rename(columns={"demand_weight": "assigned_demand_weight"})


def summarize_scenarios(demand: gpd.GeoDataFrame, rp_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = rp_nodes.merge(demand[["node_id", "demand_weight", "allocated_pop", "usage_type"]], on="node_id", how="inner")
    rows = []
    gaps = []
    for scenario, frame in merged.groupby("scenario"):
        rp = int(frame["return_period"].iloc[0])
        rows.append(
            {
                "scenario": scenario,
                "return_period": rp,
                "uniform_mean_travel_time_min": float(frame["scenario_time_min"].mean(skipna=True)),
                "weighted_mean_travel_time_min": weighted_mean(frame, "scenario_time_min"),
                "uniform_mean_delay_min": float(frame["delay_min"].mean(skipna=True)),
                "weighted_mean_delay_min": weighted_mean(frame, "delay_min"),
                "uniform_coverage_5": float((frame["scenario_time_min"].fillna(np.inf) <= 5).mean()),
                "weighted_coverage_5": weighted_coverage(frame, "scenario_time_min", 5),
                "uniform_coverage_10": float((frame["scenario_time_min"].fillna(np.inf) <= 10).mean()),
                "weighted_coverage_10": weighted_coverage(frame, "scenario_time_min", 10),
                "uniform_coverage_15": float((frame["scenario_time_min"].fillna(np.inf) <= 15).mean()),
                "weighted_coverage_15": weighted_coverage(frame, "scenario_time_min", 15),
                "uniform_isolated_share": float(frame["scenario_isolated"].mean()),
                "weighted_isolated_share": float(np.average(frame["scenario_isolated"], weights=frame["demand_weight"])),
            }
        )
        high = frame["demand_weight"] >= frame["demand_weight"].quantile(0.75)
        low = frame["demand_weight"] <= frame["demand_weight"].quantile(0.25)
        gaps.append(
            {
                "scenario": scenario,
                "return_period": rp,
                "coverage_5_gap_pctp": (weighted_coverage(frame, "scenario_time_min", 5) - float((frame["scenario_time_min"].fillna(np.inf) <= 5).mean())) * 100.0,
                "weighted_minus_uniform_delay_min": weighted_mean(frame, "delay_min") - float(frame["delay_min"].mean(skipna=True)),
                "high_weight_delay_min": float(frame.loc[high, "delay_min"].mean(skipna=True)),
                "low_weight_delay_min": float(frame.loc[low, "delay_min"].mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(gaps)


def corridor_risk(demand: gpd.GeoDataFrame, rp_corridors: pd.DataFrame, data_root: Path) -> gpd.GeoDataFrame:
    graph = load_graph(data_root)
    edges = edge_geometries(graph)
    edge_weights = assign_edges(demand, data_root)
    merged = rp_corridors.merge(edges[["edge_id", "geometry"]], on="edge_id", how="left").merge(edge_weights, on="edge_id", how="left")
    merged["assigned_demand_weight"] = merged["assigned_demand_weight"].fillna(0.0)
    merged["weighted_delay"] = merged["segment_delay_min"].fillna(0.0) * merged["assigned_demand_weight"]
    for source, target in [("d_choice", "d_choice_norm"), ("d_integration", "d_integration_norm"), ("weighted_delay", "weighted_delay_norm")]:
        values = merged[source].fillna(0.0)
        vmin = values.min()
        vmax = values.max()
        merged[target] = 0.0 if vmax == vmin else (values - vmin) / (vmax - vmin)
    merged["risk_score"] = 0.50 * merged["weighted_delay_norm"] + 0.30 * merged["d_choice_norm"] + 0.20 * merged["d_integration_norm"]
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)


def structural_operational_correlation(risk: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in risk.groupby("scenario"):
        rows.append(
            {
                "scenario": scenario,
                "return_period": int(frame["return_period"].iloc[0]),
                "corr_dchoice_weighted_delay": frame["d_choice"].corr(frame["weighted_delay"], method="spearman"),
                "corr_dintegration_weighted_delay": frame["d_integration"].corr(frame["weighted_delay"], method="spearman"),
                "corr_sfrc_weighted_delay": frame["sfrc_score"].corr(frame["weighted_delay"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def station_accessibility_summary(demand: gpd.GeoDataFrame, data_root: Path, seoul_boundary: gpd.GeoDataFrame) -> pd.DataFrame:
    facilities = load_fire_facilities(data_root, seoul_boundary)
    id_col = "서ㆍ센터ID" if "서ㆍ센터ID" in facilities.columns else None
    name_col = "서ㆍ센터명" if "서ㆍ센터명" in facilities.columns else facilities.columns[0]
    type_col = "유형구분명" if "유형구분명" in facilities.columns else None
    assigned = gpd.sjoin_nearest(
        demand[["demand_weight", "geometry"]],
        facilities[[col for col in [id_col, name_col, type_col, "geometry"] if col is not None]],
        how="left",
        distance_col="facility_distance_m",
    )
    group_cols = ([id_col] if id_col else []) + [name_col] + ([type_col] if type_col else [])
    summary = assigned.groupby(group_cols, as_index=False).agg(
        assigned_demand_weight=("demand_weight", "sum"),
        mean_distance_m=("facility_distance_m", "mean"),
        demand_point_count=("demand_weight", "size"),
    )
    return summary.sort_values("assigned_demand_weight", ascending=False)


def plot_comparison(table1: pd.DataFrame, output_png: Path) -> None:
    table = table1.sort_values("return_period").copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(table["return_period"], table["uniform_mean_delay_min"], marker="o", label="Uniform")
    axes[0].plot(table["return_period"], table["weighted_mean_delay_min"], marker="o", label="Weighted")
    axes[0].set_title("Mean delay by scenario")
    axes[0].set_xlabel("Return period")
    axes[0].set_ylabel("Delay (min)")
    axes[0].legend()

    axes[1].plot(table["return_period"], table["uniform_coverage_5"], marker="o", label="Uniform")
    axes[1].plot(table["return_period"], table["weighted_coverage_5"], marker="o", label="Weighted")
    axes[1].set_title("5-min coverage by scenario")
    axes[1].set_xlabel("Return period")
    axes[1].set_ylabel("Coverage share")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def plot_risk_map(risk: gpd.GeoDataFrame, scenario: str, output_png: Path, output_html: Path) -> None:
    subset = risk[risk["scenario"] == scenario].copy()
    subset = subset[subset.geometry.notna()].copy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    subset.plot(ax=ax, column="risk_score", cmap="inferno", linewidth=1.2, legend=True, legend_kwds={"label": "Priority defense score"})
    ax.set_title(f"{scenario} priority defense corridors")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    subset_4326 = subset.to_crs(4326)
    center = subset_4326.geometry.union_all().centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(
        subset_4326[["geometry", "risk_score", "weighted_delay", "d_choice", "d_integration"]].to_json(),
        style_function=lambda _: {"color": "#b22222", "weight": 2.0, "opacity": 0.8},
        tooltip=folium.GeoJsonTooltip(fields=["risk_score", "weighted_delay", "d_choice", "d_integration"]),
    ).add_to(fmap)
    fmap.save(output_html)


def write_colab_script(output_path: Path) -> None:
    script = """# Colab-ready execution script
!pip install osmnx geopandas shapely pyproj folium networkx matplotlib pandas numpy

from pathlib import Path
import sys

ROOT = Path('/content/New project/backend')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

!python /content/New\\ project/backend/scripts/run_population_building_weighted_rp_analysis.py \\
  --data-root '/content/drive/Othercomputers/내 Mac/Keywest_JetDrive 1/W2 Data' \\
  --population-csv '/content/drive/Othercomputers/내 Mac/Keywest_JetDrive 1/W2 Data/250_LOCAL_RESD_20260313.csv' \\
  --building-shp '/content/drive/Othercomputers/내 Mac/Keywest_JetDrive 1/W2 Data/AL_D010_11_20260309/AL_D010_11_20260309.shp' \\
  --rp-output-dir '/content/New project/backend/outputs/rp_fire_response' \\
  --output-dir '/content/New project/backend/outputs/pop_building_weighted_rp'
"""
    output_path.write_text(script)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run population-primary, building-refined RP analysis.")
    parser.add_argument("--data-root", default="/Volumes/Keywest_JetDrive 1/W2 Data")
    parser.add_argument("--population-csv", default="/Volumes/Keywest_JetDrive 1/W2 Data/250_LOCAL_RESD_20260313.csv")
    parser.add_argument("--building-shp", default="/Volumes/Keywest_JetDrive 1/W2 Data/AL_D010_11_20260309/AL_D010_11_20260309.shp")
    parser.add_argument("--rp-output-dir", default=str(ROOT / "outputs" / "rp_fire_response"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pop_building_weighted_rp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = ensure_output_dir(Path(args.output_dir))
    _, seoul_boundary = load_seoul_boundary(data_root)

    t0 = time.time()
    print("Step 1/7: loading 250m population grid...", flush=True)
    pop_grid = load_population_grid(Path(args.population_csv), seoul_boundary)
    pop_grid.drop(columns=["centroid"]).to_file(output_dir / "population_grid_250m.gpkg", driver="GPKG")
    print(f"  population grids: {len(pop_grid):,} | elapsed {time.time() - t0:.1f}s", flush=True)

    print("Step 2/7: loading building layer...", flush=True)
    buildings = load_buildings(Path(args.building_shp), seoul_boundary)
    print(f"  buildings: {len(buildings):,} | elapsed {time.time() - t0:.1f}s", flush=True)

    print("Step 3/7: building refined population demand...", flush=True)
    demand = build_refined_demand(pop_grid, buildings, data_root)
    demand.to_file(output_dir / "weighted_demand_points.gpkg", driver="GPKG")
    print(f"  refined demand points: {len(demand):,} | elapsed {time.time() - t0:.1f}s", flush=True)

    print("Step 4/7: loading RP scenario outputs...", flush=True)
    rp_nodes = pd.read_csv(Path(args.rp_output_dir) / "rp_nodes.csv")
    rp_corridors = pd.read_csv(Path(args.rp_output_dir) / "rp_corridors.csv")
    print(f"  rp_nodes: {len(rp_nodes):,} | rp_corridors: {len(rp_corridors):,} | elapsed {time.time() - t0:.1f}s", flush=True)

    print("Step 5/7: scenario summaries and station accessibility...", flush=True)
    table1, gaps = summarize_scenarios(demand, rp_nodes)
    stations = station_accessibility_summary(demand, data_root, seoul_boundary)
    table1.to_csv(output_dir / "table1_population_building_weighted.csv", index=False)
    gaps.to_csv(output_dir / "gap_analysis_population_building_weighted.csv", index=False)
    stations.to_csv(output_dir / "fire_station_accessibility_summary.csv", index=False)
    print(f"  summary files written | elapsed {time.time() - t0:.1f}s", flush=True)

    print("Step 6/7: corridor risk and structural-operational correlation...", flush=True)
    risk = corridor_risk(demand, rp_corridors, data_root)
    corr = structural_operational_correlation(risk)
    corr.to_csv(output_dir / "structural_operational_correlation.csv", index=False)
    risk.to_file(output_dir / "priority_defense_corridors.gpkg", driver="GPKG")
    print(f"  corridor outputs written | elapsed {time.time() - t0:.1f}s", flush=True)

    print("Step 7/7: figures and Colab script...", flush=True)
    plot_comparison(table1, output_dir / "figure_uniform_vs_weighted_comparison.png")
    target = "RP100" if "RP100" in set(risk["scenario"]) else str(table1.sort_values("return_period").iloc[-1]["scenario"])
    plot_risk_map(risk, target, output_dir / f"figure_priority_corridors_{target.lower()}.png", output_dir / f"figure_priority_corridors_{target.lower()}.html")
    write_colab_script(output_dir / "colab_repro.py")

    print("Population-building weighted RP analysis complete.")
    print(output_dir)
    print(table1.round(4).to_string(index=False))
    print(corr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
