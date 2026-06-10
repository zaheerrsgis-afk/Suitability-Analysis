import base64
import io
import json
from pathlib import Path

import folium
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import st_folium


# =====================================================
# Agro Climate Analysis
# Stable Streamlit Cloud Version
#
# Main performance strategy:
# - Rasters are stored on Google Drive.
# - Rasters download only if missing.
# - GeoTIFFs are converted to lightweight PNG overlays once.
# - PNG overlays are cached on disk in data/cache/.
# - Folium displays cached PNG overlays instead of reprocessing GeoTIFF every rerun.
# =====================================================


# =====================================================
# App Configuration
# =====================================================

st.set_page_config(
    page_title="Agro Climate Analysis",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RASTER_DIR = DATA_DIR / "rasters"
TABLE_DIR = DATA_DIR / "tables"
VECTOR_DIR = DATA_DIR / "vectors"
CACHE_DIR = DATA_DIR / "cache"

RASTER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCORECARD_FILE = TABLE_DIR / "Punjab_District_Risk_Scorecard.csv"
DISTRICTS_GEOJSON_FILE = VECTOR_DIR / "Punjab_Districts.geojson"

# GeoJSON is recommended. Shapefile fallback is optional.
DISTRICTS_SHP_FILE = VECTOR_DIR / "Punjab_Districts.shp"


# =====================================================
# =====================================================

GOOGLE_DRIVE_RASTER_FOLDER_URL = "https://drive.google.com/drive/folders/1UwUZ8xTzbLK116mvRbS3hVJoXlMLt2bi?usp=sharing"

PUBLIC_RASTER_FILES = [
    "Punjab_Environmental_Stress_Hotspots.tif",
    "Punjab_Drought_Prone_Areas.tif",
    "Punjab_Water_Storage_Stress.tif",
    "Punjab_Crop_Vegetation_Stress.tif",
    "Punjab_Heat_Stress_Hotspots.tif",
    "Punjab_Rainfall_Deficit_Zones.tif",
    "Punjab_Soil_Moisture_Stress.tif",
]


# =====================================================
# Layer Configuration
# =====================================================

STANDARD_LEGEND = {
    1: ("Low", "#1a9850"),
    2: ("Moderate", "#fee08b"),
    3: ("High", "#f46d43"),
    4: ("Very High", "#a50026"),
}

LAYER_CONFIG = {
    "Environmental Stress Hotspots": {
        "class_file": "Punjab_Environmental_Stress_Hotspots.tif",
        "score_col": "Environmental_Stress_Score",
        "risk_col": "Overall_Risk",
        "description": "Overall agro-climate stress combining water, drought, crop, heat and soil moisture indicators.",
        "legend_title": "Stress level",
        "legend": STANDARD_LEGEND,
    },
    "Drought-Prone Areas": {
        "class_file": "Punjab_Drought_Prone_Areas.tif",
        "score_col": "Drought_Prone_Score",
        "risk_col": "Drought_Tendency",
        "description": "Areas repeatedly exposed to drought-like conditions based on rainfall, vegetation, soil moisture and heat signals.",
        "legend_title": "Stress level",
        "legend": STANDARD_LEGEND,
    },
    "Water Storage Stress": {
        "class_file": "Punjab_Water_Storage_Stress.tif",
        "score_col": "Water_Storage_Stress_Score",
        "risk_col": "Water_Storage_Stress",
        "description": "Regional water storage stress based on GRACE/GRACE-FO total water storage anomaly. This is not exact groundwater depth.",
        "legend_title": "Stress level",
        "legend": STANDARD_LEGEND,
    },
    "Crop and Vegetation Stress": {
        "class_file": "Punjab_Crop_Vegetation_Stress.tif",
        "score_col": "Crop_Vegetation_Stress_Score",
        "risk_col": "Crop_Vegetation_Stress",
        "description": "Areas repeatedly showing below-normal crop and vegetation condition.",
        "legend_title": "Stress level",
        "legend": STANDARD_LEGEND,
    },
    "Heat Stress Hotspots": {
        "class_file": "Punjab_Heat_Stress_Hotspots.tif",
        "score_col": "Heat_Stress_Score",
        "risk_col": "Heat_Stress",
        "description": "Areas repeatedly exposed to higher land surface temperature conditions.",
        "legend_title": "Stress level",
        "legend": STANDARD_LEGEND,
    },
    "Rainfall Deficit Zones": {
        "class_file": "Punjab_Rainfall_Deficit_Zones.tif",
        "score_col": "Rainfall_Deficit_Score",
        "risk_col": "Rainfall_Deficit_Tendency",
        "description": "Areas repeatedly receiving below-normal rainfall.",
        "legend_title": "Stress level",
        "legend": STANDARD_LEGEND,
    },
    "Soil Moisture Stress": {
        "class_file": "Punjab_Soil_Moisture_Stress.tif",
        "score_col": "Soil_Moisture_Stress_Score",
        "risk_col": "Soil_Moisture_Stress",
        "description": "Areas repeatedly showing below-normal soil moisture condition.",
        "legend_title": "Stress level",
        "legend": STANDARD_LEGEND,
    },
}


# =====================================================
# CSS
# =====================================================

CUSTOM_CSS = """
<style>
.stApp {background: #f5f7fb;}
.block-container {padding-top: 0.75rem; padding-bottom: 1.5rem; max-width: 1800px;}
.panel {
    background: #ffffff; border: 1px solid #e7eaf0; border-radius: 18px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    padding: 14px; margin-bottom: 14px;
}
.step-title {
    display: flex; align-items: center; gap: 8px; color: #166534;
    font-weight: 800; font-size: 1.05rem; margin-bottom: 12px;
}
.step-badge {
    width: 22px; height: 22px; background: linear-gradient(135deg, #22c55e, #86efac);
    color: white; border-radius: 999px; display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.82rem; font-weight: 900;
}
.small-help {color: #9ca3af; font-size: 0.82rem; margin-top: 4px;}
.module-card {
    border: 1px solid #86efac; border-radius: 12px; padding: 12px;
    margin: 0; background: #f0fdf4;
}
.module-title {font-weight: 800; color: #1f2937;}
.module-sub {color: #9ca3af; font-size: 0.82rem;}
.map-card {
    background: #ffffff; border: 1px solid #86efac; border-radius: 18px;
    box-shadow: 0 8px 24px rgba(34,197,94,0.12); padding: 10px;
}
.map-header {position: relative; z-index: 2; margin-bottom: 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;}
.pill {
    background: rgba(255,255,255,0.92); border: 1px solid #e5e7eb;
    border-radius: 999px; padding: 5px 10px; font-size: 0.82rem;
    color: #374151; font-weight: 700;
}
.result-title {color: #166534; font-weight: 900; font-size: 1.05rem; margin-bottom: 10px;}
.alert-box {
    background: #fef3c7; border: 1px solid #f59e0b; border-radius: 12px;
    padding: 12px; color: #92400e; font-size: 0.92rem;
}
.metric-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 10px;}
.metric-mini {background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px;}
.metric-mini .label {color: #6b7280; font-size: 0.78rem;}
.metric-mini .value {color: #111827; font-size: 1.05rem; font-weight: 900;}
.legend-box {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px;}
.legend-row {display: flex; align-items: center; gap: 8px; margin: 8px 0; font-size: 0.92rem; color: #374151;}
.legend-swatch {width: 22px; height: 16px; border-radius: 5px; border: 1px solid rgba(0,0,0,0.15);}
h1, h2, h3, h4 {color: #111827;}
.stSelectbox label, .stSlider label, .stCheckbox label {color: #374151 !important; font-weight: 700;}
div[data-testid="stHeader"] {background: rgba(245,247,251,0.7);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================
# File Utilities
# =====================================================

def file_available(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def missing_required_rasters():
    return [f for f in PUBLIC_RASTER_FILES if not file_available(RASTER_DIR / f)]


@st.cache_resource(show_spinner=False)
def download_rasters_from_drive_once():
    """
    Download rasters once. If files already exist, no download happens.
    """
    if not missing_required_rasters():
        return True, "Rasters already exist."

    try:
        gdown.download_folder(
            url=GOOGLE_DRIVE_RASTER_FOLDER_URL,
            output=str(RASTER_DIR),
            quiet=True,
            use_cookies=False,
            remaining_ok=True
        )
        missing = missing_required_rasters()
        if missing:
            return False, "Missing rasters after download: " + ", ".join(missing)
        return True, "Rasters downloaded."
    except Exception as e:
        return False, f"Raster download failed: {e}"


def ensure_rasters_available():
    if not missing_required_rasters():
        return True

    with st.spinner("Downloading raster layers from Google Drive. This happens only once after app startup..."):
        ok, msg = download_rasters_from_drive_once()

    if not ok:
        st.error(msg)
        st.info("Make sure the Google Drive folder is public and contains the seven required TIFF files.")
        return False

    # Stop exactly once after download, so Streamlit Cloud settles and user reloads.
    marker = CACHE_DIR / ".download_complete"
    if not marker.exists():
        marker.write_text("downloaded", encoding="utf-8")
        st.success("Raster layers downloaded successfully. Please refresh the page once.")
        st.stop()

    return True


# =====================================================
# Data Loading
# =====================================================

@st.cache_data(show_spinner=False)
def load_scorecard(path: str) -> pd.DataFrame:
    p = Path(path)
    if not file_available(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, regex=True)]
    return df


def clean_geojson(gj):
    if not gj or gj.get("type") != "FeatureCollection":
        return gj
    clean = []
    for ft in gj.get("features", []):
        geom = ft.get("geometry")
        props = ft.get("properties", {})
        if not isinstance(geom, dict):
            continue
        if geom.get("type") in ["Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint"]:
            if geom.get("coordinates"):
                clean.append({"type": "Feature", "properties": props, "geometry": geom})
    gj["features"] = clean
    return gj


@st.cache_data(show_spinner=False)
def load_boundary_geojson(geojson_path: str, shp_path: str):
    gj_path = Path(geojson_path)
    shp = Path(shp_path)

    if file_available(gj_path):
        with open(gj_path, "r", encoding="utf-8") as f:
            return clean_geojson(json.load(f)), "GeoJSON"

    # Shapefile fallback is disabled unless geopandas is available.
    if file_available(shp):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shp)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")
            return clean_geojson(json.loads(gdf.to_json())), "Shapefile"
        except Exception:
            return None, "Shapefile found but geopandas unavailable"

    return None, "No boundary"


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_district_property(gj):
    if not gj or not gj.get("features"):
        return None
    props = gj["features"][0].get("properties", {})
    for f in ["DISTRICT", "District_Name", "District", "DIST_NAME", "NAME", "Name"]:
        if f in props:
            return f
    return None


def selected_district_feature(gj, district_name):
    if not gj or district_name == "All Punjab":
        return None
    prop = find_district_property(gj)
    if not prop:
        return None
    for ft in gj.get("features", []):
        if str(ft.get("properties", {}).get(prop, "")) == str(district_name):
            return ft
    return None


def geojson_for_single_feature(feature):
    return {"type": "FeatureCollection", "features": [feature]} if feature else None


def selected_or_province_shapes(gj, district_name):
    ft = selected_district_feature(gj, district_name)
    if ft:
        return [ft["geometry"]], ft
    if gj and gj.get("features"):
        return [f["geometry"] for f in gj.get("features", []) if f.get("geometry")], None
    return None, None


def extract_coords(geom):
    coords = []
    def walk(obj):
        if isinstance(obj, list):
            if len(obj) >= 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                coords.append((float(obj[0]), float(obj[1])))
            else:
                for x in obj:
                    walk(x)
    if geom and "coordinates" in geom:
        walk(geom["coordinates"])
    return coords


def selected_feature_bounds(feature):
    if not feature:
        return None
    coords = extract_coords(feature.get("geometry", {}))
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return [[min(ys), min(xs)], [max(ys), max(xs)]]


# =====================================================
# Dynamic Statistics
# =====================================================

def label_from_score(v):
    if pd.isna(v):
        return "Unknown"
    if v <= 0.25:
        return "Low"
    if v <= 0.50:
        return "Moderate"
    if v <= 0.75:
        return "High"
    return "Very High"


def layer_risk_labels(df, layer_name, use_balanced=False):
    cfg = LAYER_CONFIG[layer_name]
    risk_col = cfg.get("risk_col")
    score_col = cfg.get("score_col")

    if df.empty:
        return pd.Series(dtype=str)

    if use_balanced and score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        if scores.notna().sum() >= 4:
            q1, q2, q3 = scores.quantile([0.25, 0.50, 0.75]).values
            def q_label(v):
                if pd.isna(v):
                    return "Unknown"
                if v <= q1:
                    return "Low"
                if v <= q2:
                    return "Moderate"
                if v <= q3:
                    return "High"
                return "Very High"
            return scores.apply(q_label)

    if risk_col and risk_col in df.columns:
        return df[risk_col].astype(str)

    if score_col and score_col in df.columns:
        return pd.to_numeric(df[score_col], errors="coerce").apply(label_from_score)

    return pd.Series(["Unknown"] * len(df), index=df.index)


def dynamic_layer_stats(df, layer_name, selected_district, use_balanced=False):
    if df.empty:
        return {"high": "—", "very_high": "—", "top": "—", "selected": "—"}

    cfg = LAYER_CONFIG[layer_name]
    score_col = cfg.get("score_col")
    district_col = get_col(df, ["District_Name", "DISTRICT", "District", "DIST_NAME"])

    work = df.copy()
    work["_layer_risk"] = layer_risk_labels(work, layer_name, use_balanced=use_balanced)

    high_count = int(work["_layer_risk"].str.lower().eq("high").sum())
    very_high_count = int(work["_layer_risk"].str.lower().eq("very high").sum())

    if high_count == 0 and very_high_count == 0:
        order = ["Very High", "High", "Moderate", "Low"]
        highest = "Unknown"
        for p in order:
            if work["_layer_risk"].str.lower().eq(p.lower()).any():
                highest = p
                break
        high_text = f"0 ({highest} highest)"
    else:
        high_text = str(high_count)

    top = "—"
    if score_col and score_col in work.columns and district_col:
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        ranked = work.sort_values(score_col, ascending=False)
        if len(ranked):
            top = str(ranked.iloc[0][district_col])

    selected = "All Punjab"
    if selected_district != "All Punjab" and district_col:
        row = work[work[district_col].astype(str) == str(selected_district)]
        if len(row):
            selected = str(row.iloc[0]["_layer_risk"])

    return {"high": high_text, "very_high": very_high_count, "top": top, "selected": selected}


# =====================================================
# Raster to Cached PNG Overlay
# =====================================================

def class_to_rgba(arr, legend, alpha=175):
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    for val, (_, color) in legend.items():
        h = color.lstrip("#")
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
        rgba[arr == int(val)] = (r, g, b, alpha)
    rgba[~np.isin(arr, [1, 2, 3, 4])] = (0, 0, 0, 0)
    return rgba


def crop_to_valid(data, transform):
    valid = np.isfinite(data)
    if not np.any(valid):
        return data, transform
    rows, cols = np.where(valid)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    cropped = data[r0:r1 + 1, c0:c1 + 1]
    new_transform = transform * rasterio.Affine.translation(c0, r0)
    return cropped, new_transform


def choose_raster_path(layer_name):
    return RASTER_DIR / LAYER_CONFIG[layer_name]["class_file"]


def overlay_cache_paths(layer_name, selected_district):
    safe_layer = layer_name.replace(" ", "_").replace("/", "_")
    safe_district = selected_district.replace(" ", "_").replace("/", "_")
    base = CACHE_DIR / f"{safe_layer}__{safe_district}"
    return base.with_suffix(".png"), base.with_suffix(".json")


def create_overlay_png_once(path, layer_name, shapes_json, selected_district, max_size=520):
    """
    Creates a small PNG overlay once and stores PNG + bounds JSON on disk.
    Later reruns use these cached files directly.
    """
    png_path, json_path = overlay_cache_paths(layer_name, selected_district)

    if png_path.exists() and json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return png_path, meta["bounds"]

    p = Path(path)
    if not file_available(p):
        return None, None

    with rasterio.open(p) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        src_crs = src.crs
        nodata = src.nodata

        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        if src_crs and src_crs.to_string() != "EPSG:4326":
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            dst = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=data,
                destination=dst,
                src_transform=transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
            )
            data = dst
            transform = dst_transform

        if shapes_json:
            from rasterio.features import geometry_mask
            mask_arr = geometry_mask(shapes_json, out_shape=data.shape, transform=transform, invert=True)
            data = np.where(mask_arr, data, np.nan)

        data, transform = crop_to_valid(data, transform)

        h, w = data.shape
        factor = max(1, int(max(h, w) / max_size))
        if factor > 1:
            data = data[::factor, ::factor]
            transform = transform * rasterio.Affine.scale(factor, factor)

        arr = np.rint(data).astype(np.int16)
        arr[~np.isfinite(data)] = 0

        rgba = class_to_rgba(arr, LAYER_CONFIG[layer_name]["legend"], alpha=170)

        buf = io.BytesIO()
        plt.imsave(buf, rgba)
        png_path.write_bytes(buf.getvalue())

        west, south, east, north = rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)
        bounds = [[south, west], [north, east]]

        json_path.write_text(json.dumps({"bounds": bounds}), encoding="utf-8")

        return png_path, bounds


def png_to_data_url(png_path):
    encoded = base64.b64encode(Path(png_path).read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_map(selected_layer, selected_district, opacity, boundary_gj):
    shapes, selected_feature = selected_or_province_shapes(boundary_gj, selected_district)
    raster_path = choose_raster_path(selected_layer)

    m = folium.Map(
        location=[31.2, 72.9],
        zoom_start=7,
        tiles="CartoDB positron",
        control_scale=True
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)

    try:
        png_path, bounds = create_overlay_png_once(str(raster_path), selected_layer, shapes, selected_district)
        if png_path and bounds:
            folium.raster_layers.ImageOverlay(
                image=png_to_data_url(png_path),
                bounds=bounds,
                opacity=opacity,
                name=selected_layer,
                interactive=False,
                zindex=2,
            ).add_to(m)
    except Exception as e:
        folium.Marker([31.2, 72.9], tooltip="Raster display error", popup=str(e)).add_to(m)

    if boundary_gj and boundary_gj.get("features"):
        prop = find_district_property(boundary_gj)
        tooltip = folium.GeoJsonTooltip(fields=[prop], aliases=["District"], sticky=True) if prop else None

        if selected_district == "All Punjab":
            folium.GeoJson(
                boundary_gj,
                name="Punjab districts",
                style_function=lambda feature: {
                    "fillColor": "transparent",
                    "color": "#0f8f6a",
                    "weight": 0.9,
                    "fillOpacity": 0.0,
                    "opacity": 0.65,
                },
                tooltip=tooltip
            ).add_to(m)

    if selected_feature:
        single = geojson_for_single_feature(selected_feature)
        prop = find_district_property(single)
        tooltip = folium.GeoJsonTooltip(fields=[prop], aliases=["District"], sticky=True) if prop else None
        folium.GeoJson(
            single,
            name=f"Selected district: {selected_district}",
            style_function=lambda feature: {
                "fillColor": "transparent",
                "color": "#16a34a",
                "weight": 2.4,
                "fillOpacity": 0.0,
                "opacity": 1.0,
            },
            tooltip=tooltip
        ).add_to(m)
        bounds = selected_feature_bounds(selected_feature)
        if bounds:
            m.fit_bounds(bounds)

    folium.LayerControl(collapsed=True).add_to(m)
    return m, raster_path.name


def render_legend(layer_name):
    cfg = LAYER_CONFIG[layer_name]
    rows = ""
    for _, (label, color) in cfg["legend"].items():
        rows += f"""
        <div class="legend-row">
            <div class="legend-swatch" style="background:{color};"></div>
            <span>{label}</span>
        </div>
        """
    st.markdown(f"""<div class="legend-box"><b>{cfg["legend_title"]}</b>{rows}</div>""", unsafe_allow_html=True)


# =====================================================
# Load Data
# =====================================================

scorecard = load_scorecard(str(SCORECARD_FILE))
boundary_gj, boundary_source = load_boundary_geojson(str(DISTRICTS_GEOJSON_FILE), str(DISTRICTS_SHP_FILE))
rasters_ready = ensure_rasters_available()

district_col = get_col(scorecard, ["District_Name", "DISTRICT", "District", "DIST_NAME"]) if not scorecard.empty else None
district_names = ["All Punjab"]

if district_col:
    district_names += sorted(scorecard[district_col].dropna().astype(str).unique().tolist())
elif boundary_gj:
    prop = find_district_property(boundary_gj)
    if prop:
        district_names += sorted([
            str(f.get("properties", {}).get(prop))
            for f in boundary_gj.get("features", [])
            if f.get("properties", {}).get(prop) is not None
        ])


# =====================================================
# Header
# =====================================================

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
        <div>
            <h2 style="margin:0; color:#166534; font-size:2rem; font-weight:850;">Agro Climate Analysis</h2>
            <div style="color:#6b7280; font-size:0.95rem;">
                Satellite-based agro-climate stress monitoring and district-level risk analysis
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Layout
# =====================================================

left_col, map_col, right_col = st.columns([1.05, 3.15, 1.1], gap="medium")


with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">1</span> Area of Interest</div>', unsafe_allow_html=True)
    selected_district = st.selectbox("Select district / AOI", district_names, index=0, label_visibility="collapsed")
    st.markdown('<div class="small-help">Choose All Punjab or select one district for detailed analysis.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">2</span> Satellite Indicators</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
          <div class="module-card"><div class="module-title">CHIRPS</div><div class="module-sub">Rainfall</div></div>
          <div class="module-card"><div class="module-title">Sentinel-2</div><div class="module-sub">NDVI</div></div>
          <div class="module-card"><div class="module-title">MODIS</div><div class="module-sub">LST</div></div>
          <div class="module-card"><div class="module-title">GLDAS</div><div class="module-sub">Soil moisture</div></div>
          <div class="module-card"><div class="module-title">GRACE-FO</div><div class="module-sub">Water storage</div></div>
          <div class="module-card"><div class="module-title">GNSS</div><div class="module-sub">GPS / Baidu</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">3</span> Analysis Module</div>', unsafe_allow_html=True)
    selected_layer = st.selectbox("Select analysis layer", list(LAYER_CONFIG.keys()), label_visibility="collapsed")
    opacity = st.slider("Layer opacity", 0.10, 1.00, 0.62, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)


layer_stats = dynamic_layer_stats(scorecard, selected_layer, selected_district, use_balanced=False)


with map_col:
    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="map-header">
            <span class="pill">AOI: {selected_district}</span>
            <span class="pill">Layer: {selected_layer}</span>
            <span class="pill">Baseline: 2018–2024</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if rasters_ready:
        m, raster_name = render_map(selected_layer, selected_district, opacity, boundary_gj)
        st_folium(m, width=None, height=540, returned_objects=[])
        st.caption(f"Raster displayed: {raster_name}. Cached PNG overlay is used after first view.")
    else:
        st.warning("Raster layers are not available. Please check Google Drive sharing and filenames.")
    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Active AOI</div>', unsafe_allow_html=True)
    if selected_district == "All Punjab":
        st.markdown('<div class="alert-box"><b>All Punjab selected.</b><br>Select a district to inspect local risk and download a district-wise profile.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box"><b>{selected_district}</b><br>District-specific raster and statistics are active.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Result & Legend</div>', unsafe_allow_html=True)
    render_legend(selected_layer)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">High districts</div><div class="value">{layer_stats["high"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">Very high districts</div><div class="value">{layer_stats["very_high"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">Top district</div><div class="value">{layer_stats["top"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">Selected risk</div><div class="value">{layer_stats["selected"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Export & Reports</div>', unsafe_allow_html=True)
    if not scorecard.empty and district_col and selected_district != "All Punjab":
        row = scorecard[scorecard[district_col].astype(str) == str(selected_district)]
        st.download_button(
            "Download District Profile",
            data=row.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_district.replace(' ', '_')}_Risk_Profile.csv",
            mime="text/csv",
            use_container_width=True
        )
    elif file_available(SCORECARD_FILE):
        with open(SCORECARD_FILE, "rb") as f:
            st.download_button("Download Scorecard", f, file_name=SCORECARD_FILE.name, mime="text/csv", use_container_width=True)

    raster_path = RASTER_DIR / LAYER_CONFIG[selected_layer]["class_file"]
    if file_available(raster_path):
        with open(raster_path, "rb") as f:
            st.download_button("Download Result Raster", f, file_name=raster_path.name, mime="image/tiff", use_container_width=True)

    if boundary_gj:
        st.download_button(
            "Download AOI GeoJSON",
            data=json.dumps(boundary_gj).encode("utf-8"),
            file_name="Punjab_Districts.geojson",
            mime="application/geo+json",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
tab1, tab2, tab3 = st.tabs(["District Scorecard", "District Profile", "Methodology"])

with tab1:
    st.subheader(f"District Scorecard — {selected_layer}")
    if scorecard.empty:
        st.warning("Scorecard CSV not found.")
    else:
        cfg = LAYER_CONFIG[selected_layer]
        score_col = cfg.get("score_col")
        risk_col = cfg.get("risk_col")
        work = scorecard.copy()
        if selected_district != "All Punjab" and district_col:
            work = work[work[district_col].astype(str) == str(selected_district)]
        work["Selected_Layer_Risk"] = layer_risk_labels(work, selected_layer, use_balanced=False)
        display_cols = [c for c in ["District_Name", "DISTRICT", "Selected_Layer_Risk", score_col, risk_col, "Overall_Risk", "Priority_Level", "Main_Concern", "Recommended_Action"] if c and c in work.columns]
        display_cols = list(dict.fromkeys(display_cols))
        view = work[display_cols] if display_cols else work
        if score_col and score_col in view.columns:
            view = view.copy()
            view[score_col] = pd.to_numeric(view[score_col], errors="coerce")
            view = view.sort_values(score_col, ascending=False)
        st.dataframe(view, use_container_width=True, height=340)

with tab2:
    st.subheader("District Profile")
    if scorecard.empty or not district_col:
        st.warning("District profile unavailable.")
    else:
        district_for_profile = selected_district
        if district_for_profile == "All Punjab":
            district_for_profile = st.selectbox("Select district for profile", sorted(scorecard[district_col].dropna().astype(str).unique().tolist()))
        row = scorecard[scorecard[district_col].astype(str) == str(district_for_profile)].iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Overall Risk", row.get("Overall_Risk", "—"))
        with c2:
            st.metric("Main Concern", row.get("Main_Concern", "—"))
        with c3:
            st.metric("Priority", row.get("Priority_Level", "—"))
        st.success(row.get("Recommended_Action", "Routine monitoring recommended."))

with tab3:
    st.subheader("Methodology")
    st.markdown(
        """
        This dashboard uses backend satellite indicators from 2018–2024 to prepare public-facing agro-climate risk layers for Punjab.

        Rasters are downloaded from Google Drive only when missing. Each raster is converted once into a lightweight cached PNG overlay.
        This prevents repeated heavy GeoTIFF processing during Streamlit reruns.

        **Indicators used:** CHIRPS rainfall, Sentinel-2 NDVI, MODIS LST, GLDAS soil moisture,
        GRACE/GRACE-FO total water storage anomaly, and GNSS/GPS/Baidu as an optional field-observation/navigation indicator.

        **Important note:** Water Storage Stress is not exact groundwater depth. It represents broad regional water storage pressure.
        """
    )
