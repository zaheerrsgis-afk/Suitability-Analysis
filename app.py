# app.py
# Suitability Analysis Dashboard
#
# Updated:
# - Data folder settings section removed from dashboard.
# - Google Drive raster folder is fixed inside code.
# - Rasters auto-download silently if local data/rasters is empty.
# - Default view: Bahawalnagar + Wheat Suitability Index.
# - All Punjab: original raster, no clipping.
# - District selected: raster clipped to selected district.
# - Raster display is cached by Streamlit to improve district switching.
# - Legend appears only in right-side Result and Legend panel.

import re
import json
import base64
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MeasureControl, Fullscreen
from streamlit_folium import st_folium

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds, transform_geom
from rasterio.enums import Resampling

import geopandas as gpd
from PIL import Image


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Suitability Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

RASTER_DIR = "data/rasters"
BOUNDARY_DIR = "data/boundaries"

GOOGLE_DRIVE_RASTER_FOLDER_URL = "https://drive.google.com/drive/folders/1RRm2jzG4LOvCC9WkLqTLB7tEonca8fA4?usp=sharing"

DEFAULT_DISTRICT = "Bahawalnagar"
DEFAULT_LAYER = "Wheat Suitability Index"

# Smaller value = faster map rendering.
MAX_RENDER_SIZE = 520


# =========================================================
# Google Drive Download
# =========================================================

def local_rasters_exist(raster_dir: str) -> bool:
    folder = Path(raster_dir)
    if not folder.exists():
        return False
    tif_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    return len(tif_files) > 0


@st.cache_resource(show_spinner=False)
def ensure_rasters_from_google_drive(folder_url: str, output_dir: str):
    """
    Downloads rasters only once per Streamlit server session.
    The Google Drive folder must be shared as:
    Anyone with the link can view.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if local_rasters_exist(output_dir):
        return True, "Local rasters already available."

    try:
        import gdown
    except Exception:
        return False, "gdown is missing. Add gdown>=5.1 to requirements.txt."

    try:
        gdown.download_folder(
            url=folder_url,
            output=output_dir,
            quiet=True,
            use_cookies=False
        )
        return True, "Rasters downloaded from Google Drive."
    except Exception as e:
        return False, f"Google Drive download failed: {e}"


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f8f5;
    }

    .main .block-container {
        padding-top: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }

    .hero-card {
        background: linear-gradient(90deg, #e6f5ec 0%, #dceede 100%);
        border: 1px solid #cfe8d8;
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 20px rgba(16, 77, 48, 0.06);
    }

    .hero-title {
        color: #006b38;
        font-size: 30px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #356654;
        font-size: 14px;
        margin-top: 12px;
        line-height: 1.5;
    }

    .panel-card {
        background: #ffffff;
        border: 1px solid #e1ebe5;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 18px rgba(30, 60, 40, 0.06);
    }

    .panel-title {
        color: #087343;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #6f7e75;
        font-size: 12px;
        line-height: 1.5;
    }

    .module-active {
        background: #edfff4;
        border: 1px solid #35cf77;
        color: #006b38;
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .module-disabled {
        background: #edf0ee;
        border: 1px solid #d5ddd8;
        color: #9aa6a0;
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 8px;
        cursor: not-allowed;
        opacity: 0.80;
    }

    .result-pill {
        display: inline-block;
        background: #edfff4;
        border: 1px solid #bdeccf;
        color: #007a3d;
        padding: 7px 11px;
        border-radius: 9px;
        font-size: 12px;
        font-weight: 700;
        margin: 4px 4px 4px 0;
    }

    .legend-row {
        display:flex;
        align-items:center;
        margin-bottom:6px;
    }

    .legend-color {
        width:18px;
        height:12px;
        border-radius:3px;
        margin-right:8px;
        border:1px solid rgba(0,0,0,0.08);
    }

    .legend-text {
        font-size:12px;
        color:#264335;
    }

    .gradient-legend {
        width:100%;
        height:14px;
        border-radius:8px;
        border:1px solid rgba(0,0,0,0.12);
        margin-top:8px;
    }

    .gradient-labels {
        display:flex;
        justify-content:space-between;
        font-size:12px;
        color:#264335;
        margin-top:6px;
    }

    .leaflet-container {
        border-radius: 16px !important;
        border: 1px solid #36d27d !important;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stCheckbox"] label {
        color: #255340 !important;
        font-weight: 700 !important;
        font-size: 12px !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        border-radius: 9px;
        border: 1px solid #c8d9cf;
        background: #ffffff;
        color: #163c2b;
        font-weight: 700;
    }

    .stDownloadButton > button:hover {
        border-color: #23c36b;
        color: #00743a;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Layer Configuration
# =========================================================

LAYER_CONFIG = {
    "suitability": {
        "label": "Wheat Suitability Index",
        "keywords": ["suitability"],
        "type": "Suitability",
        "unit": "index, 0-1",
        "default_min": 0.40,
        "default_max": 0.70,
        "palette": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
    },
    "temperature": {
        "label": "Temperature",
        "keywords": ["temperature", "temp", "lst"],
        "type": "Climate",
        "unit": "degree Celsius",
        "default_min": 15,
        "default_max": 35,
        "palette": ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
    },
    "rainfall": {
        "label": "Rainfall",
        "keywords": ["rainfall", "precip", "chirps"],
        "type": "Climate",
        "unit": "rainfall",
        "default_min": None,
        "default_max": None,
        "palette": ["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"]
    },
    "soc": {
        "label": "Soil Organic Carbon",
        "keywords": ["soc", "organic", "carbon"],
        "type": "Soil",
        "unit": "SOC",
        "default_min": None,
        "default_max": None,
        "palette": ["#fff7bc", "#fec44f", "#d95f0e", "#8c2d04"]
    },
    "ph": {
        "label": "Soil pH",
        "keywords": ["ph"],
        "type": "Soil",
        "unit": "pH",
        "default_min": 5,
        "default_max": 9,
        "palette": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
    },
    "elevation": {
        "label": "Elevation",
        "keywords": ["elevation", "srtm", "dem"],
        "type": "Topography",
        "unit": "m",
        "default_min": None,
        "default_max": None,
        "palette": ["#edf8fb", "#b2e2e2", "#66c2a4", "#238b45", "#005824"]
    },
    "water": {
        "label": "Water Occurrence",
        "keywords": ["water", "occurrence", "hydrology"],
        "type": "Hydrology",
        "unit": "%",
        "default_min": 0,
        "default_max": 100,
        "palette": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    },
    "landcover": {
        "label": "Landcover",
        "keywords": ["landcover", "land_cover", "dynamic", "world", "lulc"],
        "type": "Landcover",
        "unit": "class",
        "default_min": 0,
        "default_max": 8,
        "palette": [
            "#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
            "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"
        ]
    }
}

SUITABILITY_CLASSES = [
    {"class": "Low / Marginal", "range": "< 0.45", "color": "#d7191c"},
    {"class": "Moderate", "range": "0.45-0.55", "color": "#fdae61"},
    {"class": "Suitable", "range": "0.55-0.65", "color": "#ffffbf"},
    {"class": "Highly Suitable", "range": ">= 0.65", "color": "#1a9641"},
]

LANDCOVER_CLASSES = [
    ("Water", "#419bdf"),
    ("Trees", "#397d49"),
    ("Grass", "#88b053"),
    ("Flooded vegetation", "#7a87c6"),
    ("Crops", "#e49635"),
    ("Shrub and scrub", "#dfc35a"),
    ("Built area", "#c4281b"),
    ("Bare ground", "#a59b8f"),
    ("Snow/Ice", "#b39fe1"),
]

SCIENCE_WEIGHTS = pd.DataFrame({
    "Factor": [
        "Temperature",
        "Rainfall",
        "Soil Organic Carbon",
        "Soil pH",
        "Elevation",
        "Water Occurrence",
        "Landcover"
    ],
    "Weight": [0.25, 0.25, 0.15, 0.15, 0.05, 0.05, 0.10],
    "Agronomic meaning": [
        "Controls germination, vegetative growth, grain filling, and heat stress.",
        "Represents seasonal water availability during the wheat cycle.",
        "Reflects soil fertility, structure, and nutrient-holding capacity.",
        "Controls nutrient availability and soil chemical suitability.",
        "Represents topographic suitability and broad agro-ecological limits.",
        "Represents hydrological support and proximity to recurring surface water.",
        "Separates agricultural/natural land from built-up, bare, and water classes."
    ]
})


# =========================================================
# Helper Functions
# =========================================================

def normalize_name(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r"-\d{10,}-\d{10,}", "", name)
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def detect_layer_key(filename: str) -> str:
    lower = filename.lower()
    for key, cfg in LAYER_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in lower:
                return key
    return "unknown"


def list_rasters(raster_dir: str) -> list:
    folder = Path(raster_dir)
    if not folder.exists():
        return []
    return sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))


def find_boundary_file(boundary_dir: str):
    folder = Path(boundary_dir)
    if not folder.exists():
        return None

    for pattern in ["*.geojson", "*.json", "*.gpkg", "*.shp"]:
        files = list(folder.glob(pattern))
        if files:
            return files[0]

    return None


@st.cache_data(show_spinner=False)
def read_boundary(boundary_path: str):
    if boundary_path is None:
        return None

    gdf = gpd.read_file(boundary_path)

    if gdf.empty:
        return None

    gdf = gdf.to_crs(epsg=4326)

    possible_name_cols = [
        "DISTRICT", "District", "district", "DIST_NAME", "DISTRICT_N",
        "NAME", "Name", "name", "ADM2_EN", "ADM2_NAME", "Tehsil", "tehsil"
    ]

    name_col = None
    for col in possible_name_cols:
        if col in gdf.columns:
            name_col = col
            break

    if name_col is None:
        gdf["district_name"] = [f"Area {i + 1}" for i in range(len(gdf))]
    else:
        gdf["district_name"] = gdf[name_col].astype(str).str.strip()

    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)

    return gdf


@st.cache_data(show_spinner=False)
def raster_metadata(path: str):
    with rasterio.open(path) as src:
        bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        return {
            "name": normalize_name(Path(path).name),
            "file": Path(path).name,
            "path": path,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "bounds_4326": bounds_4326,
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
        }


def get_selected_geometry(boundary_gdf, district_name):
    if boundary_gdf is None or district_name == "All Punjab":
        return None

    selected = boundary_gdf[boundary_gdf["district_name"] == district_name]

    if selected.empty:
        return None

    return selected.iloc[0].geometry


def geometry_to_geojson_str(geometry):
    if geometry is None:
        return None

    return json.dumps(
        gpd.GeoSeries([geometry], crs="EPSG:4326").__geo_interface__["features"][0]["geometry"]
    )


def safe_stats(arr: np.ndarray):
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {
            "min": np.nan,
            "mean": np.nan,
            "max": np.nan,
            "std": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "count": 0
        }

    return {
        "min": float(np.nanmin(finite)),
        "mean": float(np.nanmean(finite)),
        "max": float(np.nanmax(finite)),
        "std": float(np.nanstd(finite)),
        "p25": float(np.nanpercentile(finite, 25)),
        "p50": float(np.nanpercentile(finite, 50)),
        "p75": float(np.nanpercentile(finite, 75)),
        "count": int(finite.size)
    }


def get_display_range(arr: np.ndarray, key: str):
    cfg = LAYER_CONFIG.get(key, {})
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return 0.0, 1.0

    default_min = cfg.get("default_min")
    default_max = cfg.get("default_max")

    if default_min is not None and default_max is not None:
        return float(default_min), float(default_max)

    sample = finite
    if sample.size > 100000:
        sample = np.random.default_rng(42).choice(sample, 100000, replace=False)

    p2, p98 = np.nanpercentile(sample, [2, 98])

    if p2 == p98:
        p2 = np.nanmin(sample)
        p98 = np.nanmax(sample)

    return float(p2), float(p98)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def vectorized_linear_rgba(arr, palette, vmin, vmax, opacity):
    arr = arr.astype("float32")
    valid = np.isfinite(arr)

    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if not np.any(valid):
        return rgba

    colors = np.array([hex_to_rgb(c) for c in palette], dtype=np.float32)
    n = len(colors)

    clipped = np.clip(arr, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin + 1e-9)
    scaled = norm * (n - 1)

    idx0 = np.floor(scaled).astype(np.int16)
    idx1 = np.clip(idx0 + 1, 0, n - 1)
    idx0 = np.clip(idx0, 0, n - 1)

    frac = (scaled - idx0).astype(np.float32)
    rgb = colors[idx0] * (1 - frac[..., None]) + colors[idx1] * frac[..., None]

    rgba[..., 0:3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, int(255 * opacity), 0).astype(np.uint8)

    return rgba


def build_rgba(arr: np.ndarray, key: str, vmin: float, vmax: float, opacity: float = 0.72):
    cfg = LAYER_CONFIG.get(key, LAYER_CONFIG["suitability"])
    palette = cfg["palette"]

    if key == "landcover":
        valid = np.isfinite(arr)
        class_arr = np.rint(arr).astype("int16")
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)

        for i, color in enumerate(palette):
            rgb = hex_to_rgb(color).astype(np.uint8)
            mask_i = valid & (class_arr == i)
            rgba[mask_i, 0] = rgb[0]
            rgba[mask_i, 1] = rgb[1]
            rgba[mask_i, 2] = rgb[2]
            rgba[mask_i, 3] = int(255 * opacity)

        return rgba

    return vectorized_linear_rgba(arr, palette, vmin, vmax, opacity)


def rgba_to_data_url(rgba: np.ndarray):
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = tempfile.SpooledTemporaryFile()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def classify_suitability(arr: np.ndarray):
    out = np.full(arr.shape, np.nan, dtype="float32")
    valid = np.isfinite(arr)

    out[(arr < 0.45) & valid] = 1
    out[(arr >= 0.45) & (arr < 0.55) & valid] = 2
    out[(arr >= 0.55) & (arr < 0.65) & valid] = 3
    out[(arr >= 0.65) & valid] = 4

    return out


def render_right_side_legend(layer_key, layer_label, vmin, vmax):
    cfg = LAYER_CONFIG.get(layer_key, LAYER_CONFIG["suitability"])
    palette = cfg["palette"]

    if layer_key == "suitability":
        st.markdown("##### Suitability classes")
        for c in SUITABILITY_CLASSES:
            st.markdown(
                f"""
                <div class="legend-row">
                    <div class="legend-color" style="background:{c['color']};"></div>
                    <div class="legend-text"><b>{c['class']}</b> - {c['range']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif layer_key == "landcover":
        st.markdown("##### Landcover classes")
        for name, color in LANDCOVER_CLASSES:
            st.markdown(
                f"""
                <div class="legend-row">
                    <div class="legend-color" style="background:{color};"></div>
                    <div class="legend-text">{name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.markdown(f"##### {layer_label}")

        gradient = f"linear-gradient(to right, {', '.join(palette)})"

        st.markdown(
            f"""
            <div class="gradient-legend" style="background:{gradient};"></div>
            <div class="gradient-labels">
                <span>{vmin:.2f}</span>
                <span>{vmax:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


@st.cache_data(show_spinner=False, max_entries=80)
def read_raster_for_display(
    path: str,
    layer_key: str,
    selected_geojson_str: str = None,
    max_size: int = MAX_RENDER_SIZE
):
    """
    Cached raster renderer:
    - All Punjab: no clipping, downsampled display raster.
    - District: clipped display raster.
    After a district/layer is opened once, switching back to it is faster.
    """

    with rasterio.open(path) as src:
        nodata = src.nodata

        if selected_geojson_str:
            geom_4326 = json.loads(selected_geojson_str)

            try:
                geom_for_mask = transform_geom(
                    "EPSG:4326",
                    src.crs,
                    geom_4326,
                    precision=6
                )
            except Exception:
                geom_for_mask = geom_4326

            data, out_transform = mask(
                src,
                [geom_for_mask],
                crop=True,
                filled=True,
                nodata=nodata if nodata is not None else -9999
            )

            arr = data[0].astype("float32")

            if nodata is not None:
                arr[arr == nodata] = np.nan
            else:
                arr[arr == -9999] = np.nan

            bounds = rasterio.transform.array_bounds(
                arr.shape[0],
                arr.shape[1],
                out_transform
            )

            crs = src.crs

            if max(arr.shape) > max_size:
                factor = max(arr.shape[0] / max_size, arr.shape[1] / max_size)
                new_h = max(1, int(arr.shape[0] / factor))
                new_w = max(1, int(arr.shape[1] / factor))

                temp = np.nan_to_num(arr, nan=-9999).astype("float32")
                img = Image.fromarray(temp)

                img = img.resize(
                    (new_w, new_h),
                    resample=Image.Resampling.NEAREST
                    if layer_key == "landcover"
                    else Image.Resampling.BILINEAR
                )

                arr = np.array(img).astype("float32")
                arr[arr == -9999] = np.nan

        else:
            scale = max(src.width / max_size, src.height / max_size, 1)
            out_width = max(1, int(src.width / scale))
            out_height = max(1, int(src.height / scale))

            resampling = Resampling.nearest if layer_key == "landcover" else Resampling.bilinear

            arr = src.read(
                1,
                out_shape=(out_height, out_width),
                resampling=resampling
            ).astype("float32")

            display_transform = src.transform * src.transform.scale(
                src.width / out_width,
                src.height / out_height
            )

            bounds = rasterio.transform.array_bounds(
                out_height,
                out_width,
                display_transform
            )

            crs = src.crs

            if nodata is not None:
                arr[arr == nodata] = np.nan

        arr[~np.isfinite(arr)] = np.nan

    bounds_4326 = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)

    return arr, bounds_4326


def build_download_csv(stats_dict, selected_layer, selected_district):
    df = pd.DataFrame([{
        "AOI": selected_district,
        "Layer": selected_layer,
        "Minimum": stats_dict["min"],
        "Mean": stats_dict["mean"],
        "Maximum": stats_dict["max"],
        "StdDev": stats_dict["std"],
        "P25": stats_dict["p25"],
        "Median": stats_dict["p50"],
        "P75": stats_dict["p75"],
        "ValidPixels": stats_dict["count"]
    }])
    return df.to_csv(index=False).encode("utf-8")


def build_map(
    arr,
    bounds_4326,
    layer_key,
    layer_label,
    vmin,
    vmax,
    selected_geometry=None,
    opacity=0.72,
    show_classified=False
):
    south, west, north, east = bounds_4326[1], bounds_4326[0], bounds_4326[3], bounds_4326[2]
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light Map",
        overlay=False,
        control=True
    ).add_to(m)

    if show_classified and layer_key == "suitability":
        display_arr = classify_suitability(arr)
        rgba = np.zeros((display_arr.shape[0], display_arr.shape[1], 4), dtype=np.uint8)

        class_colors = {
            1: "#d7191c",
            2: "#fdae61",
            3: "#ffffbf",
            4: "#1a9641"
        }

        for cls, col in class_colors.items():
            rgb = hex_to_rgb(col).astype(np.uint8)
            msk = np.isfinite(display_arr) & (display_arr == cls)
            rgba[msk, 0] = rgb[0]
            rgba[msk, 1] = rgb[1]
            rgba[msk, 2] = rgb[2]
            rgba[msk, 3] = int(255 * opacity)
    else:
        rgba = build_rgba(arr, layer_key, vmin, vmax, opacity=opacity)

    data_url = rgba_to_data_url(rgba)

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=[[south, west], [north, east]],
        opacity=1,
        name=layer_label,
        interactive=False,
        cross_origin=False,
        zindex=10
    ).add_to(m)

    if selected_geometry is not None:
        selected_geojson = json.loads(
            gpd.GeoSeries([selected_geometry], crs="EPSG:4326").to_json()
        )

        folium.GeoJson(
            selected_geojson,
            name="Selected District White Outline",
            style_function=lambda x: {
                "color": "#ffffff",
                "weight": 5,
                "fillOpacity": 0,
                "opacity": 1
            }
        ).add_to(m)

        folium.GeoJson(
            selected_geojson,
            name="Selected District Outline",
            style_function=lambda x: {
                "color": "#008b4a",
                "weight": 2,
                "fillOpacity": 0,
                "opacity": 1
            }
        ).add_to(m)

    MeasureControl(position="bottomleft").add_to(m)
    Fullscreen(position="topright").add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    try:
        if selected_geometry is not None:
            gxmin, gymin, gxmax, gymax = selected_geometry.bounds
            m.fit_bounds([[gymin, gxmin], [gymax, gxmax]])
        else:
            m.fit_bounds([[south, west], [north, east]])
    except Exception:
        pass

    return m


# =========================================================
# Header
# =========================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Suitability Analysis</div>
        <div class="hero-subtitle">
            Interactive suitability dashboard using wheat suitability rasters, agro-climatic indicators,
            soil factors, landcover, district boundaries, and weighted suitability science.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Silent Data Preparation
# =========================================================

with st.spinner("Preparing dashboard data..."):
    download_ok, download_msg = ensure_rasters_from_google_drive(
        GOOGLE_DRIVE_RASTER_FOLDER_URL,
        RASTER_DIR
    )

if not download_ok:
    st.error(download_msg)
    st.stop()


# =========================================================
# Load Files
# =========================================================

rasters = list_rasters(RASTER_DIR)
boundary_file = find_boundary_file(BOUNDARY_DIR)
boundary_gdf = read_boundary(str(boundary_file)) if boundary_file else None

if not rasters:
    st.error(
        "No GeoTIFF rasters found. Check Google Drive folder sharing and make sure it contains .tif files."
    )
    st.stop()


# =========================================================
# Raster Inventory
# =========================================================

metadata_rows = []
raster_options = {}

for r in rasters:
    key = detect_layer_key(r.name)
    cfg = LAYER_CONFIG.get(
        key,
        {
            "label": normalize_name(r.name),
            "type": "Unknown",
            "unit": "",
            "palette": ["#d7191c", "#ffffbf", "#1a9641"]
        }
    )

    meta = raster_metadata(str(r))
    display_label = cfg["label"] if key != "unknown" else normalize_name(r.name)

    option_label = display_label
    if option_label in raster_options:
        option_label = f"{display_label} - {r.name}"

    raster_options[option_label] = {
        "path": str(r),
        "key": key,
        "meta": meta,
        "label": display_label
    }

    metadata_rows.append({
        "Layer": display_label,
        "File": r.name,
        "Type": cfg.get("type", "Unknown"),
        "CRS": meta["crs"],
        "Size": f"{meta['width']} x {meta['height']}",
        "Bands": meta["count"]
    })

inventory_df = pd.DataFrame(metadata_rows)


# =========================================================
# Main Layout
# =========================================================

left_col, map_col, right_col = st.columns([1.05, 2.7, 0.9], gap="large")


# =========================================================
# Left Panel
# =========================================================

with left_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Area of Interest</div>
            <div class="small-muted">
                Default view opens Bahawalnagar with Wheat Suitability Index.
                All Punjab shows the original raster. Selecting a district clips the selected raster to that district.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if boundary_gdf is not None:
        district_names = ["All Punjab"] + sorted(boundary_gdf["district_name"].dropna().unique().tolist())

        default_district_index = 0
        for i, dname in enumerate(district_names):
            if dname.strip().lower() == DEFAULT_DISTRICT.lower():
                default_district_index = i
                break

        selected_district = st.selectbox(
            "District filter",
            district_names,
            index=default_district_index
        )
    else:
        selected_district = "All Punjab"
        st.warning("No district boundary found. Add GeoJSON or complete shapefile in data/boundaries.")

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Type of Suitability</div>
            <div class="module-active">Crop</div>
            <div class="small-muted" style="margin-bottom:8px;">Crop suitability is active.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.selectbox(
        "Crop",
        ["✓ Wheat"],
        index=0,
        disabled=True
    )

    st.markdown(
        """
        <div class="panel-card">
            <div class="module-disabled">Soil</div>
            <div class="module-disabled">Fertilizer</div>
            <div class="small-muted" style="margin-top:8px;">
                Soil and fertilizer suitability modules are available as future modules and are disabled in this version.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Suitability Layers</div>
            <div class="small-muted">
                Select the final suitability layer or one input factor.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    preferred_order = [
        "Wheat Suitability Index",
        "Temperature",
        "Rainfall",
        "Soil Organic Carbon",
        "Soil pH",
        "Elevation",
        "Water Occurrence",
        "Landcover"
    ]

    available_labels = list(raster_options.keys())

    def order_score(x):
        for i, p in enumerate(preferred_order):
            if x.startswith(p):
                return i
        return 999

    available_labels = sorted(available_labels, key=order_score)

    default_layer_index = 0
    for i, layer_name in enumerate(available_labels):
        if DEFAULT_LAYER.lower() in layer_name.lower():
            default_layer_index = i
            break

    selected_layer_option = st.selectbox(
        "Select layer",
        available_labels,
        index=default_layer_index
    )

    selected_info = raster_options[selected_layer_option]

    selected_path = selected_info["path"]
    selected_key = selected_info["key"]
    selected_label = selected_info["label"]
    selected_cfg = LAYER_CONFIG.get(selected_key, LAYER_CONFIG["suitability"])

    show_classified = False
    if selected_key == "suitability":
        show_classified = st.checkbox("Show classified suitability", value=False)

    opacity = st.slider("Layer opacity", 0.10, 1.00, 0.72, 0.05)


# =========================================================
# Selected Geometry
# =========================================================

selected_geometry = get_selected_geometry(boundary_gdf, selected_district)
selected_geojson_str = geometry_to_geojson_str(selected_geometry)


# =========================================================
# Read Raster
# =========================================================

try:
    with st.spinner("Rendering selected layer..."):
        arr, bounds_4326 = read_raster_for_display(
            selected_path,
            selected_key,
            selected_geojson_str=selected_geojson_str,
            max_size=MAX_RENDER_SIZE
        )
except Exception as e:
    st.error(f"Could not read raster: {e}")
    st.stop()

stats = safe_stats(arr)
auto_vmin, auto_vmax = get_display_range(arr, selected_key)

if selected_key == "suitability":
    vmin, vmax = 0.40, 0.70
elif selected_key == "landcover":
    vmin, vmax = 0, 8
else:
    vmin, vmax = auto_vmin, auto_vmax


# =========================================================
# Map
# =========================================================

with map_col:
    m = build_map(
        arr=arr,
        bounds_4326=bounds_4326,
        layer_key=selected_key,
        layer_label=selected_label,
        vmin=vmin,
        vmax=vmax,
        selected_geometry=selected_geometry,
        opacity=opacity,
        show_classified=show_classified
    )

    st_folium(
        m,
        width=None,
        height=650,
        returned_objects=[]
    )


# =========================================================
# Right Panel
# =========================================================

with right_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Active AOI</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <span class="result-pill">AOI: {selected_district}</span><br>
        <span class="result-pill">Type: Crop</span><br>
        <span class="result-pill">Crop: Wheat</span><br>
        <span class="result-pill">Layer: {selected_label}</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Result and Legend</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="small-muted">
            <b>{selected_label}</b><br>
            Unit: {selected_cfg.get("unit", "")}<br>
            Display range: {vmin:.3f} to {vmax:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    render_right_side_legend(selected_key, selected_label, vmin, vmax)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Export</div>
        """,
        unsafe_allow_html=True
    )

    csv_bytes = build_download_csv(stats, selected_label, selected_district)

    st.download_button(
        "Download Layer Stats CSV",
        data=csv_bytes,
        file_name=f"{selected_district.replace(' ', '_')}_{normalize_name(selected_label)}_stats.csv",
        mime="text/csv"
    )

    with open(selected_path, "rb") as f:
        st.download_button(
            "Download Selected Raster",
            data=f,
            file_name=Path(selected_path).name,
            mime="image/tiff"
        )

    st.caption("For PDF, use browser Print and Save as PDF.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Optional Expanders
# =========================================================

with st.expander("Science and Weights", expanded=False):
    st.subheader("Scientific model used in the dashboard")

    st.write(
        "The dashboard follows a weighted overlay logic for wheat suitability. "
        "Temperature and rainfall are treated as the strongest drivers, followed by soil organic carbon "
        "and soil pH. Elevation, water occurrence, and landcover provide additional physical and land-use constraints."
    )

    st.dataframe(SCIENCE_WEIGHTS, use_container_width=True, hide_index=True)

    st.subheader("Suitability thresholds")
    st.dataframe(pd.DataFrame(SUITABILITY_CLASSES), use_container_width=True, hide_index=True)

    st.warning(
        "Agronomic review note: rainfall should ideally represent seasonal total rainfall for the wheat season, "
        "not only average rainfall. Soil organic carbon and pH should also be checked against the actual unit scale "
        "of the exported rasters."
    )


with st.expander("Raster Diagnostics", expanded=False):
    diagnostic_df = pd.DataFrame([{
        "AOI": selected_district,
        "Layer": selected_label,
        "Minimum": stats["min"],
        "P25": stats["p25"],
        "Median": stats["p50"],
        "Mean": stats["mean"],
        "P75": stats["p75"],
        "Maximum": stats["max"],
        "Std. Dev.": stats["std"],
        "Valid Pixels": stats["count"]
    }])

    st.dataframe(diagnostic_df, use_container_width=True, hide_index=True)

    finite = arr[np.isfinite(arr)]

    if finite.size > 0:
        if finite.size > 100000:
            finite = np.random.default_rng(42).choice(finite, 100000, replace=False)

        hist_values, bin_edges = np.histogram(finite, bins=25)
        hist_df = pd.DataFrame({
            "Value": bin_edges[:-1],
            "Frequency": hist_values
        })
        st.bar_chart(hist_df.set_index("Value"))


with st.expander("Raster Inventory", expanded=False):
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)

    if boundary_file:
        st.success(f"Boundary file loaded: {boundary_file}")
    else:
        st.error("No boundary file found in data/boundaries.")

    st.code(
        """
GitHub project folder:
  app.py
  requirements.txt

  data/
    boundaries/
      Punjab_Districts.geojson

Google Drive:
  Heavy raster .tif files stored in linked Drive folder.

Cache:
  Rendered raster layers are cached after first opening to improve switching speed.
        """,
        language="text"
    )
