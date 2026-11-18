import ctypes
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Harris & λ- Corner Detector", layout="wide")


@st.cache_resource
def load_engine():
    base_dir = Path(__file__).resolve().parent
    system = platform.system()
    lib_name = (
        "engine.dll"
        if system == "Windows"
        else ("libengine.dylib" if system == "Darwin" else "libengine.so")
    )
    lib_path = base_dir / lib_name

    if not lib_path.exists():
        return None, str(lib_path)

    lib = ctypes.CDLL(str(lib_path))
    lib.detect_corners.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # img_data
        ctypes.c_int,                     # height
        ctypes.c_int,                     # width
        ctypes.c_int,                     # channels
        ctypes.c_int,                     # method (0=Harris, 1=lambda-)
        ctypes.c_int,                     # gauss_ksize
        ctypes.c_double,                  # sigma
        ctypes.c_double,                  # k
        ctypes.c_double,                  # threshold_ratio
        ctypes.c_int,                     # nms_window
        ctypes.c_int,                     # max_corners
        ctypes.POINTER(ctypes.c_int),     # out_x
        ctypes.POINTER(ctypes.c_int),     # out_y
        ctypes.POINTER(ctypes.c_double),  # out_scores
        ctypes.POINTER(ctypes.c_double),  # out_response
        ctypes.POINTER(ctypes.c_double),  # out_time_ms
    ]
    lib.detect_corners.restype = ctypes.c_int
    return lib, str(lib_path)


def run_detector(
    lib,
    img: np.ndarray,
    method: int,
    gauss_ksize: int,
    sigma: float,
    k: float,
    threshold_ratio: float,
    nms_window: int,
    max_corners: int = 50000,
):
    img_c = np.ascontiguousarray(img, dtype=np.uint8)
    if img_c.ndim == 2:
        h, w = img_c.shape
        channels = 1
    else:
        h, w, channels = img_c.shape

    out_x = np.empty(max_corners, dtype=np.int32)
    out_y = np.empty(max_corners, dtype=np.int32)
    out_scores = np.empty(max_corners, dtype=np.float64)
    out_response = np.empty((h, w), dtype=np.float64)
    time_ms = ctypes.c_double(0.0)

    count = lib.detect_corners(
        img_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        h,
        w,
        channels,
        method,
        gauss_ksize,
        sigma,
        k,
        threshold_ratio,
        nms_window,
        max_corners,
        out_x.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_response.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(time_ms),
    )

    return out_x[:count], out_y[:count], out_scores[:count], out_response, time_ms.value


def decode_uploaded_image(uploaded_file, force_grayscale: bool):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    # Normalize alpha channels or BGR -> RGB while preserving true 2D grayscale
    if raw.ndim == 2:
        img = raw
    elif raw.shape[2] == 4:
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    if force_grayscale and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def draw_corners(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: int):
    canvas = (
        cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    )
    for x, y in zip(xs, ys):
        cv2.circle(canvas, (int(x), int(y)), radius, (255, 40, 40), -1)
    return canvas


def build_response_heatmap(response: np.ndarray):
    pos = np.maximum(response, 0.0)
    max_val = pos.max()
    if max_val <= 1e-12:
        return np.zeros((*response.shape, 3), dtype=np.uint8)
    # Log-scale positive responses so both strong and moderate corners are visible
    norm = np.log1p((pos / max_val) * 1000.0) / np.log1p(1000.0)
    gray_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    bgr_map = cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr_map, cv2.COLOR_BGR2RGB)


# --- UI Layout ---
st.title("Feature Extraction: Harris & Shi-Tomasi (λ⁻)")

engine, expected_path = load_engine()
if engine is None:
    st.error(
        f"Shared library not found at `{expected_path}`.\n\n"
        "Compile `engine.cpp` first:\n"
        "```bash\n"
        "g++ -O3 -shared -static -o engine.dll engine.cpp\n"
        "```"
    )
    st.stop()

with st.sidebar:
    st.header("Operator & Parameters")
    operator = st.radio("Corner Operator", ["Harris (R)", "Shi-Tomasi (λ⁻)"])
    method_id = 0 if operator == "Harris (R)" else 1

    gauss_ksize = st.select_slider(
        "Gaussian Window Size", options=[3, 5, 7, 9, 11], value=5
    )
    sigma = st.slider("Gaussian Sigma (σ)", 0.5, 5.0, 1.0, step=0.1)

    if method_id == 0:
        k_val = st.slider("Harris k", 0.01, 0.15, 0.04, step=0.005, format="%.3f")
    else:
        k_val = 0.04

    threshold_ratio = st.slider(
        "Threshold (% of max response)", 0.1, 25.0, 1.0, step=0.1
    ) / 100.0

    nms_window = st.select_slider(
        "NMS Window Size", options=[3, 5, 7, 9, 11, 15], value=5
    )

    st.divider()
    st.header("Input & Display")
    force_gs = st.checkbox(
        "Force Grayscale (override RGB)",
        value=False,
        help="By default, 2D images run Grayscale and 3-channel images run RGB tensor Harris.",
    )
    marker_radius = st.slider("Marker Radius (px)", 1, 8, 3)

uploaded_files = st.file_uploader(
    "Upload one or more images (grayscale or color)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more images to run the C++ corner detector.")
    st.stop()

results = []
for f in uploaded_files:
    img = decode_uploaded_image(f, force_gs)
    if img is None:
        st.warning(f"Could not decode `{f.name}`.")
        continue

    xs, ys, scores, resp, elapsed_ms = run_detector(
        engine, img, method_id, gauss_ksize, sigma, k_val, threshold_ratio, nms_window
    )
    results.append(
        {
            "name": f.name,
            "img": img,
            "xs": xs,
            "ys": ys,
            "scores": scores,
            "response": resp,
            "time_ms": elapsed_ms,
        }
    )

if not results:
    st.stop()

# Batch benchmark table when multiple images are uploaded
if len(results) > 1:
    st.subheader("Computation Time Report")
    summary_rows = [
        {
            "Image": r["name"],
            "Resolution": f"{r['img'].shape[1]} × {r['img'].shape[0]}",
            "Mode": "Grayscale (1 ch)" if r["img"].ndim == 2 else "RGB (3 ch)",
            "Operator": operator,
            "Corners Found": len(r["xs"]),
            "Time (ms)": round(r["time_ms"], 3),
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    selected_name = st.selectbox("Inspect Image", [r["name"] for r in results])
    current = next(r for r in results if r["name"] == selected_name)
else:
    current = results[0]

# Metrics for the selected image
mode_label = "Grayscale (1-ch)" if current["img"].ndim == 2 else "RGB (3-ch)"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Input Mode", mode_label)
c2.metric("Resolution", f"{current['img'].shape[1]} × {current['img'].shape[0]}")
c3.metric("Corners Detected", f"{len(current['xs']):,}")
c4.metric("C++ Execution Time", f"{current['time_ms']:.3f} ms")

# Visual inspection
overlay = draw_corners(current["img"], current["xs"], current["ys"], marker_radius)
heatmap = build_response_heatmap(current["response"])

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Detected Corners")
    st.image(overlay, use_container_width=True)

with col_right:
    st.subheader(f"{operator} Response Map (Log Scale)")
    st.image(heatmap, use_container_width=True)