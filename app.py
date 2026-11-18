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

    # SIFT + matching (only present if engine was rebuilt from the latest engine.cpp)
    if hasattr(lib, "compute_sift_descriptors") and hasattr(lib, "match_descriptors"):
        lib.compute_sift_descriptors.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # img_data
            ctypes.c_int,                     # height
            ctypes.c_int,                     # width
            ctypes.c_int,                     # channels
            ctypes.POINTER(ctypes.c_int),     # kp_x
            ctypes.POINTER(ctypes.c_int),     # kp_y
            ctypes.c_int,                     # n_kp
            ctypes.c_double,                  # sigma
            ctypes.POINTER(ctypes.c_float),   # out_desc (n_kp * 128)
            ctypes.POINTER(ctypes.c_double),  # out_orient
            ctypes.POINTER(ctypes.c_double),  # out_time_ms
        ]
        lib.compute_sift_descriptors.restype = ctypes.c_int
        lib.match_descriptors.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # desc1
            ctypes.c_int,                     # n1
            ctypes.POINTER(ctypes.c_float),   # desc2
            ctypes.c_int,                     # n2
            ctypes.c_int,                     # method (0=SSD, 1=NCC)
            ctypes.c_double,                  # ratio
            ctypes.c_int,                     # cross_check
            ctypes.c_int,                     # max_matches
            ctypes.POINTER(ctypes.c_int),     # out_i
            ctypes.POINTER(ctypes.c_int),     # out_j
            ctypes.POINTER(ctypes.c_double),  # out_score
            ctypes.POINTER(ctypes.c_double),  # out_time_ms
        ]
        lib.match_descriptors.restype = ctypes.c_int
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


def run_sift(lib, img: np.ndarray, xs: np.ndarray, ys: np.ndarray, sigma: float):
    """Compute 128-D SIFT descriptors at the given (Harris) keypoints."""
    img_c = np.ascontiguousarray(img, dtype=np.uint8)
    if img_c.ndim == 2:
        h, w = img_c.shape
        channels = 1
    else:
        h, w, channels = img_c.shape

    n = len(xs)
    if n == 0:
        return np.zeros((0, 128), np.float32), np.zeros(0), 0.0

    kx = np.ascontiguousarray(xs, dtype=np.int32)
    ky = np.ascontiguousarray(ys, dtype=np.int32)
    desc = np.empty((n, 128), dtype=np.float32)
    orient = np.empty(n, dtype=np.float64)
    time_ms = ctypes.c_double(0.0)

    lib.compute_sift_descriptors(
        img_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        h,
        w,
        channels,
        kx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ky.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n,
        sigma,
        desc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        orient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(time_ms),
    )
    return desc, orient, time_ms.value


def run_matcher(lib, desc1, desc2, method: int, ratio: float, cross_check: bool):
    """Match descriptors. method: 0 = SSD, 1 = NCC. Returns (idx1, idx2, score, time_ms)."""
    d1 = np.ascontiguousarray(desc1, dtype=np.float32)
    d2 = np.ascontiguousarray(desc2, dtype=np.float32)
    cap = max(len(d1), 1)
    out_i = np.empty(cap, dtype=np.int32)
    out_j = np.empty(cap, dtype=np.int32)
    out_s = np.empty(cap, dtype=np.float64)
    time_ms = ctypes.c_double(0.0)

    count = lib.match_descriptors(
        d1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(d1),
        d2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(d2),
        method,
        ratio,
        int(cross_check),
        cap,
        out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(time_ms),
    )
    return out_i[:count], out_j[:count], out_s[:count], time_ms.value


def draw_matches(r1, r2, idx1, idx2, max_lines: int = 150):
    """Side-by-side image with lines connecting matched keypoints."""
    def to_rgb(im):
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else im.copy()

    a, b = to_rgb(r1["img"]), to_rgb(r2["img"])
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] :] = b
    off = a.shape[1]

    rng = np.random.default_rng(0)
    sel = np.arange(len(idx1))
    if len(sel) > max_lines:
        sel = rng.choice(sel, max_lines, replace=False)
    for m in sel:
        color = tuple(int(v) for v in rng.integers(60, 255, 3))
        p1 = (int(r1["sift_x"][idx1[m]]), int(r1["sift_y"][idx1[m]]))
        p2 = (int(r2["sift_x"][idx2[m]]) + off, int(r2["sift_y"][idx2[m]]))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)
    return canvas


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

if not hasattr(engine, "compute_sift_descriptors"):
    st.error(
        "The compiled library is outdated (no SIFT / matching functions). Rebuild it:\n"
        "```bash\n"
        "# Windows\n"
        "g++ -O3 -shared -static -o engine.dll engine.cpp\n"
        "# Linux\n"
        "g++ -O3 -shared -fPIC -o libengine.so engine.cpp\n"
        "# macOS\n"
        "g++ -O3 -shared -fPIC -o libengine.dylib engine.cpp\n"
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
    st.header("SIFT Descriptors (on Harris points)")
    sift_sigma = st.slider("Descriptor scale (σ)", 1.0, 4.0, 2.0, step=0.1)
    max_kp = st.slider(
        "Max keypoints per image (strongest first)", 100, 5000, 1500, step=100
    )

    st.header("Matching")
    ratio = st.slider("Ratio test threshold", 0.5, 1.0, 0.8, step=0.05,
                      help="1.0 disables the ratio test.")
    cross_check = st.checkbox("Cross-check (mutual best match)", value=True)

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
    # SIFT descriptors computed on the strongest Harris keypoints (already sorted by score)
    sift_x, sift_y = xs[:max_kp], ys[:max_kp]
    desc, orient, sift_ms = run_sift(engine, img, sift_x, sift_y, sift_sigma)

    results.append(
        {
            "name": f.name,
            "img": img,
            "xs": xs,
            "ys": ys,
            "scores": scores,
            "response": resp,
            "time_ms": elapsed_ms,
            "sift_x": sift_x,
            "sift_y": sift_y,
            "desc": desc,
            "orient": orient,
            "sift_ms": sift_ms,
        }
    )

if not results:
    st.stop()

# Computation time report (Harris + SIFT) for all images
st.subheader("Computation Time Report")
summary_rows = [
    {
        "Image": r["name"],
        "Resolution": f"{r['img'].shape[1]} × {r['img'].shape[0]}",
        "Mode": "Grayscale (1 ch)" if r["img"].ndim == 2 else "RGB (3 ch)",
        "Operator": operator,
        "Corners Found": len(r["xs"]),
        "Harris/λ⁻ Time (ms)": round(r["time_ms"], 3),
        "SIFT Keypoints": len(r["sift_x"]),
        "SIFT Time (ms)": round(r["sift_ms"], 3),
    }
    for r in results
]
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

if len(results) > 1:
    selected_name = st.selectbox("Inspect Image", [r["name"] for r in results])
    current = next(r for r in results if r["name"] == selected_name)
else:
    current = results[0]

# Metrics for the selected image
mode_label = "Grayscale (1-ch)" if current["img"].ndim == 2 else "RGB (3-ch)"
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Input Mode", mode_label)
c2.metric("Resolution", f"{current['img'].shape[1]} × {current['img'].shape[0]}")
c3.metric("Corners Detected", f"{len(current['xs']):,}")
c4.metric("Harris/λ⁻ Time", f"{current['time_ms']:.3f} ms")
c5.metric("SIFT Time", f"{current['sift_ms']:.3f} ms")

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


# ---------------------------------------------------------------------------
# SIFT descriptors on Harris keypoints
# ---------------------------------------------------------------------------
st.divider()
st.header("SIFT Descriptors")
st.caption(
    f"{len(current['sift_x'])} Harris keypoints → 128-D descriptors "
    f"(4×4 cells × 8 orientation bins) in {current['sift_ms']:.3f} ms. "
    "Circles show the descriptor support; the line shows the dominant orientation."
)


def draw_sift_keypoints(r, limit: int = 300):
    canvas = cv2.cvtColor(r["img"], cv2.COLOR_GRAY2RGB) if r["img"].ndim == 2 else r["img"].copy()
    rad = int(round(3 * sift_sigma * 2.5))
    for x, y, th in list(zip(r["sift_x"], r["sift_y"], r["orient"]))[:limit]:
        cv2.circle(canvas, (int(x), int(y)), rad, (60, 220, 60), 1, cv2.LINE_AA)
        end = (int(x + rad * np.cos(th)), int(y + rad * np.sin(th)))
        cv2.line(canvas, (int(x), int(y)), end, (255, 60, 60), 1, cv2.LINE_AA)
    return canvas


st.image(draw_sift_keypoints(current), use_container_width=True)
if len(current["desc"]):
    with st.expander("Descriptor matrix preview (first 5 keypoints)"):
        st.dataframe(
            pd.DataFrame(current["desc"][:5]).round(4), use_container_width=True
        )

# ---------------------------------------------------------------------------
# Feature matching: SSD vs NCC
# ---------------------------------------------------------------------------
st.divider()
st.header("Feature Matching (SSD vs NCC)")

if len(results) < 2:
    st.info("Upload at least two images to match their SIFT features.")
    st.stop()

names = [r["name"] for r in results]
mc1, mc2 = st.columns(2)
name_a = mc1.selectbox("Image A", names, index=0)
name_b = mc2.selectbox("Image B", names, index=1)
img_a = next(r for r in results if r["name"] == name_a)
img_b = next(r for r in results if r["name"] == name_b)

rows = []
match_data = {}
for label, mid in (("SSD", 0), ("NCC", 1)):
    i1, i2, sc, t_ms = run_matcher(engine, img_a["desc"], img_b["desc"], mid, ratio, cross_check)
    match_data[label] = (i1, i2, sc)
    rows.append(
        {
            "Method": label,
            "Descriptors A": len(img_a["desc"]),
            "Descriptors B": len(img_b["desc"]),
            "Matches": len(i1),
            "Matching Time (ms)": round(t_ms, 3),
            "Mean Score": round(float(sc.mean()), 4) if len(sc) else None,
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("SSD: lower score = better match.  NCC: score in [-1, 1], higher = better match.")

tab_ssd, tab_ncc = st.tabs(["SSD matches", "NCC matches"])
for tab, label in ((tab_ssd, "SSD"), (tab_ncc, "NCC")):
    with tab:
        i1, i2, _ = match_data[label]
        if len(i1) == 0:
            st.warning("No matches found. Try a higher ratio threshold or disable cross-check.")
        else:
            st.image(draw_matches(img_a, img_b, i1, i2), use_container_width=True)

if len(results) > 2 and st.checkbox("Run matching report for all image pairs"):
    pair_rows = []
    for a_i in range(len(results)):
        for b_i in range(a_i + 1, len(results)):
            ra, rb = results[a_i], results[b_i]
            row = {"Pair": f"{ra['name']}  ↔  {rb['name']}"}
            for label, mid in (("SSD", 0), ("NCC", 1)):
                i1, _, _, t_ms = run_matcher(engine, ra["desc"], rb["desc"], mid, ratio, cross_check)
                row[f"{label} Matches"] = len(i1)
                row[f"{label} Time (ms)"] = round(t_ms, 3)
            pair_rows.append(row)
    st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)
