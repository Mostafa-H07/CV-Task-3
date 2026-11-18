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


def run_matcher(lib, desc1, desc2, method: int, ratio: float):
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
st.title("Feature Extraction: Harris / Shi-Tomasi → SIFT → Matching")

engine, expected_path = load_engine()
if engine is None or not hasattr(engine, "compute_sift_descriptors"):
    st.error(
        "Shared library missing or outdated. Compile `engine.cpp`:\n"
        "```bash\n"
        "g++ -O3 -shared -static -o engine.dll engine.cpp          # Windows\n"
        "g++ -O3 -shared -fPIC -o libengine.so engine.cpp          # Linux\n"
        "g++ -O3 -shared -fPIC -o libengine.dylib engine.cpp       # macOS\n"
        "```"
    )
    st.stop()

uploaded_files = st.file_uploader(
    "Upload images (grayscale or color)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
force_gs = st.checkbox("Force grayscale (convert color images)", value=False)

if not uploaded_files:
    st.info("Upload one or more images to start.")
    st.stop()

images = []
for f in uploaded_files:
    img = decode_uploaded_image(f, force_gs)
    if img is None:
        st.warning(f"Could not decode `{f.name}`.")
    else:
        images.append({"name": f.name, "img": img})
if not images:
    st.stop()

tab_corners, tab_sift, tab_match = st.tabs(
    ["1. Harris / Shi-Tomasi", "2. SIFT", "3. Matching"]
)

# ---------------------------------------------------------------------------
# Tab 1: corner detection
# ---------------------------------------------------------------------------
with tab_corners:
    operator = st.radio("Corner operator", ["Harris (R)", "Shi-Tomasi (λ⁻)"], horizontal=True)
    method_id = 0 if operator == "Harris (R)" else 1

    p1, p2, p3, p4, p5 = st.columns(5)
    gauss_ksize = p1.select_slider("Gaussian window", options=[3, 5, 7, 9, 11], value=5)
    sigma = p2.slider("Gaussian σ", 0.5, 5.0, 1.0, step=0.1)
    k_val = p3.slider("Harris k", 0.01, 0.15, 0.04, step=0.005, format="%.3f",
                      disabled=(method_id == 1))
    threshold_ratio = p4.slider("Threshold (% of max)", 0.1, 25.0, 1.0, step=0.1) / 100.0
    nms_window = p5.select_slider("NMS window", options=[3, 5, 7, 9, 11, 15], value=5)
    marker_radius = st.slider("Marker radius (px)", 1, 8, 3)

    for r in images:
        r["xs"], r["ys"], r["scores"], r["response"], r["time_ms"] = run_detector(
            engine, r["img"], method_id, gauss_ksize, sigma, k_val, threshold_ratio, nms_window
        )

    st.subheader("Computation time")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Image": r["name"],
                    "Resolution": f"{r['img'].shape[1]} × {r['img'].shape[0]}",
                    "Mode": "Grayscale" if r["img"].ndim == 2 else "RGB",
                    "Operator": operator,
                    "Corners": len(r["xs"]),
                    "Time (ms)": round(r["time_ms"], 3),
                }
                for r in images
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    cur = st.selectbox("Inspect image", [r["name"] for r in images], key="inspect_corners")
    cur = next(r for r in images if r["name"] == cur)
    left, right = st.columns(2)
    with left:
        st.subheader("Detected corners")
        st.image(draw_corners(cur["img"], cur["xs"], cur["ys"], marker_radius),
                 width="stretch")
    with right:
        st.subheader("Response map (log scale)")
        st.image(build_response_heatmap(cur["response"]), width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: SIFT descriptors on the detected keypoints
# ---------------------------------------------------------------------------
with tab_sift:
    st.caption(f"SIFT is computed on the {operator} keypoints from tab 1 (strongest first).")
    s1, s2 = st.columns(2)
    sift_sigma = s1.slider("Descriptor scale (σ)", 1.0, 4.0, 2.0, step=0.1)
    max_kp = s2.slider("Max keypoints per image", 100, 5000, 1500, step=100)

    for r in images:
        r["sift_x"], r["sift_y"] = r["xs"][:max_kp], r["ys"][:max_kp]
        r["desc"], r["orient"], r["sift_ms"] = run_sift(
            engine, r["img"], r["sift_x"], r["sift_y"], sift_sigma
        )

    st.subheader("Computation time")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Image": r["name"],
                    "Keypoints": len(r["sift_x"]),
                    "Descriptor size": "128",
                    "Time (ms)": round(r["sift_ms"], 3),
                }
                for r in images
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    cur = st.selectbox("Inspect image", [r["name"] for r in images], key="inspect_sift")
    cur = next(r for r in images if r["name"] == cur)

    canvas = cv2.cvtColor(cur["img"], cv2.COLOR_GRAY2RGB) if cur["img"].ndim == 2 else cur["img"].copy()
    rad = int(round(3 * sift_sigma * 2.5))
    for x, y, th in list(zip(cur["sift_x"], cur["sift_y"], cur["orient"]))[:300]:
        cv2.circle(canvas, (int(x), int(y)), rad, (60, 220, 60), 1, cv2.LINE_AA)
        cv2.line(canvas, (int(x), int(y)),
                 (int(x + rad * np.cos(th)), int(y + rad * np.sin(th))), (255, 60, 60), 1, cv2.LINE_AA)
    st.caption("Circle = descriptor support region, red line = dominant orientation (first 300 keypoints).")
    st.image(canvas, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: matching (SSD vs NCC)
# ---------------------------------------------------------------------------
with tab_match:
    if len(images) < 2:
        st.info("Upload at least two images to match their features.")
    else:
        names = [r["name"] for r in images]
        m1, m2, m3 = st.columns(3)
        name_a = m1.selectbox("Image A", names, index=0)
        name_b = m2.selectbox("Image B", names, index=1)
        ratio = m3.slider("Ratio test threshold", 0.5, 1.0, 0.8, step=0.05,
                          help="Lower = fewer but more reliable matches. 1.0 disables the test.")
        ra = next(r for r in images if r["name"] == name_a)
        rb = next(r for r in images if r["name"] == name_b)

        out = {}
        for label, mid in (("SSD", 0), ("NCC", 1)):
            out[label] = run_matcher(engine, ra["desc"], rb["desc"], mid, ratio)

        st.subheader("Computation time")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Method": label,
                        "Descriptors A": len(ra["desc"]),
                        "Descriptors B": len(rb["desc"]),
                        "Matches": len(out[label][0]),
                        "Time (ms)": round(out[label][3], 3),
                    }
                    for label in ("SSD", "NCC")
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        for label in ("SSD", "NCC"):
            i1, i2, _, _ = out[label]
            st.subheader(f"{label} matches")
            if len(i1) == 0:
                st.warning("No matches. Try a higher ratio threshold.")
            else:
                st.image(draw_matches(ra, rb, i1, i2), width="stretch")
