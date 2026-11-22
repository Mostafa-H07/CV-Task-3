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


@st.cache_data(show_spinner=False)
def run_detector(
    _lib,
    cache_key,
    _img: np.ndarray,
    method: int,
    gauss_ksize: int,
    sigma: float,
    k: float,
    threshold_ratio: float,
    nms_window: int,
    need_response: bool = True,
    max_corners: int = 50000,
):
    img_c = np.ascontiguousarray(_img, dtype=np.uint8)
    if img_c.ndim == 2:
        h, w = img_c.shape
        channels = 1
    else:
        h, w, channels = img_c.shape

    out_x = np.empty(max_corners, dtype=np.int32)
    out_y = np.empty(max_corners, dtype=np.int32)
    out_scores = np.empty(max_corners, dtype=np.float64)
    out_response = np.empty((h, w), dtype=np.float64) if need_response else None
    time_ms = ctypes.c_double(0.0)
    response_ptr = (
        out_response.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if out_response is not None
        else ctypes.POINTER(ctypes.c_double)()
    )

    count = _lib.detect_corners(
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
        response_ptr,
        ctypes.byref(time_ms),
    )

    return out_x[:count], out_y[:count], out_scores[:count], out_response, time_ms.value


@st.cache_data(show_spinner=False)
def run_sift(
    _lib,
    cache_key,
    _img: np.ndarray,
    _xs: np.ndarray,
    _ys: np.ndarray,
    sigma: float,
):
    """Compute 128-D SIFT descriptors at the given (Harris / Shi-Tomasi) keypoints."""
    img_c = np.ascontiguousarray(_img, dtype=np.uint8)
    if img_c.ndim == 2:
        h, w = img_c.shape
        channels = 1
    else:
        h, w, channels = img_c.shape

    n = len(_xs)
    if n == 0:
        return np.zeros((0, 128), np.float32), np.zeros(0), 0.0

    kx = np.ascontiguousarray(_xs, dtype=np.int32)
    ky = np.ascontiguousarray(_ys, dtype=np.int32)
    desc = np.empty((n, 128), dtype=np.float32)
    orient = np.empty(n, dtype=np.float64)
    time_ms = ctypes.c_double(0.0)

    _lib.compute_sift_descriptors(
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


@st.cache_data(show_spinner=False)
def run_matcher(
    _lib,
    cache_key_a,
    cache_key_b,
    _desc1: np.ndarray,
    _desc2: np.ndarray,
    method: int,
    ratio: float,
):
    """Match descriptors. method: 0 = SSD, 1 = NCC. Returns (idx1, idx2, score, time_ms)."""
    if len(_desc1) == 0 or len(_desc2) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float64), 0.0

    d1 = np.ascontiguousarray(_desc1, dtype=np.float32)
    d2 = np.ascontiguousarray(_desc2, dtype=np.float32)
    cap = max(len(d1), 1)
    out_i = np.empty(cap, dtype=np.int32)
    out_j = np.empty(cap, dtype=np.int32)
    out_s = np.empty(cap, dtype=np.float64)
    time_ms = ctypes.c_double(0.0)

    count = _lib.match_descriptors(
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


@st.cache_data(show_spinner=False)
def decode_image_bytes(raw_bytes: bytes, force_grayscale: bool):
    file_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
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


def draw_matches(
    img1: np.ndarray,
    xs1: np.ndarray,
    ys1: np.ndarray,
    img2: np.ndarray,
    xs2: np.ndarray,
    ys2: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    scores: np.ndarray,
    higher_is_better: bool,
    line_thickness: int = 2,
    point_radius: int = 4,
    max_lines: int = 150,
):
    """Side-by-side image with lines connecting the best-scoring matched keypoints."""
    def to_rgb(im):
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else im.copy()

    a, b = to_rgb(img1), to_rgb(img2)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] :] = b
    off = a.shape[1]

    order = np.argsort(-scores if higher_is_better else scores)[:max_lines]
    if len(order) > 0:
        hues = np.linspace(0, 179, len(order), endpoint=False).astype(np.uint8)
        hsv_palette = np.zeros((len(order), 1, 3), dtype=np.uint8)
        hsv_palette[:, 0, 0] = hues
        hsv_palette[:, 0, 1] = 255
        hsv_palette[:, 0, 2] = 255
        rgb_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2RGB)[:, 0]
    else:
        rgb_palette = np.zeros((0, 3), dtype=np.uint8)
    for line_index, m in enumerate(order):
        color = tuple(int(v) for v in rgb_palette[line_index])
        p1 = (int(xs1[idx1[m]]), int(ys1[idx1[m]]))
        p2 = (int(xs2[idx2[m]]) + off, int(ys2[idx2[m]]))
        cv2.line(canvas, p1, p2, (0, 0, 0), line_thickness + 2, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, line_thickness, cv2.LINE_AA)
        cv2.circle(canvas, p1, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, point_radius, color, -1, cv2.LINE_AA)
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


def init_state():
    defaults = {
        "cfg_operator": "Harris (R)",
        "cfg_gauss_ksize": 5,
        "cfg_sigma": 1.0,
        "cfg_k": 0.04,
        "cfg_threshold_pct": 1.0,
        "cfg_nms_window": 5,
        "cfg_marker_radius": 3,
        "cfg_sift_sigma": 2.0,
        "cfg_max_kp": 1500,
        "cfg_ratio": 0.8,
        "cfg_sift_keypoint_scale": 1.2,
        "cfg_sift_line_thickness": 2,
        "cfg_sift_max_drawn": 300,
        "cfg_match_line_thickness": 2,
        "cfg_match_point_radius": 4,
        "cfg_match_max_drawn": 150,
        "known_image_names": [],
        "sift_log": [],
        "last_sift_compute_sig": None,
        "latest_corner_runs": {},
        "pinned_corner_runs": [],
        "pinned_sift_runs": [],
        "latest_match_runs": {},
        "pinned_match_runs": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def bind_widget(widget_key: str, cfg_key: str):
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[cfg_key]


def sync_widget(widget_key: str, cfg_key: str):
    st.session_state[cfg_key] = st.session_state[widget_key]


# --- Engine Check ---
engine, expected_path = load_engine()
if engine is None or not hasattr(engine, "compute_sift_descriptors"):
    st.title("Feature Extraction: Harris / Shi-Tomasi → SIFT → Matching")
    st.error(
        "Shared library missing or outdated. Compile `engine.cpp`:\n"
        "```bash\n"
        "g++ -O3 -shared -static -o engine.dll engine.cpp          # Windows\n"
        "g++ -O3 -shared -fPIC -o libengine.so engine.cpp          # Linux\n"
        "g++ -O3 -shared -fPIC -o libengine.dylib engine.cpp       # macOS\n"
        "```"
    )
    st.stop()

init_state()

# --- Pinned Sidebar: Inputs & Navigation ---
with st.sidebar:
    with st.expander(
        "Inputs",
        expanded=not bool(st.session_state.get("known_image_names")),
    ):
        uploaded_files = st.file_uploader(
            "Upload images (grayscale or color)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        force_gs = st.checkbox("Force grayscale (convert color images)", value=False)

    st.divider()
    stage = st.radio(
        "Pipeline stage",
        ["1. Harris / Shi-Tomasi", "2. SIFT", "3. Matching"],
    )

st.title("Feature Extraction: Harris / Shi-Tomasi → SIFT → Matching")

if not uploaded_files:
    st.info("Upload one or more images in the sidebar to start.")
    st.stop()

images = {}
img_tokens = {}
for f in uploaded_files:
    img = decode_image_bytes(f.getvalue(), force_gs)
    if img is None:
        st.warning(f"Could not decode `{f.name}`.")
    else:
        images[f.name] = img
        file_id = getattr(f, "file_id", None)
        if file_id is not None:
            img_tokens[f.name] = (file_id, force_gs)
        else:
            img_tokens[f.name] = (f.name, getattr(f, "size", 0), force_gs)

if not images:
    st.stop()

image_names = list(images.keys())

new_names = [name for name in image_names if name not in st.session_state["known_image_names"]]
if new_names:
    newest = new_names[-1]
    st.session_state["stage1_img"] = newest
    st.session_state["stage2_img"] = newest

if "stage1_img" in st.session_state and st.session_state["stage1_img"] not in image_names:
    st.session_state["stage1_img"] = image_names[-1]
if "stage2_img" in st.session_state and st.session_state["stage2_img"] not in image_names:
    st.session_state["stage2_img"] = image_names[-1]

st.session_state["known_image_names"] = list(image_names)


def get_corner_params():
    op = st.session_state["cfg_operator"]
    method_id = 0 if op == "Harris (R)" else 1
    k_effective = 0.0 if method_id == 1 else st.session_state["cfg_k"]
    return (
        op,
        method_id,
        st.session_state["cfg_gauss_ksize"],
        st.session_state["cfg_sigma"],
        k_effective,
        st.session_state["cfg_threshold_pct"] / 100.0,
        st.session_state["cfg_nms_window"],
    )


# ---------------------------------------------------------------------------
# Stage 1: Corner Detection (Harris / Shi-Tomasi)
# ---------------------------------------------------------------------------
if stage == "1. Harris / Shi-Tomasi":
    with st.sidebar:
        st.divider()
        st.header("Corner Parameters")
        selected_name = st.selectbox("Inspect image", image_names, key="stage1_img")
        op_options = ["Harris (R)", "Shi-Tomasi (λ⁻)"]
        bind_widget("w_operator", "cfg_operator")
        st.radio(
            "Corner operator",
            op_options,
            key="w_operator",
            on_change=sync_widget,
            args=("w_operator", "cfg_operator"),
        )
        method_id = 0 if st.session_state["cfg_operator"] == "Harris (R)" else 1

        bind_widget("w_gauss_ksize", "cfg_gauss_ksize")
        st.select_slider(
            "Gaussian window",
            options=[3, 5, 7, 9, 11],
            key="w_gauss_ksize",
            on_change=sync_widget,
            args=("w_gauss_ksize", "cfg_gauss_ksize"),
        )
        bind_widget("w_sigma", "cfg_sigma")
        st.slider(
            "Gaussian σ",
            0.5,
            5.0,
            step=0.1,
            key="w_sigma",
            on_change=sync_widget,
            args=("w_sigma", "cfg_sigma"),
        )
        bind_widget("w_k", "cfg_k")
        st.slider(
            "Harris k",
            0.01,
            0.15,
            step=0.005,
            format="%.3f",
            disabled=(method_id == 1),
            key="w_k",
            on_change=sync_widget,
            args=("w_k", "cfg_k"),
        )
        bind_widget("w_threshold_pct", "cfg_threshold_pct")
        st.slider(
            "Threshold (% of max)",
            0.1,
            25.0,
            step=0.1,
            key="w_threshold_pct",
            on_change=sync_widget,
            args=("w_threshold_pct", "cfg_threshold_pct"),
        )
        bind_widget("w_nms_window", "cfg_nms_window")
        st.select_slider(
            "NMS window",
            options=[3, 5, 7, 9, 11, 15],
            key="w_nms_window",
            on_change=sync_widget,
            args=("w_nms_window", "cfg_nms_window"),
        )
        bind_widget("w_marker_radius", "cfg_marker_radius")
        st.slider(
            "Marker radius (px)",
            1,
            8,
            key="w_marker_radius",
            on_change=sync_widget,
            args=("w_marker_radius", "cfg_marker_radius"),
        )

    cur_img = images[selected_name]
    op, method_id, gauss_ksize, sigma, k_eff, threshold_ratio, nms_window = get_corner_params()

    xs, ys, scores, response, time_ms = run_detector(
        engine,
        img_tokens[selected_name],
        cur_img,
        method_id,
        gauss_ksize,
        sigma,
        k_eff,
        threshold_ratio,
        nms_window,
        need_response=True,
    )

    mode_str = "Grayscale" if cur_img.ndim == 2 else "RGB"
    current_row = {
        "Image": selected_name,
        "Resolution": f"{cur_img.shape[1]} × {cur_img.shape[0]}",
        "Mode": mode_str,
        "Operator": op,
        "ksize": gauss_ksize,
        "σ": round(sigma, 2),
        "k": "—" if method_id == 1 else f"{st.session_state['cfg_k']:.3f}",
        "Thresh %": round(st.session_state["cfg_threshold_pct"], 2),
        "NMS": nms_window,
        "Corners": len(xs),
        "Time (ms)": round(time_ms, 3),
    }

    # Keep latest run per (Image, Operator) so Harris & Shi-Tomasi stay side-by-side
    st.session_state["latest_corner_runs"][(selected_name, op)] = current_row

    left, right = st.columns(2)
    with left:
        st.subheader(f"Detected corners — {op}")
        st.image(
            draw_corners(cur_img, xs, ys, st.session_state["cfg_marker_radius"]),
            width="stretch",
        )
    with right:
        st.subheader("Response map (log scale)")
        st.image(build_response_heatmap(response), width="stretch")

    st.subheader("Live comparison (latest per image & operator)")
    visible_rows = [
        row
        for (img_name, _), row in st.session_state["latest_corner_runs"].items()
        if img_name in images
    ]
    st.dataframe(pd.DataFrame(visible_rows), width="stretch", hide_index=True)

    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("Pin current run"):
        st.session_state["pinned_corner_runs"].append(current_row.copy())
    if b2.button("Clear pinned", disabled=(len(st.session_state["pinned_corner_runs"]) == 0)):
        st.session_state["pinned_corner_runs"] = []
        st.rerun()

    if st.session_state["pinned_corner_runs"]:
        st.subheader("Pinned snapshots")
        st.dataframe(
            pd.DataFrame(st.session_state["pinned_corner_runs"]),
            width="stretch",
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Stage 2: SIFT Descriptors
# ---------------------------------------------------------------------------
elif stage == "2. SIFT":
    with st.sidebar:
        st.divider()
        st.header("SIFT Parameters")
        selected_name = st.selectbox("Inspect image", image_names, key="stage2_img")
        bind_widget("w_sift_sigma", "cfg_sift_sigma")
        st.slider(
            "Descriptor scale (σ)",
            1.0,
            4.0,
            step=0.1,
            key="w_sift_sigma",
            on_change=sync_widget,
            args=("w_sift_sigma", "cfg_sift_sigma"),
        )
        bind_widget("w_max_kp", "cfg_max_kp")
        st.slider(
            "Max keypoints per image",
            100,
            5000,
            step=100,
            key="w_max_kp",
            on_change=sync_widget,
            args=("w_max_kp", "cfg_max_kp"),
        )
        bind_widget("w_sift_keypoint_scale", "cfg_sift_keypoint_scale")
        st.slider(
            "Keypoint size scale",
            0.5,
            3.0,
            step=0.1,
            key="w_sift_keypoint_scale",
            on_change=sync_widget,
            args=("w_sift_keypoint_scale", "cfg_sift_keypoint_scale"),
        )
        bind_widget("w_sift_line_thickness", "cfg_sift_line_thickness")
        st.slider(
            "Keypoint line thickness",
            1,
            6,
            key="w_sift_line_thickness",
            on_change=sync_widget,
            args=("w_sift_line_thickness", "cfg_sift_line_thickness"),
        )
        bind_widget("w_sift_max_drawn", "cfg_sift_max_drawn")
        st.slider(
            "Max drawn keypoints",
            50,
            1000,
            step=50,
            key="w_sift_max_drawn",
            on_change=sync_widget,
            args=("w_sift_max_drawn", "cfg_sift_max_drawn"),
        )

    cur_img = images[selected_name]
    op, method_id, gauss_ksize, sigma, k_eff, threshold_ratio, nms_window = get_corner_params()
    sift_sigma = st.session_state["cfg_sift_sigma"]
    max_kp = st.session_state["cfg_max_kp"]

    detector_cache_key = (
        img_tokens[selected_name],
        method_id,
        gauss_ksize,
        round(sigma, 2),
        round(k_eff, 4),
        round(threshold_ratio, 4),
        nms_window,
    )
    xs, ys, _, _, corner_ms = run_detector(
        engine,
        img_tokens[selected_name],
        cur_img,
        method_id,
        gauss_ksize,
        sigma,
        k_eff,
        threshold_ratio,
        nms_window,
        need_response=False,
    )
    sift_x, sift_y = xs[:max_kp], ys[:max_kp]
    sift_cache_key = detector_cache_key + (round(sift_sigma, 2), max_kp)
    desc, orient, sift_ms = run_sift(
        engine,
        sift_cache_key,
        cur_img,
        sift_x,
        sift_y,
        sift_sigma,
    )
    current_sift_row = {
        "Image": selected_name,
        "Operator": op,
        "SIFT σ": round(sift_sigma, 2),
        "Max KP": max_kp,
        "Keypoints": len(sift_x),
        "Descriptor size": "128",
        "Corner time (ms)": round(corner_ms, 3),
        "SIFT time (ms)": round(sift_ms, 3),
    }
    sift_compute_sig = (
        selected_name,
        op,
        gauss_ksize,
        round(sigma, 2),
        round(k_eff, 4),
        round(threshold_ratio, 4),
        nms_window,
        round(sift_sigma, 2),
        max_kp,
    )
    if sift_compute_sig != st.session_state["last_sift_compute_sig"]:
        st.session_state["sift_log"].append(current_sift_row.copy())
        st.session_state["last_sift_compute_sig"] = sift_compute_sig

    st.caption(f"SIFT is computed on the {op} keypoints from Stage 1 (strongest first).")

    canvas = (
        cv2.cvtColor(cur_img, cv2.COLOR_GRAY2RGB)
        if cur_img.ndim == 2
        else cur_img.copy()
    )
    keypoint_scale = st.session_state["cfg_sift_keypoint_scale"]
    line_thickness = st.session_state["cfg_sift_line_thickness"]
    max_drawn = st.session_state["cfg_sift_max_drawn"]
    rad = max(2, int(round(3 * sift_sigma * 2.5 * keypoint_scale)))
    for x, y, th in list(zip(sift_x, sift_y, orient))[:max_drawn]:
        center = (int(x), int(y))
        endpoint = (
            int(x + rad * np.cos(th)),
            int(y + rad * np.sin(th)),
        )
        cv2.circle(canvas, center, rad, (0, 0, 0), line_thickness + 2, cv2.LINE_AA)
        cv2.circle(canvas, center, rad, (0, 255, 120), line_thickness, cv2.LINE_AA)
        cv2.line(
            canvas,
            center,
            endpoint,
            (0, 0, 0),
            line_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            center,
            endpoint,
            (255, 40, 40),
            line_thickness,
            cv2.LINE_AA,
        )
    st.caption(f"Circle = descriptor support region, red line = dominant orientation (first {max_drawn} keypoints).")
    st.image(canvas, width="stretch")

    st.subheader("SIFT computation log")
    visible_sift_rows = [
        row
        for row in st.session_state["sift_log"]
        if row["Image"] in images
    ]
    st.dataframe(pd.DataFrame(visible_sift_rows), width="stretch", hide_index=True)

    b1, b2, b3, _ = st.columns([1, 1, 1, 5])
    if b1.button("Pin current run", key="pin_sift"):
        st.session_state["pinned_sift_runs"].append(current_sift_row.copy())
    if b2.button("Clear pinned", key="clear_sift", disabled=(len(st.session_state["pinned_sift_runs"]) == 0)):
        st.session_state["pinned_sift_runs"] = []
        st.rerun()
    if b3.button("Clear log", key="clear_sift_log", disabled=(len(st.session_state["sift_log"]) == 0)):
        st.session_state["sift_log"] = []
        st.session_state["last_sift_compute_sig"] = sift_compute_sig
        st.rerun()

    if st.session_state["pinned_sift_runs"]:
        st.subheader("Pinned snapshots")
        st.dataframe(
            pd.DataFrame(st.session_state["pinned_sift_runs"]),
            width="stretch",
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Stage 3: Matching (SSD vs. NCC)
# ---------------------------------------------------------------------------
else:
    if len(image_names) < 2:
        st.info("Upload at least two images in the sidebar to match their features.")
        st.stop()

    with st.sidebar:
        st.divider()
        st.header("Matching Parameters")
        name_a = st.selectbox("Image A", image_names, index=0, key="match_a")
        name_b = st.selectbox("Image B", image_names, index=1, key="match_b")
        bind_widget("w_ratio", "cfg_ratio")
        st.slider(
            "Ratio test threshold",
            0.5,
            1.0,
            step=0.05,
            help="Lower = fewer but more reliable matches. 1.0 disables the test.",
            key="w_ratio",
            on_change=sync_widget,
            args=("w_ratio", "cfg_ratio"),
        )
        bind_widget("w_match_line_thickness", "cfg_match_line_thickness")
        st.slider(
            "Match line thickness",
            1,
            6,
            key="w_match_line_thickness",
            on_change=sync_widget,
            args=("w_match_line_thickness", "cfg_match_line_thickness"),
        )
        bind_widget("w_match_point_radius", "cfg_match_point_radius")
        st.slider(
            "Endpoint radius",
            2,
            10,
            key="w_match_point_radius",
            on_change=sync_widget,
            args=("w_match_point_radius", "cfg_match_point_radius"),
        )
        bind_widget("w_match_max_drawn", "cfg_match_max_drawn")
        st.slider(
            "Max matches drawn",
            10,
            300,
            step=10,
            key="w_match_max_drawn",
            on_change=sync_widget,
            args=("w_match_max_drawn", "cfg_match_max_drawn"),
        )

    op, method_id, gauss_ksize, sigma, k_eff, threshold_ratio, nms_window = get_corner_params()
    sift_sigma = st.session_state["cfg_sift_sigma"]
    max_kp = st.session_state["cfg_max_kp"]
    ratio = st.session_state["cfg_ratio"]

    img_a, img_b = images[name_a], images[name_b]
    xa, ya, _, _, _ = run_detector(
        engine,
        img_tokens[name_a],
        img_a,
        method_id,
        gauss_ksize,
        sigma,
        k_eff,
        threshold_ratio,
        nms_window,
        need_response=False,
    )
    xb, yb, _, _, _ = run_detector(
        engine,
        img_tokens[name_b],
        img_b,
        method_id,
        gauss_ksize,
        sigma,
        k_eff,
        threshold_ratio,
        nms_window,
        need_response=False,
    )
    sift_xa, sift_ya = xa[:max_kp], ya[:max_kp]
    sift_xb, sift_yb = xb[:max_kp], yb[:max_kp]
    sift_cache_base = (
        method_id,
        gauss_ksize,
        round(sigma, 2),
        round(k_eff, 4),
        round(threshold_ratio, 4),
        nms_window,
        round(sift_sigma, 2),
        max_kp,
    )
    sift_cache_a = (img_tokens[name_a],) + sift_cache_base
    sift_cache_b = (img_tokens[name_b],) + sift_cache_base
    desc_a, _, _ = run_sift(engine, sift_cache_a, img_a, sift_xa, sift_ya, sift_sigma)
    desc_b, _, _ = run_sift(engine, sift_cache_b, img_b, sift_xb, sift_yb, sift_sigma)

    out = {
        label: run_matcher(
            engine,
            sift_cache_a,
            sift_cache_b,
            desc_a,
            desc_b,
            mid,
            ratio,
        )
        for label, mid in (("SSD", 0), ("NCC", 1))
    }
    current_match_rows = []
    for label in ("SSD", "NCC"):
        row = {
            "Image A": name_a,
            "Image B": name_b,
            "Operator": op,
            "Method": label,
            "SIFT σ": round(sift_sigma, 2),
            "Ratio": round(ratio, 2),
            "Descriptors A": len(desc_a),
            "Descriptors B": len(desc_b),
            "Matches": len(out[label][0]),
            "Time (ms)": round(out[label][3], 3),
        }
        current_match_rows.append(row)
        st.session_state["latest_match_runs"][(name_a, name_b, op, label)] = row

    st.caption(f"Matching {op} + SIFT (σ={sift_sigma}, max {max_kp} keypoints) between `{name_a}` and `{name_b}`.")
    st.subheader("Computation time")
    st.dataframe(
        pd.DataFrame(current_match_rows),
        width="stretch",
        hide_index=True,
    )

    for label, higher_is_better in (("SSD", False), ("NCC", True)):
        i1, i2, m_scores, _ = out[label]
        st.subheader(f"{label} matches (top {st.session_state['cfg_match_max_drawn']} by score)")
        if len(i1) == 0:
            st.warning("No matches. Try a higher ratio threshold.")
        else:
            st.image(
                draw_matches(
                    img_a, sift_xa, sift_ya,
                    img_b, sift_xb, sift_yb,
                    i1, i2, m_scores,
                    higher_is_better=higher_is_better,
                    line_thickness=st.session_state["cfg_match_line_thickness"],
                    point_radius=st.session_state["cfg_match_point_radius"],
                    max_lines=st.session_state["cfg_match_max_drawn"],
                ),
                width="stretch",
            )

    st.subheader("Live comparison (latest per image pair, operator & method)")
    visible_match_rows = [
        row
        for (img_a_name, img_b_name, _, _), row in st.session_state["latest_match_runs"].items()
        if img_a_name in images and img_b_name in images
    ]
    st.dataframe(pd.DataFrame(visible_match_rows), width="stretch", hide_index=True)

    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("Pin current run", key="pin_match"):
        st.session_state["pinned_match_runs"].extend(
            row.copy() for row in current_match_rows
        )
    if b2.button("Clear pinned", key="clear_match", disabled=(len(st.session_state["pinned_match_runs"]) == 0)):
        st.session_state["pinned_match_runs"] = []
        st.rerun()

    if st.session_state["pinned_match_runs"]:
        st.subheader("Pinned snapshots")
        st.dataframe(
            pd.DataFrame(st.session_state["pinned_match_runs"]),
            width="stretch",
            hide_index=True,
        )