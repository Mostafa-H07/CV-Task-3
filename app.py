import ctypes
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Harris / Scale-Invariant SIFT Pipeline", layout="wide")

_DP = ctypes.POINTER(ctypes.c_double)
_FP = ctypes.POINTER(ctypes.c_float)
_IP = ctypes.POINTER(ctypes.c_int)
_UP = ctypes.POINTER(ctypes.c_uint8)


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
        _UP,                              # img_data
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
        _IP,                              # out_x
        _IP,                              # out_y
        _DP,                              # out_scores
        _DP,                              # out_response
        _DP,                              # out_time_ms
    ]
    lib.detect_corners.restype = ctypes.c_int

    # Scale-space SIFT + matching (only present if engine was rebuilt from the latest engine.cpp)
    if hasattr(lib, "detect_and_describe_sift") and hasattr(lib, "match_descriptors"):
        lib.detect_and_describe_sift.argtypes = [
            _UP,                          # img_data
            ctypes.c_int,                 # height
            ctypes.c_int,                 # width
            ctypes.c_int,                 # channels
            ctypes.c_int,                 # n_layers (S)
            ctypes.c_double,              # sigma0
            ctypes.c_double,              # contrast_thr
            ctypes.c_double,              # edge_thr (r)
            ctypes.c_int,                 # upsample
            ctypes.c_int,                 # max_kp
            _DP,                          # out_x (original-image pixels)
            _DP,                          # out_y
            _DP,                          # out_sigma (characteristic scale)
            _DP,                          # out_orient
            _DP,                          # out_response
            _FP,                          # out_desc (max_kp * 128)
            _DP,                          # out_time_ms
        ]
        lib.detect_and_describe_sift.restype = ctypes.c_int

        lib.match_descriptors.argtypes = [
            _FP,                          # desc1
            ctypes.c_int,                 # n1
            _FP,                          # desc2
            ctypes.c_int,                 # n2
            ctypes.c_int,                 # method (0=SSD, 1=NCC)
            ctypes.c_double,              # ratio
            _IP,                          # out_i
            _IP,                          # out_j
            _DP,                          # out_score
            _DP,                          # out_time_ms
        ]
        lib.match_descriptors.restype = ctypes.c_int
    return lib, str(lib_path)


def _as_uint8(img: np.ndarray):
    img_c = np.ascontiguousarray(img, dtype=np.uint8)
    if img_c.ndim == 2:
        h, w = img_c.shape
        channels = 1
    else:
        h, w, channels = img_c.shape
    return img_c, h, w, channels


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
    img_c, h, w, channels = _as_uint8(_img)

    out_x = np.empty(max_corners, dtype=np.int32)
    out_y = np.empty(max_corners, dtype=np.int32)
    out_scores = np.empty(max_corners, dtype=np.float64)
    out_response = np.empty((h, w), dtype=np.float64) if need_response else None
    time_ms = ctypes.c_double(0.0)
    response_ptr = (
        out_response.ctypes.data_as(_DP) if out_response is not None else _DP()
    )

    count = _lib.detect_corners(
        img_c.ctypes.data_as(_UP),
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
        out_x.ctypes.data_as(_IP),
        out_y.ctypes.data_as(_IP),
        out_scores.ctypes.data_as(_DP),
        response_ptr,
        ctypes.byref(time_ms),
    )

    return out_x[:count].copy(), out_y[:count].copy(), out_scores[:count].copy(), out_response, time_ms.value


@st.cache_data(show_spinner=False)
def run_dog_sift(
    _lib,
    cache_key,
    _img: np.ndarray,
    n_layers: int,
    sigma0: float,
    contrast_thr: float,
    edge_thr: float,
    upsample: bool,
    max_kp: int,
):
    """Full scale-space SIFT: Gaussian/DoG pyramid -> 3-D extrema -> orientation -> descriptor.

    Returns (xs, ys, sigmas, orientations, responses, descriptors, time_ms); positions and
    sigmas are in original-image pixels, every keypoint carries its own characteristic scale.
    """
    img_c, h, w, channels = _as_uint8(_img)

    xs = np.empty(max_kp, dtype=np.float64)
    ys = np.empty(max_kp, dtype=np.float64)
    sigmas = np.empty(max_kp, dtype=np.float64)
    orient = np.empty(max_kp, dtype=np.float64)
    resp = np.empty(max_kp, dtype=np.float64)
    desc = np.zeros((max_kp, 128), dtype=np.float32)
    time_ms = ctypes.c_double(0.0)

    n = _lib.detect_and_describe_sift(
        img_c.ctypes.data_as(_UP),
        h,
        w,
        channels,
        int(n_layers),
        float(sigma0),
        float(contrast_thr),
        float(edge_thr),
        int(upsample),
        int(max_kp),
        xs.ctypes.data_as(_DP),
        ys.ctypes.data_as(_DP),
        sigmas.ctypes.data_as(_DP),
        orient.ctypes.data_as(_DP),
        resp.ctypes.data_as(_DP),
        desc.ctypes.data_as(_FP),
        ctypes.byref(time_ms),
    )
    n = max(int(n), 0)
    return (
        xs[:n].copy(),
        ys[:n].copy(),
        sigmas[:n].copy(),
        orient[:n].copy(),
        resp[:n].copy(),
        desc[:n].copy(),
        time_ms.value,
    )


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
        d1.ctypes.data_as(_FP),
        len(d1),
        d2.ctypes.data_as(_FP),
        len(d2),
        method,
        ratio,
        out_i.ctypes.data_as(_IP),
        out_j.ctypes.data_as(_IP),
        out_s.ctypes.data_as(_DP),
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


def draw_sift_keypoints(
    img: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    sigmas: np.ndarray,
    orient: np.ndarray,
    size_mult: float,
    thickness: int,
    max_drawn: int,
):
    """Draw each keypoint with ITS OWN scale: circle radius = size_mult * sigma.

    Circle colour encodes the scale (viridis over log2 sigma); the red line is the
    keypoint's dominant orientation.
    """
    canvas = (
        cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    )
    n = min(len(xs), max_drawn)
    if n == 0:
        return canvas

    log_s = np.log2(np.maximum(sigmas[:n], 1e-6))
    span = max(float(log_s.max() - log_s.min()), 1e-6)
    norm = ((log_s - log_s.min()) / span * 255.0).astype(np.uint8).reshape(-1, 1)
    palette_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    palette = palette_bgr[:, ::-1]                      # canvas is RGB
    max_rad = int(max(img.shape[0], img.shape[1]))

    # draw large (coarse-scale) keypoints first so small ones stay visible on top
    for i in np.argsort(-sigmas[:n]):
        x, y, s, th = float(xs[i]), float(ys[i]), float(sigmas[i]), float(orient[i])
        rad = int(min(max(2, round(size_mult * s)), max_rad))
        center = (int(round(x)), int(round(y)))
        endpoint = (
            int(round(x + rad * np.cos(th))),
            int(round(y + rad * np.sin(th))),
        )
        color = tuple(int(v) for v in palette[i])
        cv2.circle(canvas, center, rad, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.circle(canvas, center, rad, color, thickness, cv2.LINE_AA)
        cv2.line(canvas, center, endpoint, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.line(canvas, center, endpoint, (255, 40, 40), thickness, cv2.LINE_AA)
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
        p1 = (int(round(xs1[idx1[m]])), int(round(ys1[idx1[m]])))
        p2 = (int(round(xs2[idx2[m]])) + off, int(round(ys2[idx2[m]])))
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


def scale_histogram(sigmas: np.ndarray, sigma0: float):
    """Keypoint count per scale band [sigma0*2^k, sigma0*2^(k+1)) in original pixels."""
    if len(sigmas) == 0:
        return pd.DataFrame({"Keypoints": []})
    k = np.floor(np.log2(np.maximum(sigmas, 1e-6) / sigma0)).astype(int)
    counts = pd.Series(k).value_counts().sort_index()
    labels = [
        f"B{int(i):02d}: σ {sigma0 * 2.0 ** int(i):.1f}–{sigma0 * 2.0 ** (int(i) + 1):.1f} px"
        for i in counts.index
    ]
    return pd.DataFrame({"Keypoints": counts.values}, index=labels)


def init_state():
    defaults = {
        "cfg_operator": "Harris (R)",
        "cfg_gauss_ksize": 5,
        "cfg_sigma": 1.0,
        "cfg_k": 0.04,
        "cfg_threshold_pct": 1.0,
        "cfg_nms_window": 5,
        "cfg_marker_radius": 3,
        "cfg_sift_layers": 3,
        "cfg_sift_sigma0": 1.6,
        "cfg_sift_contrast": 0.04,
        "cfg_sift_edge": 10.0,
        "cfg_sift_upsample": True,
        "cfg_max_kp": 1500,
        "cfg_ratio": 0.8,
        "cfg_sift_circle_mult": 6.0,
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
if engine is None or not hasattr(engine, "detect_and_describe_sift"):
    st.title("Feature Extraction: Harris / Shi-Tomasi → Scale-Invariant SIFT → Matching")
    st.error(
        "Shared library missing or outdated (the scale-space SIFT functions were not found). "
        "Compile the latest `engine.cpp` next to `app.py`:\n"
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

st.title("Feature Extraction: Harris / Shi-Tomasi → Scale-Invariant SIFT → Matching")

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

if len(image_names) >= 2:
    if "match_a" in st.session_state and st.session_state["match_a"] not in image_names:
        st.session_state["match_a"] = image_names[0]
    if (
        "match_b" in st.session_state
        and (
            st.session_state["match_b"] not in image_names
            or st.session_state["match_b"] == st.session_state.get("match_a")
        )
    ):
        st.session_state["match_b"] = next(
            (n for n in image_names if n != st.session_state.get("match_a", image_names[0])),
            image_names[1],
        )

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


def get_sift_params():
    """(n_layers, sigma0, contrast_thr, edge_thr, upsample) from the session settings."""
    s = st.session_state
    return (
        int(s["cfg_sift_layers"]),
        round(float(s["cfg_sift_sigma0"]), 2),
        round(float(s["cfg_sift_contrast"]), 4),
        round(float(s["cfg_sift_edge"]), 1),
        bool(s["cfg_sift_upsample"]),
    )


def compute_features(name: str):
    """Keypoints (x, y, sigma, orientation) + 128-D descriptors for one image."""
    img = images[name]
    layers, sigma0, contrast, edge, upsample = get_sift_params()
    max_kp = st.session_state["cfg_max_kp"]
    key = (img_tokens[name], "dog", layers, sigma0, contrast, edge, upsample, max_kp)
    xs, ys, sg, ori, resp, desc, ms = run_dog_sift(
        engine, key, img, layers, sigma0, contrast, edge, upsample, max_kp
    )
    return {
        "key": key,
        "xs": xs,
        "ys": ys,
        "sigmas": sg,
        "orient": ori,
        "desc": desc,
        "detect_ms": None,
        "sift_ms": ms,
        "sigma0": sigma0,
    }


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
# Stage 2: Scale-invariant SIFT
# ---------------------------------------------------------------------------
elif stage == "2. SIFT":
    with st.sidebar:
        st.divider()
        st.header("SIFT Parameters")
        selected_name = st.selectbox("Inspect image", image_names, key="stage2_img")

        bind_widget("w_sift_layers", "cfg_sift_layers")
        st.select_slider(
            "Scales per octave (S)",
            options=[2, 3, 4, 5],
            key="w_sift_layers",
            on_change=sync_widget,
            args=("w_sift_layers", "cfg_sift_layers"),
        )
        bind_widget("w_sift_sigma0", "cfg_sift_sigma0")
        st.slider(
            "Base blur σ₀",
            1.0,
            2.5,
            step=0.1,
            key="w_sift_sigma0",
            on_change=sync_widget,
            args=("w_sift_sigma0", "cfg_sift_sigma0"),
        )
        bind_widget("w_sift_upsample", "cfg_sift_upsample")
        st.checkbox(
            "Double image size first (finds the smallest features)",
            key="w_sift_upsample",
            on_change=sync_widget,
            args=("w_sift_upsample", "cfg_sift_upsample"),
        )
        bind_widget("w_sift_contrast", "cfg_sift_contrast")
        st.slider(
            "DoG contrast threshold",
            0.01,
            0.12,
            step=0.005,
            format="%.3f",
            key="w_sift_contrast",
            on_change=sync_widget,
            args=("w_sift_contrast", "cfg_sift_contrast"),
        )
        bind_widget("w_sift_edge", "cfg_sift_edge")
        st.slider(
            "Edge threshold (r)",
            4.0,
            20.0,
            step=1.0,
            key="w_sift_edge",
            on_change=sync_widget,
            args=("w_sift_edge", "cfg_sift_edge"),
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

        st.subheader("Display")
        bind_widget("w_sift_circle_mult", "cfg_sift_circle_mult")
        st.slider(
            "Circle radius (× σ)",
            1.0,
            8.0,
            step=0.5,
            key="w_sift_circle_mult",
            help="6 × σ is the half-width of the descriptor support region (4 cells of 3σ).",
            on_change=sync_widget,
            args=("w_sift_circle_mult", "cfg_sift_circle_mult"),
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
    feats = compute_features(selected_name)
    layers, sigma0, contrast, edge, upsample = get_sift_params()
    max_kp = st.session_state["cfg_max_kp"]
    sigmas = feats["sigmas"]
    n_kp = len(feats["xs"])
    total_ms = feats["sift_ms"]
    current_sift_row = {
        "Image": selected_name,
        "S": layers,
        "σ₀": sigma0,
        "Contrast": f"{contrast:.3f}",
        "Edge r": f"{edge:.0f}",
        "Upsample": "yes" if upsample else "no",
        "Max KP": max_kp,
        "Keypoints": n_kp,
        "σ min / median / max": (
            f"{sigmas.min():.2f} / {np.median(sigmas):.2f} / {sigmas.max():.2f}" if n_kp else "—"
        ),
        "Descriptor size": "128",
        "SIFT time (ms)": round(total_ms, 3),
    }

    sift_compute_sig = (selected_name,) + tuple(feats["key"])
    if sift_compute_sig != st.session_state["last_sift_compute_sig"]:
        st.session_state["sift_log"].append(current_sift_row.copy())
        st.session_state["last_sift_compute_sig"] = sift_compute_sig

    st.caption(
        "Keypoints come from Difference-of-Gaussian extrema in (x, y, scale). "
        "Each keypoint has its own characteristic σ, which sets its orientation window "
        "and descriptor cell width. SIFT time covers pyramid, detection, orientation and descriptors."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Keypoints", n_kp)
    m2.metric("Smallest σ (px)", f"{sigmas.min():.2f}" if n_kp else "—")
    m3.metric("Largest σ (px)", f"{sigmas.max():.2f}" if n_kp else "—")
    m4.metric("SIFT time (ms)", f"{total_ms:.1f}")

    canvas = draw_sift_keypoints(
        cur_img,
        feats["xs"],
        feats["ys"],
        sigmas,
        feats["orient"],
        st.session_state["cfg_sift_circle_mult"],
        st.session_state["cfg_sift_line_thickness"],
        st.session_state["cfg_sift_max_drawn"],
    )
    st.caption(
        f"Circle radius = {st.session_state['cfg_sift_circle_mult']:g} × the keypoint's own σ "
        f"(colour = scale, purple small → yellow large); red line = dominant orientation "
        f"(first {st.session_state['cfg_sift_max_drawn']} keypoints)."
    )
    st.image(canvas, width="stretch")

    st.subheader("Keypoints per scale band")
    st.bar_chart(scale_histogram(sigmas, sigma0))

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

    ratio = st.session_state["cfg_ratio"]
    max_kp = st.session_state["cfg_max_kp"]
    layers, sigma0, contrast, edge, upsample = get_sift_params()

    img_a, img_b = images[name_a], images[name_b]
    fa = compute_features(name_a)
    fb = compute_features(name_b)

    out = {
        label: run_matcher(
            engine,
            fa["key"],
            fb["key"],
            fa["desc"],
            fb["desc"],
            mid,
            ratio,
        )
        for label, mid in (("SSD", 0), ("NCC", 1))
    }
    current_match_rows = []
    for label in ("SSD", "NCC"):
        i1, i2, _, t_ms = out[label]
        if len(i1):
            scale_ratio = float(np.median(fb["sigmas"][i2] / fa["sigmas"][i1]))
            scale_txt = f"{scale_ratio:.3f}"
        else:
            scale_txt = "—"
        row = {
            "Image A": name_a,
            "Image B": name_b,
            "Method": label,
            "S": layers,
            "σ₀": sigma0,
            "Ratio": round(ratio, 2),
            "Descriptors A": len(fa["desc"]),
            "Descriptors B": len(fb["desc"]),
            "Matches": len(i1),
            "Median σ_B / σ_A": scale_txt,
            "Time (ms)": round(t_ms, 3),
        }
        current_match_rows.append(row)
        st.session_state["latest_match_runs"][(name_a, name_b, label)] = row

    st.caption(
        f"Matching scale-invariant DoG SIFT (max {max_kp} keypoints) between "
        f"`{name_a}` and `{name_b}`. “Median σ_B / σ_A” is the scale change between the two "
        "images estimated from the matched keypoints."
    )
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
                    img_a, fa["xs"], fa["ys"],
                    img_b, fb["xs"], fb["ys"],
                    i1, i2, m_scores,
                    higher_is_better=higher_is_better,
                    line_thickness=st.session_state["cfg_match_line_thickness"],
                    point_radius=st.session_state["cfg_match_point_radius"],
                    max_lines=st.session_state["cfg_match_max_drawn"],
                ),
                width="stretch",
            )

    st.subheader("Live comparison (latest per image pair & method)")
    visible_match_rows = [
        row
        for (img_a_name, img_b_name, _), row in st.session_state["latest_match_runs"].items()
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
