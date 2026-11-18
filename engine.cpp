#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

namespace {

inline int clamp_idx(int v, int max_v) {
    return (v < 0) ? 0 : (v >= max_v ? max_v - 1 : v);
}

struct Corner {
    int x;
    int y;
    double score;
};

// Build normalized 1D Gaussian kernel
std::vector<double> make_gaussian_kernel_1d(int ksize, double sigma) {
    if (ksize < 3) ksize = 3;
    if (ksize % 2 == 0) ksize += 1;
    if (sigma <= 0.0) sigma = 0.3 * ((ksize - 1) * 0.5 - 1.0) + 0.8;

    int radius = ksize / 2;
    std::vector<double> kernel(ksize);
    double sum = 0.0;
    double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

    for (int i = -radius; i <= radius; ++i) {
        double val = std::exp(-(i * i) * inv_two_sigma2);
        kernel[i + radius] = val;
        sum += val;
    }
    for (double& v : kernel) {
        v /= sum;
    }
    return kernel;
}

// Separable 2D Gaussian blur (horizontal pass + vertical pass) with replicate borders
void gaussian_blur_2d(
    const std::vector<double>& src,
    std::vector<double>& dst,
    std::vector<double>& temp,
    int height,
    int width,
    const std::vector<double>& kernel
) {
    int radius = static_cast<int>(kernel.size()) / 2;

    // Horizontal pass: src -> temp
    for (int r = 0; r < height; ++r) {
        size_t row_offset = static_cast<size_t>(r) * width;
        for (int c = 0; c < width; ++c) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int sc = clamp_idx(c + k, width);
                acc += src[row_offset + sc] * kernel[k + radius];
            }
            temp[row_offset + c] = acc;
        }
    }

    // Vertical pass: temp -> dst
    for (int r = 0; r < height; ++r) {
        size_t row_offset = static_cast<size_t>(r) * width;
        for (int c = 0; c < width; ++c) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int sr = clamp_idx(r + k, height);
                acc += temp[static_cast<size_t>(sr) * width + c] * kernel[k + radius];
            }
            dst[row_offset + c] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// SIFT descriptor helpers (applied on top of already-detected Harris keypoints)
// ---------------------------------------------------------------------------
constexpr double PI_D = 3.14159265358979323846;
constexpr int SIFT_D = 4;                       // 4x4 spatial grid
constexpr int SIFT_NB = 8;                      // 8 orientation bins per cell
constexpr int SIFT_DIM = SIFT_D * SIFT_D * SIFT_NB;  // 128
constexpr int ORI_BINS = 36;

// RGB -> luminance (or plain copy for grayscale), scaled to [0, 1]
void to_gray(const uint8_t* img, int h, int w, int ch, std::vector<double>& g) {
    g.assign(static_cast<size_t>(h) * w, 0.0);
    for (size_t i = 0; i < g.size(); ++i) {
        if (ch == 1) {
            g[i] = img[i] / 255.0;
        } else {
            g[i] = (0.299 * img[i * 3] + 0.587 * img[i * 3 + 1] + 0.114 * img[i * 3 + 2]) / 255.0;
        }
    }
}

// Central-difference gradient magnitude and orientation ([0, 2*pi)) for every pixel
void gradient_maps(const std::vector<double>& g, int h, int w,
                   std::vector<double>& mag, std::vector<double>& ang) {
    mag.assign(g.size(), 0.0);
    ang.assign(g.size(), 0.0);
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            double dx = 0.5 * (g[static_cast<size_t>(r) * w + clamp_idx(c + 1, w)] -
                               g[static_cast<size_t>(r) * w + clamp_idx(c - 1, w)]);
            double dy = 0.5 * (g[static_cast<size_t>(clamp_idx(r + 1, h)) * w + c] -
                               g[static_cast<size_t>(clamp_idx(r - 1, h)) * w + c]);
            size_t idx = static_cast<size_t>(r) * w + c;
            mag[idx] = std::sqrt(dx * dx + dy * dy);
            double a = std::atan2(dy, dx);
            if (a < 0) a += 2.0 * PI_D;
            ang[idx] = a;
        }
    }
}

// Dominant gradient orientation around (x, y): 36-bin Gaussian-weighted histogram,
// smoothed, peak refined by parabolic interpolation. Returns angle in [0, 2*pi).
double dominant_orientation(const std::vector<double>& mag, const std::vector<double>& ang,
                            int h, int w, int x, int y, double sigma) {
    double s = 1.5 * sigma;
    int rad = static_cast<int>(std::round(3.0 * s));
    double inv2s2 = 1.0 / (2.0 * s * s);
    double hist[ORI_BINS] = {0};

    for (int dy = -rad; dy <= rad; ++dy) {
        int r = y + dy;
        if (r < 1 || r >= h - 1) continue;
        for (int dx = -rad; dx <= rad; ++dx) {
            int c = x + dx;
            if (c < 1 || c >= w - 1) continue;
            size_t idx = static_cast<size_t>(r) * w + c;
            double wgt = std::exp(-(dx * dx + dy * dy) * inv2s2) * mag[idx];
            int bin = static_cast<int>(std::floor(ang[idx] * ORI_BINS / (2.0 * PI_D) + 0.5)) % ORI_BINS;
            hist[bin] += wgt;
        }
    }

    // circular smoothing
    for (int it = 0; it < 6; ++it) {
        double tmp[ORI_BINS];
        for (int i = 0; i < ORI_BINS; ++i) {
            tmp[i] = (hist[(i + ORI_BINS - 1) % ORI_BINS] + hist[i] + hist[(i + 1) % ORI_BINS]) / 3.0;
        }
        std::copy(tmp, tmp + ORI_BINS, hist);
    }

    int best = 0;
    for (int i = 1; i < ORI_BINS; ++i) if (hist[i] > hist[best]) best = i;

    double l = hist[(best + ORI_BINS - 1) % ORI_BINS];
    double m = hist[best];
    double rr = hist[(best + 1) % ORI_BINS];
    double denom = l - 2.0 * m + rr;
    double offset = (std::fabs(denom) > 1e-12) ? 0.5 * (l - rr) / denom : 0.0;

    double theta = (best + offset) * 2.0 * PI_D / ORI_BINS;
    if (theta < 0) theta += 2.0 * PI_D;
    if (theta >= 2.0 * PI_D) theta -= 2.0 * PI_D;
    return theta;
}

// 128-D SIFT descriptor (4x4 cells x 8 bins, trilinear interpolation, rotated to theta)
void sift_descriptor(const std::vector<double>& mag, const std::vector<double>& ang,
                     int h, int w, int x, int y, double sigma, double theta, float* out) {
    double hist_w = 3.0 * sigma;  // width of one spatial cell in pixels
    int rad = static_cast<int>(std::round(hist_w * std::sqrt(2.0) * (SIFT_D + 1) * 0.5));
    double cos_t = std::cos(theta), sin_t = std::sin(theta);
    double hist[SIFT_D][SIFT_D][SIFT_NB] = {};

    for (int dy = -rad; dy <= rad; ++dy) {
        int r = y + dy;
        if (r < 1 || r >= h - 1) continue;
        for (int dx = -rad; dx <= rad; ++dx) {
            int c = x + dx;
            if (c < 1 || c >= w - 1) continue;

            // rotate the sample into the keypoint's canonical frame, in cell units
            double cr = (cos_t * dx + sin_t * dy) / hist_w;
            double rr = (-sin_t * dx + cos_t * dy) / hist_w;
            double rbin = rr + SIFT_D / 2.0 - 0.5;
            double cbin = cr + SIFT_D / 2.0 - 0.5;
            if (rbin <= -1.0 || rbin >= SIFT_D || cbin <= -1.0 || cbin >= SIFT_D) continue;

            size_t idx = static_cast<size_t>(r) * w + c;
            double a = ang[idx] - theta;
            while (a < 0) a += 2.0 * PI_D;
            while (a >= 2.0 * PI_D) a -= 2.0 * PI_D;
            double obin = a * SIFT_NB / (2.0 * PI_D);

            double wgt = std::exp(-(rr * rr + cr * cr) / (0.5 * SIFT_D * SIFT_D)) * mag[idx];

            int r0 = static_cast<int>(std::floor(rbin));
            int c0 = static_cast<int>(std::floor(cbin));
            int o0 = static_cast<int>(std::floor(obin));
            double fr = rbin - r0, fc = cbin - c0, fo = obin - o0;

            for (int i = 0; i < 2; ++i) {
                int ri = r0 + i;
                if (ri < 0 || ri >= SIFT_D) continue;
                double wr = wgt * (i ? fr : 1.0 - fr);
                for (int j = 0; j < 2; ++j) {
                    int ci = c0 + j;
                    if (ci < 0 || ci >= SIFT_D) continue;
                    double wc = wr * (j ? fc : 1.0 - fc);
                    for (int k = 0; k < 2; ++k) {
                        int oi = (o0 + k) % SIFT_NB;
                        hist[ri][ci][oi] += wc * (k ? fo : 1.0 - fo);
                    }
                }
            }
        }
    }

    // flatten -> normalize -> clip at 0.2 -> renormalize (illumination invariance)
    double v[SIFT_DIM];
    int n = 0;
    for (int i = 0; i < SIFT_D; ++i)
        for (int j = 0; j < SIFT_D; ++j)
            for (int k = 0; k < SIFT_NB; ++k) v[n++] = hist[i][j][k];

    for (int pass = 0; pass < 2; ++pass) {
        double norm = 0.0;
        for (int i = 0; i < SIFT_DIM; ++i) norm += v[i] * v[i];
        norm = std::sqrt(norm);
        if (norm < 1e-12) norm = 1.0;
        for (int i = 0; i < SIFT_DIM; ++i) {
            v[i] /= norm;
            if (pass == 0 && v[i] > 0.2) v[i] = 0.2;
        }
    }
    for (int i = 0; i < SIFT_DIM; ++i) out[i] = static_cast<float>(v[i]);
}

// For each row of A, find nearest row in B (and second-nearest distance).
// method 0: dist = SSD;  method 1: dist = 1 - NCC (A, B must be zero-mean & unit-norm).
void nearest_neighbours(const std::vector<float>& A, int na, const std::vector<float>& B, int nb,
                        int method, std::vector<int>& idx, std::vector<double>& d1,
                        std::vector<double>& d2) {
    idx.assign(na, -1);
    d1.assign(na, std::numeric_limits<double>::infinity());
    d2.assign(na, std::numeric_limits<double>::infinity());
    for (int i = 0; i < na; ++i) {
        const float* a = &A[static_cast<size_t>(i) * SIFT_DIM];
        for (int j = 0; j < nb; ++j) {
            const float* b = &B[static_cast<size_t>(j) * SIFT_DIM];
            double acc = 0.0;
            if (method == 0) {
                for (int k = 0; k < SIFT_DIM; ++k) {
                    double d = static_cast<double>(a[k]) - b[k];
                    acc += d * d;
                }
            } else {
                for (int k = 0; k < SIFT_DIM; ++k) acc += static_cast<double>(a[k]) * b[k];
                acc = 1.0 - acc;
            }
            if (acc < d1[i]) {
                d2[i] = d1[i];
                d1[i] = acc;
                idx[i] = j;
            } else if (acc < d2[i]) {
                d2[i] = acc;
            }
        }
    }
}

// zero-mean + unit-norm each descriptor so that a dot product == NCC
std::vector<float> normalize_for_ncc(const float* d, int n) {
    std::vector<float> out(static_cast<size_t>(n) * SIFT_DIM);
    for (int i = 0; i < n; ++i) {
        const float* s = d + static_cast<size_t>(i) * SIFT_DIM;
        float* o = &out[static_cast<size_t>(i) * SIFT_DIM];
        double mean = 0.0;
        for (int k = 0; k < SIFT_DIM; ++k) mean += s[k];
        mean /= SIFT_DIM;
        double norm = 0.0;
        for (int k = 0; k < SIFT_DIM; ++k) {
            o[k] = static_cast<float>(s[k] - mean);
            norm += static_cast<double>(o[k]) * o[k];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-12) norm = 1.0;
        for (int k = 0; k < SIFT_DIM; ++k) o[k] = static_cast<float>(o[k] / norm);
    }
    return out;
}

} // namespace

/**
 * Shared C API entry point.
 *
 * @param img_data         Contiguous uint8 buffer: (H*W) for grayscale, (H*W*3) for RGB
 * @param height           Image height
 * @param width            Image width
 * @param channels         1 (grayscale) or 3 (RGB)
 * @param method           0 = Harris (R = det - k*trace^2), 1 = Shi-Tomasi (lambda-)
 * @param gauss_ksize      Gaussian kernel size (odd >= 3)
 * @param sigma            Gaussian standard deviation
 * @param k                Harris sensitivity constant (e.g., 0.04)
 * @param threshold_ratio  Relative threshold in (0, 1], multiplied by max positive response
 * @param nms_window       Non-Maximum Suppression window size (odd >= 3)
 * @param max_corners      Capacity of out_x, out_y, out_scores
 * @param out_x            Output buffer for corner x-coordinates (columns)
 * @param out_y            Output buffer for corner y-coordinates (rows)
 * @param out_scores       Output buffer for corner response values
 * @param out_response     Optional (H*W) double buffer for full response map (pass nullptr to skip)
 * @param out_time_ms      Output pointer for execution time in milliseconds
 * @return                 Number of detected corners written to output buffers
 */
EXPORT int detect_corners(
    const uint8_t* img_data,
    int height,
    int width,
    int channels,
    int method,
    int gauss_ksize,
    double sigma,
    double k,
    double threshold_ratio,
    int nms_window,
    int max_corners,
    int* out_x,
    int* out_y,
    double* out_scores,
    double* out_response,
    double* out_time_ms
) {
    if (!img_data || height < 3 || width < 3 || (channels != 1 && channels != 3)) {
        if (out_time_ms) *out_time_ms = 0.0;
        return 0;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    const size_t n_pixels = static_cast<size_t>(height) * width;
    std::vector<double> Ixx(n_pixels, 0.0);
    std::vector<double> Iyy(n_pixels, 0.0);
    std::vector<double> Ixy(n_pixels, 0.0);

    // 1. Compute Sobel derivatives and accumulate structure tensor across channels
    //    For grayscale (channels == 1): standard Ixx, Iyy, Ixy
    //    For RGB (channels == 3):       sum of products over R, G, B
    constexpr double inv_255 = 1.0 / 255.0;

    for (int ch = 0; ch < channels; ++ch) {
        auto pixel_at = [&](int r, int c) -> double {
            int rr = clamp_idx(r, height);
            int cc = clamp_idx(c, width);
            size_t idx = (static_cast<size_t>(rr) * width + cc) * channels + ch;
            return static_cast<double>(img_data[idx]) * inv_255;
        };

        for (int r = 0; r < height; ++r) {
            size_t row_offset = static_cast<size_t>(r) * width;
            for (int c = 0; c < width; ++c) {
                double p00 = pixel_at(r - 1, c - 1);
                double p01 = pixel_at(r - 1, c);
                double p02 = pixel_at(r - 1, c + 1);
                double p10 = pixel_at(r, c - 1);
                double p12 = pixel_at(r, c + 1);
                double p20 = pixel_at(r + 1, c - 1);
                double p21 = pixel_at(r + 1, c);
                double p22 = pixel_at(r + 1, c + 1);

                double gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
                double gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

                size_t idx = row_offset + c;
                Ixx[idx] += gx * gx;
                Iyy[idx] += gy * gy;
                Ixy[idx] += gx * gy;
            }
        }
    }

    // 2. Smooth Ixx, Iyy, Ixy with 2D Gaussian filter
    std::vector<double> kernel = make_gaussian_kernel_1d(gauss_ksize, sigma);
    std::vector<double> Sxx(n_pixels), Syy(n_pixels), Sxy(n_pixels), temp(n_pixels);

    gaussian_blur_2d(Ixx, Sxx, temp, height, width, kernel);
    gaussian_blur_2d(Iyy, Syy, temp, height, width, kernel);
    gaussian_blur_2d(Ixy, Sxy, temp, height, width, kernel);

    // 3. Compute Corner Response (Harris R or Shi-Tomasi lambda-)
    std::vector<double> response(n_pixels, 0.0);
    double max_resp = 0.0;

    for (size_t i = 0; i < n_pixels; ++i) {
        double sxx = Sxx[i];
        double syy = Syy[i];
        double sxy = Sxy[i];
        double val = 0.0;

        if (method == 0) {
            // Harris: R = det(M) - k * trace(M)^2
            double det = sxx * syy - sxy * sxy;
            double trace = sxx + syy;
            val = det - k * trace * trace;
        } else {
            // Shi-Tomasi: lambda- = 0.5 * ((Sxx + Syy) - sqrt((Sxx - Syy)^2 + 4*Sxy^2))
            double trace = sxx + syy;
            double diff = sxx - syy;
            double disc = std::sqrt(diff * diff + 4.0 * sxy * sxy);
            val = 0.5 * (trace - disc);
        }

        response[i] = val;
        if (val > max_resp) {
            max_resp = val;
        }
    }

    // 4. Thresholding + Non-Maximum Suppression (NMS)
    if (nms_window < 3) nms_window = 3;
    if (nms_window % 2 == 0) nms_window += 1;
    int nms_rad = nms_window / 2;

    std::vector<Corner> corners;
    if (max_resp > 1e-12) {
        double thresh = threshold_ratio * max_resp;

        for (int r = nms_rad; r < height - nms_rad; ++r) {
            for (int c = nms_rad; c < width - nms_rad; ++c) {
                size_t idx = static_cast<size_t>(r) * width + c;
                double val = response[idx];
                if (val <= thresh) continue;

                bool is_local_max = true;
                for (int dr = -nms_rad; dr <= nms_rad && is_local_max; ++dr) {
                    size_t nb_row = static_cast<size_t>(r + dr) * width;
                    for (int dc = -nms_rad; dc <= nms_rad; ++dc) {
                        if (dr == 0 && dc == 0) continue;
                        size_t nb_idx = nb_row + (c + dc);
                        double nb_val = response[nb_idx];
                        // Deterministic tie-breaking on flat plateaus
                        if (nb_val > val || (nb_val == val && nb_idx < idx)) {
                            is_local_max = false;
                            break;
                        }
                    }
                }

                if (is_local_max) {
                    corners.push_back({c, r, val});
                }
            }
        }
    }

    // Sort strongest corners first so truncation (if any) preserves top features
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) {
        return a.score > b.score;
    });

    auto t_end = std::chrono::high_resolution_clock::now();
    if (out_time_ms) {
        *out_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }

    // Copy response map if caller requested it (for heatmap visualization)
    if (out_response) {
        std::copy(response.begin(), response.end(), out_response);
    }

    int count = std::min(static_cast<int>(corners.size()), max_corners);
    for (int i = 0; i < count; ++i) {
        out_x[i] = corners[i].x;
        out_y[i] = corners[i].y;
        out_scores[i] = corners[i].score;
    }

    return count;
}

/**
 * Compute 128-D SIFT descriptors at pre-detected (Harris / Shi-Tomasi) keypoints.
 *
 * Pipeline: grayscale -> Gaussian smoothing (sigma) -> gradient maps ->
 *           dominant orientation per keypoint -> 4x4x8 rotated histogram -> normalize/clip.
 * Harris is single-scale, so every keypoint uses the same descriptor scale `sigma`.
 *
 * @param kp_x, kp_y   Keypoint coordinates (as returned by detect_corners)
 * @param n_kp         Number of keypoints
 * @param sigma        Descriptor scale (typ. 1.6 - 3.0)
 * @param out_desc     Output buffer of n_kp * 128 floats (row-major)
 * @param out_orient   Optional output buffer of n_kp doubles (orientation, radians); may be nullptr
 * @param out_time_ms  Output pointer for execution time in milliseconds
 * @return             Number of descriptors written (== n_kp), 0 on invalid input
 */
EXPORT int compute_sift_descriptors(
    const uint8_t* img_data,
    int height,
    int width,
    int channels,
    const int* kp_x,
    const int* kp_y,
    int n_kp,
    double sigma,
    float* out_desc,
    double* out_orient,
    double* out_time_ms
) {
    if (!img_data || !kp_x || !kp_y || !out_desc || height < 3 || width < 3 ||
        (channels != 1 && channels != 3) || n_kp <= 0) {
        if (out_time_ms) *out_time_ms = 0.0;
        return 0;
    }
    if (sigma < 0.5) sigma = 0.5;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<double> gray, smooth, temp(static_cast<size_t>(height) * width);
    to_gray(img_data, height, width, channels, gray);
    smooth.resize(gray.size());

    // pre-smooth at the descriptor scale
    int ksize = 2 * static_cast<int>(std::ceil(3.0 * sigma)) + 1;
    gaussian_blur_2d(gray, smooth, temp, height, width, make_gaussian_kernel_1d(ksize, sigma));

    std::vector<double> mag, ang;
    gradient_maps(smooth, height, width, mag, ang);

    for (int i = 0; i < n_kp; ++i) {
        int x = clamp_idx(kp_x[i], width);
        int y = clamp_idx(kp_y[i], height);
        double theta = dominant_orientation(mag, ang, height, width, x, y, sigma);
        sift_descriptor(mag, ang, height, width, x, y, sigma, theta,
                        out_desc + static_cast<size_t>(i) * SIFT_DIM);
        if (out_orient) out_orient[i] = theta;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    if (out_time_ms) *out_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return n_kp;
}

/**
 * Match two sets of 128-D descriptors.
 *
 * @param method       0 = SSD (minimise), 1 = normalized cross-correlation (maximise)
 * @param ratio        Lowe ratio-test threshold (e.g. 0.8). Use >= 1.0 to disable.
 * @param cross_check  1 = keep only mutual best matches
 * @param out_i/out_j  Indices into set 1 / set 2 for every accepted match
 * @param out_score    SSD value (method 0) or NCC value in [-1, 1] (method 1)
 * @param out_time_ms  Execution time in milliseconds
 * @return             Number of matches written
 */
EXPORT int match_descriptors(
    const float* desc1,
    int n1,
    const float* desc2,
    int n2,
    int method,
    double ratio,
    int cross_check,
    int max_matches,
    int* out_i,
    int* out_j,
    double* out_score,
    double* out_time_ms
) {
    if (!desc1 || !desc2 || n1 <= 0 || n2 <= 0 || max_matches <= 0) {
        if (out_time_ms) *out_time_ms = 0.0;
        return 0;
    }
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<float> A, B;
    if (method == 1) {
        A = normalize_for_ncc(desc1, n1);
        B = normalize_for_ncc(desc2, n2);
    } else {
        A.assign(desc1, desc1 + static_cast<size_t>(n1) * SIFT_DIM);
        B.assign(desc2, desc2 + static_cast<size_t>(n2) * SIFT_DIM);
    }

    std::vector<int> fwd, bwd;
    std::vector<double> f1, f2, b1, b2;
    nearest_neighbours(A, n1, B, n2, method, fwd, f1, f2);
    if (cross_check) nearest_neighbours(B, n2, A, n1, method, bwd, b1, b2);

    // SSD is a squared distance, so square the ratio; NCC uses (1 - ncc) directly.
    double r = (method == 0) ? ratio * ratio : ratio;
    int count = 0;
    for (int i = 0; i < n1 && count < max_matches; ++i) {
        int j = fwd[i];
        if (j < 0) continue;
        if (f2[i] < std::numeric_limits<double>::infinity() && f1[i] > r * f2[i]) continue;
        if (cross_check && bwd[j] != i) continue;
        out_i[count] = i;
        out_j[count] = j;
        out_score[count] = (method == 0) ? f1[i] : 1.0 - f1[i];
        ++count;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    if (out_time_ms) *out_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return count;
}
