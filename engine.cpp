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
// SIFT descriptor (computed at already-detected Harris / Shi-Tomasi keypoints)
// ---------------------------------------------------------------------------
constexpr double PI_D = 3.14159265358979323846;
constexpr int GRID = 4;                      // 4x4 spatial cells
constexpr int NBINS = 8;                     // 8 orientation bins per cell
constexpr int DIM = GRID * GRID * NBINS;     // 128
constexpr int ORI_BINS = 36;                 // bins for the dominant-orientation histogram

// Grayscale in [0,1]; RGB uses luminance
void to_gray(const uint8_t* img, int h, int w, int ch, std::vector<double>& g) {
    g.resize(static_cast<size_t>(h) * w);
    for (size_t i = 0; i < g.size(); ++i) {
        g[i] = (ch == 1)
            ? img[i] / 255.0
            : (0.299 * img[i * 3] + 0.587 * img[i * 3 + 1] + 0.114 * img[i * 3 + 2]) / 255.0;
    }
}

// Gradient magnitude and angle ([0, 2pi)) per pixel (central differences)
void gradient_maps(const std::vector<double>& g, int h, int w,
                   std::vector<double>& mag, std::vector<double>& ang) {
    mag.resize(g.size());
    ang.resize(g.size());
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            double dx = 0.5 * (g[static_cast<size_t>(r) * w + clamp_idx(c + 1, w)] -
                               g[static_cast<size_t>(r) * w + clamp_idx(c - 1, w)]);
            double dy = 0.5 * (g[static_cast<size_t>(clamp_idx(r + 1, h)) * w + c] -
                               g[static_cast<size_t>(clamp_idx(r - 1, h)) * w + c]);
            size_t i = static_cast<size_t>(r) * w + c;
            mag[i] = std::sqrt(dx * dx + dy * dy);
            double a = std::atan2(dy, dx);
            ang[i] = (a < 0) ? a + 2.0 * PI_D : a;
        }
    }
}

// Step 1 - orientation: Gaussian-weighted histogram of gradient angles around the
// keypoint; the strongest bin is the keypoint's dominant orientation.
double dominant_orientation(const std::vector<double>& mag, const std::vector<double>& ang,
                            int h, int w, int x, int y, double sigma) {
    double s = 1.5 * sigma;
    int rad = static_cast<int>(std::round(3.0 * s));
    double hist[ORI_BINS] = {0};

    for (int dy = -rad; dy <= rad; ++dy) {
        int r = y + dy;
        if (r < 1 || r >= h - 1) continue;
        for (int dx = -rad; dx <= rad; ++dx) {
            int c = x + dx;
            if (c < 1 || c >= w - 1) continue;
            size_t i = static_cast<size_t>(r) * w + c;
            int bin = static_cast<int>(ang[i] * ORI_BINS / (2.0 * PI_D)) % ORI_BINS;
            hist[bin] += std::exp(-(dx * dx + dy * dy) / (2.0 * s * s)) * mag[i];
        }
    }

    int best = 0;
    for (int i = 1; i < ORI_BINS; ++i) if (hist[i] > hist[best]) best = i;
    return (best + 0.5) * 2.0 * PI_D / ORI_BINS;
}

// Step 2 - descriptor: rotate the patch to the dominant orientation, split it into
// 4x4 cells, build an 8-bin gradient-orientation histogram per cell (128 values).
void sift_descriptor(const std::vector<double>& mag, const std::vector<double>& ang,
                     int h, int w, int x, int y, double sigma, double theta, float* out) {
    double cell = 3.0 * sigma;                                    // cell width in pixels
    int rad = static_cast<int>(std::round(cell * std::sqrt(2.0) * (GRID + 1) * 0.5));
    double ct = std::cos(theta), st = std::sin(theta);
    double hist[GRID][GRID][NBINS] = {};

    for (int dy = -rad; dy <= rad; ++dy) {
        int r = y + dy;
        if (r < 1 || r >= h - 1) continue;
        for (int dx = -rad; dx <= rad; ++dx) {
            int c = x + dx;
            if (c < 1 || c >= w - 1) continue;

            // sample position in the rotated frame, in cell units (centered on keypoint)
            double u = ( ct * dx + st * dy) / cell;               // column direction
            double v = (-st * dx + ct * dy) / cell;               // row direction
            double cb = u + GRID / 2.0 - 0.5;
            double rb = v + GRID / 2.0 - 0.5;
            if (rb <= -1.0 || rb >= GRID || cb <= -1.0 || cb >= GRID) continue;

            size_t i = static_cast<size_t>(r) * w + c;
            double a = ang[i] - theta;                            // angle relative to keypoint
            if (a < 0) a += 2.0 * PI_D;
            double ob = a * NBINS / (2.0 * PI_D);

            // weight = gradient magnitude * Gaussian centered on the keypoint
            double wgt = mag[i] * std::exp(-(u * u + v * v) / (0.5 * GRID * GRID));

            // trilinear split between the neighbouring row / column / orientation bins
            int r0 = static_cast<int>(std::floor(rb));
            int c0 = static_cast<int>(std::floor(cb));
            int o0 = static_cast<int>(std::floor(ob));
            double fr = rb - r0, fc = cb - c0, fo = ob - o0;

            for (int i2 = 0; i2 < 2; ++i2) {
                int rr = r0 + i2;
                if (rr < 0 || rr >= GRID) continue;
                for (int j2 = 0; j2 < 2; ++j2) {
                    int cc = c0 + j2;
                    if (cc < 0 || cc >= GRID) continue;
                    for (int k2 = 0; k2 < 2; ++k2) {
                        hist[rr][cc][(o0 + k2) % NBINS] += wgt
                            * (i2 ? fr : 1.0 - fr) * (j2 ? fc : 1.0 - fc) * (k2 ? fo : 1.0 - fo);
                    }
                }
            }
        }
    }

    // Step 3 - normalize, clip large values at 0.2, normalize again (lighting robustness)
    double* d = &hist[0][0][0];
    for (int pass = 0; pass < 2; ++pass) {
        double norm = 0.0;
        for (int i = 0; i < DIM; ++i) norm += d[i] * d[i];
        norm = std::sqrt(norm);
        if (norm < 1e-12) norm = 1.0;
        for (int i = 0; i < DIM; ++i) {
            d[i] /= norm;
            if (pass == 0 && d[i] > 0.2) d[i] = 0.2;
        }
    }
    for (int i = 0; i < DIM; ++i) out[i] = static_cast<float>(d[i]);
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
 * 128-D SIFT descriptors at given keypoints (e.g. from detect_corners).
 * Pipeline: grayscale -> Gaussian blur (sigma) -> gradients -> per keypoint:
 *           dominant orientation -> rotated 4x4x8 histogram -> normalize/clip.
 * Harris is single-scale, so all keypoints use the same descriptor scale `sigma`.
 *
 * @param out_desc    n_kp * 128 floats
 * @param out_orient  n_kp doubles (orientation in radians) or nullptr
 * @return            n_kp (0 on invalid input)
 */
EXPORT int compute_sift_descriptors(
    const uint8_t* img_data, int height, int width, int channels,
    const int* kp_x, const int* kp_y, int n_kp, double sigma,
    float* out_desc, double* out_orient, double* out_time_ms
) {
    if (!img_data || !kp_x || !kp_y || !out_desc || height < 3 || width < 3 ||
        (channels != 1 && channels != 3) || n_kp <= 0) {
        if (out_time_ms) *out_time_ms = 0.0;
        return 0;
    }
    if (sigma < 0.5) sigma = 0.5;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<double> gray, smooth(static_cast<size_t>(height) * width), temp(smooth.size());
    to_gray(img_data, height, width, channels, gray);
    int ksize = 2 * static_cast<int>(std::ceil(3.0 * sigma)) + 1;
    gaussian_blur_2d(gray, smooth, temp, height, width, make_gaussian_kernel_1d(ksize, sigma));

    std::vector<double> mag, ang;
    gradient_maps(smooth, height, width, mag, ang);

    for (int i = 0; i < n_kp; ++i) {
        int x = clamp_idx(kp_x[i], width);
        int y = clamp_idx(kp_y[i], height);
        double theta = dominant_orientation(mag, ang, height, width, x, y, sigma);
        sift_descriptor(mag, ang, height, width, x, y, sigma, theta,
                        out_desc + static_cast<size_t>(i) * DIM);
        if (out_orient) out_orient[i] = theta;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    if (out_time_ms) *out_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return n_kp;
}

/**
 * Match descriptor set 1 -> set 2 (nearest neighbour + Lowe ratio test).
 *   method 0: SSD  = sum (a - b)^2            (smaller is better)
 *   method 1: NCC  = zero-mean normalized dot (larger is better, in [-1, 1])
 *
 * @param ratio      best/second-best ratio threshold (e.g. 0.8); >= 1 disables the test
 * @param out_i/j    matched indices into set 1 / set 2 (buffers of size n1)
 * @param out_score  SSD or NCC value of each match
 * @return           number of matches
 */
EXPORT int match_descriptors(
    const float* desc1, int n1, const float* desc2, int n2,
    int method, double ratio,
    int* out_i, int* out_j, double* out_score, double* out_time_ms
) {
    if (!desc1 || !desc2 || n1 <= 0 || n2 <= 0) {
        if (out_time_ms) *out_time_ms = 0.0;
        return 0;
    }
    auto t_start = std::chrono::high_resolution_clock::now();

    // NCC = dot product after removing the mean and scaling to unit length
    auto prepare = [&](const float* d, int n) {
        std::vector<float> o(d, d + static_cast<size_t>(n) * DIM);
        if (method == 1) {
            for (int i = 0; i < n; ++i) {
                float* p = &o[static_cast<size_t>(i) * DIM];
                double mean = 0.0, norm = 0.0;
                for (int k = 0; k < DIM; ++k) mean += p[k];
                mean /= DIM;
                for (int k = 0; k < DIM; ++k) { p[k] -= static_cast<float>(mean); norm += double(p[k]) * p[k]; }
                norm = std::sqrt(norm);
                if (norm < 1e-12) norm = 1.0;
                for (int k = 0; k < DIM; ++k) p[k] = static_cast<float>(p[k] / norm);
            }
        }
        return o;
    };
    std::vector<float> A = prepare(desc1, n1), B = prepare(desc2, n2);

    // SSD ratio test is on squared distances; NCC is turned into a distance (1 - NCC)
    double r = (method == 0) ? ratio * ratio : ratio;
    const double INF = std::numeric_limits<double>::infinity();
    int count = 0;

    for (int i = 0; i < n1; ++i) {
        const float* a = &A[static_cast<size_t>(i) * DIM];
        double best = INF, second = INF;
        int best_j = -1;

        for (int j = 0; j < n2; ++j) {
            const float* b = &B[static_cast<size_t>(j) * DIM];
            double acc = 0.0;
            if (method == 0) {
                for (int k = 0; k < DIM; ++k) { double d = double(a[k]) - b[k]; acc += d * d; }
            } else {
                for (int k = 0; k < DIM; ++k) acc += double(a[k]) * b[k];
                acc = 1.0 - acc;
            }
            if (acc < best)        { second = best; best = acc; best_j = j; }
            else if (acc < second) { second = acc; }
        }

        if (best_j < 0 || best > r * second) continue;   // ambiguous match -> reject
        out_i[count] = i;
        out_j[count] = best_j;
        out_score[count] = (method == 0) ? best : 1.0 - best;
        ++count;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    if (out_time_ms) *out_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return count;
}
