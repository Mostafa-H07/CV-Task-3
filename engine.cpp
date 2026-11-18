#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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