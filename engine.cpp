#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
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
// Scale-invariant SIFT
//
//   gray image -> Gaussian pyramid (octaves x layers) -> DoG pyramid
//              -> 3-D extrema (x, y, scale) + sub-pixel/sub-scale refinement
//              -> contrast + edge rejection  => keypoints (x, y, sigma)
//              -> orientation(s) from the Gaussian level of the keypoint's scale
//              -> 4x4x8 descriptor sampled with a window proportional to sigma
//
// Coordinate / scale conventions
//   * Octave o has pixel size  step_o = step_0 * 2^o  (in ORIGINAL image pixels),
//     where step_0 = 0.5 if the base image is up-sampled 2x, else 1.
//     A point (x_o, y_o) of octave o maps to the original image as
//     (x_o * step_o, y_o * step_o).  Up-sampling and decimation both keep pixel
//     (0,0) fixed, so this single factor is exact.
//   * Layer i of an octave is blurred by  sigma_i = sigma0 * 2^(i / S)  (octave
//     pixels).  The absolute scale of a keypoint is  sigma_i * step_o, expressed in
//     ORIGINAL image pixels.  Everything below the detector (orientation window,
//     descriptor cell width) works in octave pixels with sigma_oct = sigma / step_o.
// ---------------------------------------------------------------------------
constexpr double PI_D = 3.14159265358979323846;
constexpr int GRID = 4;                      // 4x4 spatial cells
constexpr int NBINS = 8;                     // 8 orientation bins per cell
constexpr int DIM = GRID * GRID * NBINS;     // 128
constexpr int ORI_BINS = 36;                 // bins for the dominant-orientation histogram
constexpr int SIFT_BORDER = 5;               // DoG extrema must be this far from the border
constexpr int MAX_REFINE_ITERS = 5;

struct SiftParams {
    int    n_layers     = 3;      // S: scale samples per octave (layers 1..S hold keypoints)
    double sigma0       = 1.6;    // blur of layer 0 of every octave (octave pixels)
    double contrast_thr = 0.04;   // |D(x_hat)| * S must exceed this (image range [0,1])
    double edge_thr     = 10.0;   // r in tr(H)^2/det(H) < (r+1)^2/r
    bool   upsample     = true;   // double the image before building octave 0
};

struct Plane {
    int h = 0, w = 0;
    std::vector<double> d;
    double at(int r, int c) const { return d[static_cast<size_t>(r) * w + c]; }
};

struct Octave {
    double step = 1.0;                       // octave pixel size in original pixels
    std::vector<Plane> gauss;                // S + 3 Gaussian levels
    std::vector<Plane> dog;                  // S + 2 DoG levels
    std::vector<Plane> mag, ang;             // gradients of gauss[i], filled lazily
    std::vector<char> grad_ready;
};

struct Pyramid {
    SiftParams p;
    std::vector<Octave> oct;
};

// Scale-aware keypoint. x, y, sigma are in ORIGINAL image pixels.
struct SiftKeypoint {
    double x = 0.0;
    double y = 0.0;
    double sigma = 0.0;          // characteristic scale
    double orientation = 0.0;    // radians in [0, 2*pi)
    double response = 0.0;       // |DoG| at the refined extremum
    int octave = 0;              // pyramid location, needed to fetch the right Gaussian level
    int layer = 1;
};

SiftParams make_params(int n_layers, double sigma0, double contrast_thr, double edge_thr, int upsample) {
    SiftParams p;
    p.n_layers = (n_layers < 1) ? 3 : std::min(n_layers, 8);
    p.sigma0 = (sigma0 <= 0.0) ? 1.6 : std::max(sigma0, 0.5);
    p.contrast_thr = (contrast_thr <= 0.0) ? 0.04 : contrast_thr;
    p.edge_thr = (edge_thr <= 1.0) ? 10.0 : edge_thr;
    p.upsample = (upsample != 0);
    return p;
}

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

// ----------------------------- Gaussian pyramid -----------------------------

Plane blur_plane(const Plane& src, double sigma) {
    Plane dst;
    dst.h = src.h;
    dst.w = src.w;
    dst.d.resize(src.d.size());
    std::vector<double> temp(src.d.size());
    int ksize = 2 * static_cast<int>(std::ceil(3.0 * sigma)) + 1;
    gaussian_blur_2d(src.d, dst.d, temp, src.h, src.w, make_gaussian_kernel_1d(ksize, sigma));
    return dst;
}

// Bilinear 2x up-sampling; output pixel (2r, 2c) == input pixel (r, c)
Plane upsample2x(const Plane& s) {
    Plane o;
    o.h = 2 * s.h;
    o.w = 2 * s.w;
    o.d.resize(static_cast<size_t>(o.h) * o.w);
    for (int r = 0; r < o.h; ++r) {
        double sr = 0.5 * r;
        int r0 = static_cast<int>(sr);
        int r1 = std::min(r0 + 1, s.h - 1);
        double fr = sr - r0;
        for (int c = 0; c < o.w; ++c) {
            double sc = 0.5 * c;
            int c0 = static_cast<int>(sc);
            int c1 = std::min(c0 + 1, s.w - 1);
            double fc = sc - c0;
            double top = s.at(r0, c0) * (1.0 - fc) + s.at(r0, c1) * fc;
            double bot = s.at(r1, c0) * (1.0 - fc) + s.at(r1, c1) * fc;
            o.d[static_cast<size_t>(r) * o.w + c] = top * (1.0 - fr) + bot * fr;
        }
    }
    return o;
}

// Keep every second pixel: output (r, c) == input (2r, 2c)
Plane downsample2x(const Plane& s) {
    Plane o;
    o.h = s.h / 2;
    o.w = s.w / 2;
    o.d.resize(static_cast<size_t>(o.h) * o.w);
    for (int r = 0; r < o.h; ++r)
        for (int c = 0; c < o.w; ++c)
            o.d[static_cast<size_t>(r) * o.w + c] = s.at(2 * r, 2 * c);
    return o;
}

// Returns an empty pyramid if the image is too small to hold a DoG extremum.
Pyramid build_pyramid(const std::vector<double>& gray, int h, int w, const SiftParams& p) {
    Pyramid pyr;
    pyr.p = p;

    Plane base;
    base.h = h;
    base.w = w;
    base.d = gray;
    double assumed_blur = 0.5;   // nominal blur of the camera image
    double step0 = 1.0;
    if (p.upsample) {
        base = upsample2x(base);
        assumed_blur = 1.0;      // 0.5 doubled
        step0 = 0.5;
    }
    int min_dim = std::min(base.h, base.w);
    if (min_dim < 2 * SIFT_BORDER + 2) return pyr;

    // smallest octave keeps at least ~16 pixels on its short side
    int n_oct = static_cast<int>(std::floor(std::log2(min_dim / 16.0))) + 1;
    if (n_oct < 1) n_oct = 1;

    const int S = p.n_layers;
    const double k = std::pow(2.0, 1.0 / S);
    const double sd0 = std::sqrt(std::max(p.sigma0 * p.sigma0 - assumed_blur * assumed_blur, 0.01));
    Plane cur = blur_plane(base, sd0);              // sigma = sigma0 in octave-0 pixels

    for (int o = 0; o < n_oct; ++o) {
        Octave oc;
        oc.step = step0 * std::ldexp(1.0, o);
        oc.gauss.reserve(S + 3);
        oc.gauss.push_back(cur);

        // incremental blur: sigma_i^2 = sigma_{i-1}^2 + inc^2  with sigma_i = sigma0 * k^i
        double sig_prev = p.sigma0;
        for (int i = 1; i < S + 3; ++i) {
            double inc = sig_prev * std::sqrt(k * k - 1.0);
            Plane next = blur_plane(oc.gauss.back(), inc);
            oc.gauss.push_back(std::move(next));
            sig_prev *= k;
        }

        // DoG_i = G_{i+1} - G_i
        for (int i = 0; i < S + 2; ++i) {
            Plane d;
            d.h = oc.gauss[i].h;
            d.w = oc.gauss[i].w;
            d.d.resize(oc.gauss[i].d.size());
            for (size_t j = 0; j < d.d.size(); ++j) d.d[j] = oc.gauss[i + 1].d[j] - oc.gauss[i].d[j];
            oc.dog.push_back(std::move(d));
        }

        // Layer S has sigma = 2*sigma0; decimating it gives sigma0 in the next octave's pixels
        if (o + 1 < n_oct) cur = downsample2x(oc.gauss[S]);
        pyr.oct.push_back(std::move(oc));
    }
    return pyr;
}

void ensure_gradients(Octave& oc, int layer) {
    if (oc.mag.empty()) {
        oc.mag.resize(oc.gauss.size());
        oc.ang.resize(oc.gauss.size());
        oc.grad_ready.assign(oc.gauss.size(), 0);
    }
    if (!oc.grad_ready[layer]) {
        const Plane& g = oc.gauss[layer];
        oc.mag[layer].h = oc.ang[layer].h = g.h;
        oc.mag[layer].w = oc.ang[layer].w = g.w;
        gradient_maps(g.d, g.h, g.w, oc.mag[layer].d, oc.ang[layer].d);
        oc.grad_ready[layer] = 1;
    }
}

// ----------------------------- DoG extrema detection -----------------------------

// Fit a 3-D quadratic to the DoG around (l, r, c), iterating while the offset exceeds 0.5.
// On success l/r/c hold the final integer location, off_* the sub-sample offset and
// `contrast` the interpolated DoG value. Also applies the edge (Hessian ratio) test.
bool refine_extremum(const Octave& oc, const SiftParams& p, int& l, int& r, int& c,
                     double& off_s, double& off_r, double& off_c, double& contrast) {
    const int S = p.n_layers;
    const int h = oc.dog[0].h, w = oc.dog[0].w;

    for (int it = 0; it < MAX_REFINE_ITERS; ++it) {
        if (l < 1 || l > S || r < SIFT_BORDER || r >= h - SIFT_BORDER ||
            c < SIFT_BORDER || c >= w - SIFT_BORDER) return false;

        const Plane& prv = oc.dog[l - 1];
        const Plane& cur = oc.dog[l];
        const Plane& nxt = oc.dog[l + 1];

        double gx = 0.5 * (cur.at(r, c + 1) - cur.at(r, c - 1));
        double gy = 0.5 * (cur.at(r + 1, c) - cur.at(r - 1, c));
        double gs = 0.5 * (nxt.at(r, c) - prv.at(r, c));

        double v2 = 2.0 * cur.at(r, c);
        double dxx = cur.at(r, c + 1) + cur.at(r, c - 1) - v2;
        double dyy = cur.at(r + 1, c) + cur.at(r - 1, c) - v2;
        double dss = nxt.at(r, c) + prv.at(r, c) - v2;
        double dxy = 0.25 * (cur.at(r + 1, c + 1) - cur.at(r + 1, c - 1) -
                             cur.at(r - 1, c + 1) + cur.at(r - 1, c - 1));
        double dxs = 0.25 * (nxt.at(r, c + 1) - nxt.at(r, c - 1) -
                             prv.at(r, c + 1) + prv.at(r, c - 1));
        double dys = 0.25 * (nxt.at(r + 1, c) - nxt.at(r - 1, c) -
                             prv.at(r + 1, c) + prv.at(r - 1, c));

        // Solve H * X = -g (X = [dx, dy, ds]) with Cramer's rule
        double b0 = -gx, b1 = -gy, b2 = -gs;
        double det = dxx * (dyy * dss - dys * dys) - dxy * (dxy * dss - dys * dxs) +
                     dxs * (dxy * dys - dyy * dxs);
        if (std::fabs(det) < 1e-18) return false;
        double det0 = b0 * (dyy * dss - dys * dys) - dxy * (b1 * dss - dys * b2) +
                      dxs * (b1 * dys - dyy * b2);
        double det1 = dxx * (b1 * dss - dys * b2) - b0 * (dxy * dss - dys * dxs) +
                      dxs * (dxy * b2 - b1 * dxs);
        double det2 = dxx * (dyy * b2 - b1 * dys) - dxy * (dxy * b2 - b1 * dxs) +
                      b0 * (dxy * dys - dyy * dxs);
        off_c = det0 / det;
        off_r = det1 / det;
        off_s = det2 / det;

        if (std::fabs(off_c) < 0.5 && std::fabs(off_r) < 0.5 && std::fabs(off_s) < 0.5) {
            // converged: low-contrast test on the interpolated extremum value
            contrast = cur.at(r, c) + 0.5 * (gx * off_c + gy * off_r + gs * off_s);
            if (std::fabs(contrast) * S < p.contrast_thr) return false;

            // edge rejection from the 2x2 spatial Hessian: tr^2 / det < (r+1)^2 / r
            double tr = dxx + dyy;
            double det2d = dxx * dyy - dxy * dxy;
            double rr = p.edge_thr;
            if (det2d <= 0.0 || tr * tr * rr >= (rr + 1.0) * (rr + 1.0) * det2d) return false;
            return true;
        }

        // jump to the neighbouring sample the quadratic points at (bounded to avoid huge jumps)
        if (std::fabs(off_c) > 1e4 || std::fabs(off_r) > 1e4 || std::fabs(off_s) > 1e4) return false;
        c += static_cast<int>(std::lround(off_c));
        r += static_cast<int>(std::lround(off_r));
        l += static_cast<int>(std::lround(off_s));
    }
    return false;   // did not converge
}

// 26-neighbour test in (x, y, scale)
bool is_scale_space_extremum(const Octave& oc, int l, int r, int c) {
    const double v = oc.dog[l].at(r, c);
    const bool is_max = v > 0.0;
    for (int ds = -1; ds <= 1; ++ds) {
        const Plane& pl = oc.dog[l + ds];
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (ds == 0 && dr == 0 && dc == 0) continue;
                double nb = pl.at(r + dr, c + dc);
                if (is_max ? (nb > v) : (nb < v)) return false;
            }
        }
    }
    return true;
}

// Detect DoG keypoints (position + characteristic scale; orientation is assigned later).
std::vector<SiftKeypoint> detect_dog_extrema(const Pyramid& pyr) {
    std::vector<SiftKeypoint> kps;
    const SiftParams& p = pyr.p;
    const int S = p.n_layers;
    const double prelim = 0.5 * p.contrast_thr / S;     // cheap pre-filter before refinement

    for (size_t o = 0; o < pyr.oct.size(); ++o) {
        const Octave& oc = pyr.oct[o];
        const int h = oc.dog[0].h, w = oc.dog[0].w;
        size_t first = kps.size();

        for (int l = 1; l <= S; ++l) {
            for (int r = SIFT_BORDER; r < h - SIFT_BORDER; ++r) {
                for (int c = SIFT_BORDER; c < w - SIFT_BORDER; ++c) {
                    double v = oc.dog[l].at(r, c);
                    if (std::fabs(v) <= prelim) continue;
                    if (!is_scale_space_extremum(oc, l, r, c)) continue;

                    int ll = l, rr = r, cc = c;
                    double os = 0, orow = 0, ocol = 0, contrast = 0;
                    if (!refine_extremum(oc, p, ll, rr, cc, os, orow, ocol, contrast)) continue;

                    SiftKeypoint kp;
                    double sigma_oct = p.sigma0 * std::pow(2.0, (ll + os) / S);
                    kp.x = (cc + ocol) * oc.step;               // octave -> original image
                    kp.y = (rr + orow) * oc.step;
                    kp.sigma = sigma_oct * oc.step;
                    kp.response = std::fabs(contrast);
                    kp.octave = static_cast<int>(o);
                    kp.layer = ll;
                    kps.push_back(kp);
                }
            }
        }

        // different candidates can converge to the same refined extremum: drop exact repeats
        std::sort(kps.begin() + first, kps.end(), [](const SiftKeypoint& a, const SiftKeypoint& b) {
            if (a.layer != b.layer) return a.layer < b.layer;
            if (a.y != b.y) return a.y < b.y;
            if (a.x != b.x) return a.x < b.x;
            return a.sigma < b.sigma;
        });
        auto same = [](const SiftKeypoint& a, const SiftKeypoint& b) {
            return a.layer == b.layer && a.x == b.x && a.y == b.y && a.sigma == b.sigma;
        };
        kps.erase(std::unique(kps.begin() + first, kps.end(), same), kps.end());
    }
    return kps;
}

// ----------------------------- Orientation -----------------------------

// Step 1 - orientation. `x`, `y`, `sigma` are in the pixels of the gradient maps (octave
// pixels) and `sigma` is THIS keypoint's scale: the Gaussian window has sigma_w = 1.5*sigma
// and radius 3*sigma_w, so the neighbourhood grows with the keypoint scale. Returns every
// histogram peak >= 80% of the maximum (strongest first), each refined by a parabola.
void dominant_orientations(const std::vector<double>& mag, const std::vector<double>& ang,
                           int h, int w, double x, double y, double sigma,
                           std::vector<double>& out) {
    out.clear();
    const double s = 1.5 * sigma;
    const int rad = std::max(1, static_cast<int>(std::lround(3.0 * s)));
    const int cx = static_cast<int>(std::lround(x)), cy = static_cast<int>(std::lround(y));
    double hist[ORI_BINS] = {0};

    for (int r = cy - rad; r <= cy + rad; ++r) {
        if (r < 1 || r >= h - 1) continue;
        double dy = r - y;
        for (int c = cx - rad; c <= cx + rad; ++c) {
            if (c < 1 || c >= w - 1) continue;
            double dx = c - x;
            double d2 = dx * dx + dy * dy;
            if (d2 > static_cast<double>(rad) * rad) continue;          // circular window
            size_t i = static_cast<size_t>(r) * w + c;
            int bin = static_cast<int>(ang[i] * ORI_BINS / (2.0 * PI_D)) % ORI_BINS;
            hist[bin] += std::exp(-d2 / (2.0 * s * s)) * mag[i];
        }
    }

    // circular smoothing with [1 4 6 4 1] / 16
    double sm[ORI_BINS];
    for (int i = 0; i < ORI_BINS; ++i) {
        sm[i] = (hist[(i + ORI_BINS - 2) % ORI_BINS] + hist[(i + 2) % ORI_BINS]
                 + 4.0 * (hist[(i + ORI_BINS - 1) % ORI_BINS] + hist[(i + 1) % ORI_BINS])
                 + 6.0 * hist[i]) / 16.0;
    }

    int best = 0;
    for (int i = 1; i < ORI_BINS; ++i) if (sm[i] > sm[best]) best = i;
    const double bin_w = 2.0 * PI_D / ORI_BINS;
    if (sm[best] <= 1e-12) { out.push_back(0.0); return; }      // flat patch

    std::vector<std::pair<double, double>> peaks;               // (strength, angle)
    for (int i = 0; i < ORI_BINS; ++i) {
        double l = sm[(i + ORI_BINS - 1) % ORI_BINS], c = sm[i], rg = sm[(i + 1) % ORI_BINS];
        if (c > l && c >= rg && c >= 0.8 * sm[best]) {
            double denom = l - 2.0 * c + rg;
            double d = (denom < -1e-18) ? 0.5 * (l - rg) / denom : 0.0;
            d = std::min(0.5, std::max(-0.5, d));
            double a = (i + 0.5 + d) * bin_w;                   // bin i spans [i, i+1) * bin_w
            a = std::fmod(a + 2.0 * PI_D, 2.0 * PI_D);
            peaks.emplace_back(c, a);
        }
    }
    if (peaks.empty()) peaks.emplace_back(sm[best], (best + 0.5) * bin_w);
    std::sort(peaks.begin(), peaks.end(),
              [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                  return a.first > b.first;
              });
    for (const auto& pk : peaks) out.push_back(pk.second);
}

// ----------------------------- Descriptor -----------------------------

// Step 2 - descriptor. (x, y, sigma) are in the pixels of the gradient maps; `sigma` is the
// keypoint's own scale, so the cell width (3*sigma) and the sampling radius scale with it.
// The patch is rotated to `theta`, split into 4x4 cells and an 8-bin gradient histogram is
// accumulated per cell with trilinear interpolation (128 values).
void sift_descriptor(const std::vector<double>& mag, const std::vector<double>& ang,
                     int h, int w, double x, double y, double sigma, double theta, float* out) {
    const double cell = 3.0 * sigma;                              // cell width in pixels
    const int rad = static_cast<int>(std::lround(cell * std::sqrt(2.0) * (GRID + 1) * 0.5));
    const double ct = std::cos(theta), st = std::sin(theta);
    const int cx = static_cast<int>(std::lround(x)), cy = static_cast<int>(std::lround(y));
    double hist[GRID][GRID][NBINS] = {};

    for (int r = cy - rad; r <= cy + rad; ++r) {
        if (r < 1 || r >= h - 1) continue;
        double dy = r - y;                                        // sub-pixel accurate offsets
        for (int c = cx - rad; c <= cx + rad; ++c) {
            if (c < 1 || c >= w - 1) continue;
            double dx = c - x;

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
            if (ob >= NBINS) ob -= NBINS;

            // weight = gradient magnitude * Gaussian (sigma = GRID/2 cells) centered on keypoint
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

// Describe one keypoint using ITS OWN octave / layer / sigma / orientation.
void describe_keypoint(Pyramid& pyr, const SiftKeypoint& kp, float* out) {
    Octave& oc = pyr.oct[kp.octave];
    ensure_gradients(oc, kp.layer);
    const Plane& m = oc.mag[kp.layer];
    const Plane& a = oc.ang[kp.layer];
    sift_descriptor(m.d, a.d, m.h, m.w, kp.x / oc.step, kp.y / oc.step,
                    kp.sigma / oc.step, kp.orientation, out);
}

// Assign one or more orientations to every keypoint (a keypoint with several strong
// orientation peaks is duplicated, one copy per orientation, as in Lowe's SIFT).
std::vector<SiftKeypoint> assign_orientations(Pyramid& pyr, const std::vector<SiftKeypoint>& in) {
    std::vector<SiftKeypoint> out;
    std::vector<double> thetas;
    out.reserve(in.size());
    for (const SiftKeypoint& kp : in) {
        Octave& oc = pyr.oct[kp.octave];
        ensure_gradients(oc, kp.layer);
        const Plane& m = oc.mag[kp.layer];
        const Plane& a = oc.ang[kp.layer];
        dominant_orientations(m.d, a.d, m.h, m.w, kp.x / oc.step, kp.y / oc.step,
                              kp.sigma / oc.step, thetas);
        for (double t : thetas) {
            SiftKeypoint k2 = kp;
            k2.orientation = t;
            out.push_back(k2);
        }
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
 * Scale-invariant SIFT: DoG keypoint detection (+ optional descriptors) in one pass.
 * Pipeline: gray -> Gaussian/DoG pyramid -> 3-D extrema -> sub-pixel/scale refinement ->
 *           contrast + edge rejection -> orientation(s) -> 128-D descriptors.
 * The pyramid is built once and shared by detection, orientation and description.
 *
 * Keypoints are sorted by DoG response (strongest first) and truncated to max_kp.
 * A keypoint with several strong orientations appears once per orientation.
 *
 * @param n_layers      S, scale samples per octave (<= 0 -> 3)
 * @param sigma0        base blur of each octave (<= 0 -> 1.6)
 * @param contrast_thr  contrast threshold, image range [0,1] (<= 0 -> 0.04)
 * @param edge_thr      edge ratio r (<= 1 -> 10)
 * @param upsample      non-zero: double the image first (finds the smallest features)
 * @param max_kp        capacity of every output buffer
 * @param out_x/out_y   keypoint position in ORIGINAL image pixels   (nullable)
 * @param out_sigma     characteristic scale in ORIGINAL image pixels  (nullable)
 * @param out_orient    orientation in radians [0, 2pi)                 (nullable)
 * @param out_response  |DoG| at the refined extremum                   (nullable)
 * @param out_desc      max_kp * 128 floats; nullptr = detect only      (nullable)
 * @return              number of keypoints written
 */
EXPORT int detect_and_describe_sift(
    const uint8_t* img_data, int height, int width, int channels,
    int n_layers, double sigma0, double contrast_thr, double edge_thr, int upsample,
    int max_kp,
    double* out_x, double* out_y, double* out_sigma, double* out_orient, double* out_response,
    float* out_desc, double* out_time_ms
) {
    if (!img_data || height < 3 || width < 3 || (channels != 1 && channels != 3) || max_kp <= 0) {
        if (out_time_ms) *out_time_ms = 0.0;
        return 0;
    }
    auto t_start = std::chrono::high_resolution_clock::now();

    SiftParams p = make_params(n_layers, sigma0, contrast_thr, edge_thr, upsample);
    std::vector<double> gray;
    to_gray(img_data, height, width, channels, gray);
    Pyramid pyr = build_pyramid(gray, height, width, p);

    int count = 0;
    if (!pyr.oct.empty()) {
        std::vector<SiftKeypoint> kps = assign_orientations(pyr, detect_dog_extrema(pyr));
        std::stable_sort(kps.begin(), kps.end(), [](const SiftKeypoint& a, const SiftKeypoint& b) {
            return a.response > b.response;
        });
        count = std::min(static_cast<int>(kps.size()), max_kp);
        for (int i = 0; i < count; ++i) {
            const SiftKeypoint& kp = kps[i];
            if (out_x) out_x[i] = kp.x;
            if (out_y) out_y[i] = kp.y;
            if (out_sigma) out_sigma[i] = kp.sigma;
            if (out_orient) out_orient[i] = kp.orientation;
            if (out_response) out_response[i] = kp.response;
            if (out_desc) describe_keypoint(pyr, kp, out_desc + static_cast<size_t>(i) * DIM);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    if (out_time_ms) *out_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return count;
}

/**
 * DoG keypoint detection only (no descriptors). See detect_and_describe_sift.
 */
EXPORT int detect_sift_keypoints(
    const uint8_t* img_data, int height, int width, int channels,
    int n_layers, double sigma0, double contrast_thr, double edge_thr, int upsample,
    int max_kp,
    double* out_x, double* out_y, double* out_sigma, double* out_orient, double* out_response,
    double* out_time_ms
) {
    return detect_and_describe_sift(img_data, height, width, channels, n_layers, sigma0,
                                    contrast_thr, edge_thr, upsample, max_kp,
                                    out_x, out_y, out_sigma, out_orient, out_response,
                                    nullptr, out_time_ms);
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

        if (best_j < 0 || (ratio < 1.0 && best >= r * second)) continue;   // ambiguous match -> reject
        out_i[count] = i;
        out_j[count] = best_j;
        out_score[count] = (method == 0) ? best : 1.0 - best;
        ++count;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    if (out_time_ms) *out_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return count;
}