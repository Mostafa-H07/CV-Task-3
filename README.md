# Computer Vision Task 3

A Streamlit application for demonstrating an image-feature pipeline: corner detection, scale-invariant keypoint extraction, and descriptor matching. The Python interface handles image loading and visualization, while `engine.cpp` implements the image-processing operations and exposes them to Python through `ctypes`.

## Pipeline overview

The application is divided into three stages:

1. **Harris / Shi-Tomasi corner detection**
2. **Scale-invariant SIFT keypoint description**
3. **Descriptor matching**

An uploaded image may be kept in RGB mode or converted to grayscale. The app caches results by the image token and processing parameters, so changing a setting or replacing a file starts a new run.

### Stage 1: Harris / Shi-Tomasi corners

This stage computes local structure from the image gradients.

- A Gaussian window smooths the derivatives used to build the structure / second-moment matrix.
- **Harris** combines the eigenvalues into \(R = \det(M) - k\,\mathrm{trace}(M)^2\).
- **Shi-Tomasi** uses the smaller eigenvalue as the corner response.
- Pixels below a configurable percentage of the maximum response are removed.
- Non-maximum suppression retains strong, isolated corners.
- The strongest corners are displayed over the image.

The response map is shown separately with a log-scaled visualization. This stage operates independently of SIFT and does not feed its corners into the SIFT stage.

### Stage 2: Scale-invariant SIFT

This stage creates scale-space keypoints instead of using Stage 1 corners.

1. The native engine converts the input to grayscale when necessary.
2. Optional upsampling doubles the image size to improve detection of small features.
3. A Gaussian pyramid is built at several scales per octave.
4. Adjacent Gaussian layers are subtracted to create Difference-of-Gaussians (DoG) layers.
5. The engine searches every DoG point and its local scale-space neighborhood for 3-D extrema.
6. A 3-D quadratic refinement estimates subpixel position, subscale location, and contrast.
7. Low-contrast candidates and edge-like responses are rejected.
8. Duplicate candidates are removed, and keypoints are sorted deterministically.
9. A gradient orientation histogram around each keypoint determines dominant orientations.
10. For each orientation, a 4 × 4 grid of eight-bin gradient histograms produces a 128-value descriptor.
11. Each descriptor is normalized and clipped to improve robustness to lighting and contrast changes.

The base blur, scales per octave, contrast threshold, edge threshold, upsampling, and maximum keypoint count are configurable. The visualization draws a scale-dependent circle and a line showing each keypoint's dominant orientation.

The scale histogram counts keypoints in bands such as \([\sigma_0,\ 2\sigma_0)\), \([2\sigma_0,\ 4\sigma_0)\), and larger powers of two.

### Stage 3: Descriptor matching

The app computes SIFT descriptors for both selected images and applies one of two matching methods:

- **SSD:** sums squared differences between 128-value descriptors; lower is better.
- **NCC:** compares mean-centered, normalized descriptors using a dot product; higher is better.

For every descriptor in Image A, the engine searches Image B and records the closest and second-closest matches. Lowe's ratio test rejects ambiguous matches when the best match is not sufficiently better than the second-best match. A ratio threshold of `1.0` disables the test.

Matched keypoints are displayed side by side, with lines connecting each pair. The app also reports match counts, median relative scale, and per-method execution time.

## Interaction and caching

- **Pin current run** saves a snapshot of the current Stage 1, Stage 2, or Matching result.
- **Clear pinned** removes saved snapshots.
- **Clear log** resets the Stage 2 computation log.
- Latest runs remain available in comparison tables while their input images are still uploaded.
- The Streamlit cache stores detector and SIFT results for the same image and parameter set.
- Stage 2 signatures include both the image token and grayscale state, so replacing a file or toggling grayscale detection starts a new SIFT run.

## Quickstart

1. Install Python 3.10 or newer and open a terminal in this project directory.

2. Install the dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Build `engine.dll` with a C++ compiler on Windows:

   ```powershell
   g++ -O3 -shared -static -o engine.dll engine.cpp
   ```

   For Linux or macOS, build the matching shared-library name instead:

   ```bash
   g++ -O3 -shared -fPIC -o libengine.so engine.cpp
   # macOS:
   g++ -O3 -shared -fPIC -o libengine.dylib engine.cpp
   ```

4. Start the application:

   ```powershell
   streamlit run app.py
   ```

5. Upload one or more images, then use the sidebar to move through **Harris / Shi-Tomasi**, **SIFT**, and **Matching**.
