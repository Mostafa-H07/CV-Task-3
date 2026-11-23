# Computer Vision Task 3

A Streamlit pipeline for detecting and matching image features. It combines Harris / Shi-Tomasi corner detection, scale-invariant SIFT-like keypoint extraction using Difference-of-Gaussians (DoG), and SSD / NCC descriptor matching. A native C++ engine provides the image-processing operations through a small Python `ctypes` interface.

## What the program does

- **Stage 1 - Harris / Shi-Tomasi:** detects corners, applies non-maximum suppression, and displays a log-scaled response map.
- **Stage 2 - SIFT:** builds a Gaussian / DoG scale-space, detects scale-space extrema, assigns orientations, and generates 128-dimensional descriptors.
- **Stage 3 - Matching:** compares descriptors between two uploaded images with SSD or NCC and Lowe's ratio test.

The interface supports multiple uploads, grayscale conversion, adjustable detector parameters, run logging, pinned snapshots, scale-band statistics, and visual keypoint / match overlays.

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
