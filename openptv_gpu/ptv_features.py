import sys
import os
from glob import glob
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
import cv2


import cv2
from skimage.feature import blob_log
from skimage import exposure
from cupyx.scipy.signal import convolve2d
import cupyx.scipy.ndimage as ndi
import cupy as cp

DTYPE_f = np.float32
DTYPE_i = np.uint8
THREAD_SIZE = 4

data_dir = r"D:\Projects\TRL Files\2025-09-11 V3V\Vortex Ring Demo\Vortex Ring\RawData"
# data_dir = r"D:\Projects\OpenPIV\openpiv-python-cpu\openpiv_cpu\tutorials\test1"

data_dir = data_dir.replace("\\", "/") + "/"

# Path to images.
frame_list = glob(os.path.join(data_dir, "*.T*.D*.P*.H*.V3VL*.tif"))

def detect_blobs_bloblog(frame, min_sigma=1, max_sigma=8, num_sigma=10,
                         threshold=0.02, overlap=0.5, box_scale=3.0):
    """
    Detect blobs using skimage.feature.blob_log.
    Returns boxes (N,4) in [x1,y1,x2,y2], centers (N,2) as (y,x), sigmas (N,).
    
    Parameters
    ----------
    frame : numpy.ndarray or cupy.ndarray (2D)
    min_sigma, max_sigma, num_sigma, threshold, overlap : passed to blob_log
    box_scale : float
        box half-size = box_scale * sigma  -> box width = 2*box_scale*sigma
    """
    # ensure numpy array for skimage
    was_cupy = False
    if isinstance(frame, cp.ndarray):
        was_cupy = True
        img = cp.asnumpy(frame)
    else:
        img = np.asarray(frame)

    # blob_log expects float image (not necessarily normalized)
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    # blobs: (N, 3) -> (y, x, sigma)
    if blobs.size == 0:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    ys = blobs[:, 0]
    xs = blobs[:, 1]
    sigmas = blobs[:, 2]

    # build bounding boxes: use box_scale * sigma as half-size (in pixels)
    half = box_scale * sigmas
    x1 = xs - half
    y1 = ys - half
    x2 = xs + half
    y2 = ys + half

    # clip to image bounds
    H, W = img.shape
    x1 = np.clip(x1, 0, W - 1)
    x2 = np.clip(x2, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    y2 = np.clip(y2, 0, H - 1)

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    centers = np.stack([ys, xs], axis=1).astype(np.float32)
    sigmas = sigmas.astype(np.float32)

    return boxes, centers, sigmas

code_get_subpixel = """
extern "C" __global__
void cuda_get_subpixel(
    const float* img, int H, int W,
    const int* boxes, int nboxes,
    const float eps,
    float* out_x, float* out_y,
    bool* out_valid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nboxes) return;
    
    // Read box: boxes layout: [x1, y1, x2, y2] per box
    const int bx = boxes[idx*4 + 0];
    const int by = boxes[idx*4 + 1];
    const int bx2 = boxes[idx*4 + 2];
    const int by2 = boxes[idx*4 + 3];
    
    // Clip box to image bounds (safe-guard)
    int x1 = max(0, min(bx, W-1));
    int y1 = max(0, min(by, H-1));
    int x2 = max(0, min(bx2, W-1));
    int y2 = max(0, min(by2, H-1));
    
    // If box invalid or empty
    if (x2 < x1 || y2 < y1) {
        out_valid[idx] = 0;
        out_x[idx] = 0.0f;
        out_y[idx] = 0.0f;
        return;
    }
    
    // Find integer max inside box
    float vmax = -1e30f;
    int xmax = x1;
    int ymax = y1;
    
    for (int yy = y1; yy <= y2; ++yy) {
        int row = yy * W;
        for (int xx = x1; xx <= x2; ++xx) {
            float v = img[row + xx];
            if (v > vmax) {
                vmax = v;
                xmax = xx;
                ymax = yy;
            }
        }
    }
    
    // Ensure neighbors exist (not at image border and inside box)
    if (xmax <= x1 || xmax >= x2 || ymax <= y1 || ymax >= y2) {
        // max at box edge -> cannot do 3-point interpolation reliably
        out_valid[idx] = 0;
        out_x[idx] = (float)xmax;
        out_y[idx] = (float)ymax;
        return;
    }
    
    // Read intensity neighbors (safely)
    int idx_c = ymax * W + xmax;
    float Ic = img[idx_c] + eps;
    float Il = img[idx_c - 1] + eps;
    float Ir = img[idx_c + 1] + eps;
    float Iu = img[idx_c - W] + eps; // y-1
    float Id = img[idx_c + W] + eps; // y+1
    
    // Compute logs
    float lIl = logf(Il);
    float lIc = logf(Ic);
    float lIr = logf(Ir);
    float lIu = logf(Iu);
    float lId = logf(Id);
    
    // denominator and small-threshold
    const float EPS_DEN = 1e-7f;
    
    float denom_x = (lIl - 2.0f * lIc + lIr);
    float denom_y = (lIu - 2.0f * lIc + lId);
    
    if (fabsf(denom_x) < EPS_DEN || fabsf(denom_y) < EPS_DEN) {
        // degenerate case
        out_valid[idx] = 0;
        out_x[idx] = (float)xmax;
        out_y[idx] = (float)ymax;
        return;
    }
    
    float dx = 0.5f * (lIl - lIr) / denom_x;
    float dy = 0.5f * (lIu - lId) / denom_y;
    
    out_x[idx] = (float)xmax + dx;
    out_y[idx] = (float)ymax + dy;
    out_valid[idx] = 1;
}
"""

mod_get_subpixel = cp.RawModule(code=code_get_subpixel, backend='nvcc')
cuda_get_subpixel = mod_get_subpixel.get_function('cuda_get_subpixel')

def get_subpixel(f, boxes, eps=1e-6, threads_per_block=128):
    """
    Compute subpixel positions for particle bounding boxes using 3-point Gaussian interpolation.
    
    Parameters
    ----------
    f : array_like (H, W)
        Grayscale image (preferably float32). Can be NumPy or CuPy array.
    boxes : array_like (N, 4)
        Bounding boxes, dtype int32, layout [x1, y1, x2, y2] per row.
        Coordinates are expected in pixel indices.
    eps : float
        Small value added to intensities before log to avoid log(0).
    threads_per_block : int
        CUDA threads per block.
    
    Returns
    -------
    xs_sub : cupy.ndarray (N,) float32
    ys_sub : cupy.ndarray (N,) float32
    valid  : cupy.ndarray (N,) int32  (1 if interpolation succeeded, 0 otherwise)
    """
    # Ensure inputs are on GPU and correct dtypes
    H, W = f.shape
    
    boxes = cp.asarray(boxes, dtype=cp.int32)
    nboxes = boxes.shape[0]
    if nboxes == 0:
        return cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.int32)

    # Allocate outputs
    out_x = cp.zeros((nboxes,), dtype=cp.float32)
    out_y = cp.zeros((nboxes,), dtype=cp.float32)
    out_valid = cp.zeros((nboxes,), dtype=bool)

    # Launch kernel
    blocks = (nboxes + threads_per_block - 1) // threads_per_block
    cuda_get_subpixel(
        (blocks,), (threads_per_block,),
        (f, np.int32(H), np.int32(W), boxes, np.int32(nboxes), np.float32(eps),
         out_x, out_y, out_valid)
    )

    return out_x, out_y, out_valid

# mask = (slice(192, 704), slice(320, 832))
mask = (slice(1000-32, 1000), slice(1000, 1000+32))
def read_image(path):
    frame_a = cv2.imread(path[0], cv2.IMREAD_ANYDEPTH)
    frame_b = cv2.imread(path[1], cv2.IMREAD_ANYDEPTH)
    # p2, p98 = np.percentile(frame_a, (2, 98))
    # frame_a = exposure.rescale_intensity(frame_a, in_range=(p2, p98), out_range=(0, 65535))
    # p2, p98 = np.percentile(frame_b, (2, 98))
    # frame_b = exposure.rescale_intensity(frame_b, in_range=(p2, p98), out_range=(0, 65535))
    return frame_a[mask], frame_b[mask]

k = 0
frame_a, frame_b = read_image(frame_list[k: k + 2])

coords = []
b = []
frame = [frame_a, frame_b]
threshold = [0.0005, 0.0004]
for idx in [0, 1]:
    f = frame[idx]
    # binary = binarize_gaussian_frame(f, sigma=100, C=-4000)
    # binary = binarize_frame(f, block_size=101, C=-5000)
    boxes, labels, n = detect_blobs_bloblog(f, min_sigma=1, max_sigma=3, num_sigma=5,
                                            threshold=threshold[idx], overlap=0.5, box_scale=1.5)
    # binary = clean_binary_gpu(binary, min_size=1, morph_open=True)
    
    # boxes, labels, n = get_bounding_boxes(binary)
    b.append(boxes)
    
    f = cp.asarray(frame[idx], dtype=DTYPE_f)
    x_sp, y_sp, valid = get_subpixel(f, boxes)
    x_sp, y_sp = x_sp[valid], y_sp[valid]
    coords.append(cp.column_stack([x_sp, y_sp]))
    
def plot_detected_blobs(frame, boxes, color='lime', linewidth=1.5):
    """
    Plot detected particle bounding boxes on top of the frame.

    Parameters
    ----------
    frame : np.ndarray or cp.ndarray
        Input grayscale image
    boxes : np.ndarray or cp.ndarray
        Bounding boxes [x1, y1, x2, y2]
    color : str
        Color for the rectangle
    linewidth : float
        Rectangle edge width
    """

    # Convert to NumPy for matplotlib display
    if isinstance(frame, cp.ndarray):
        frame = frame.get()
    if isinstance(boxes, cp.ndarray):
        boxes = boxes.get()

    plt.figure(figsize=(7, 7))
    plt.imshow(frame, cmap='gray', origin='upper')

    for (x1, y1, x2, y2) in boxes:
        plt.plot([x1, x2, x2, x1, x1],
                  [y1, y1, y2, y2, y1],
                  color=color, linewidth=linewidth)

    plt.title("Detected Particles (LoG + Bounding Boxes)")
    plt.axis('off')
    plt.show()

def plot_detected_centroids(frame, coords, color='red', marker='x', markersize=4):
    """
    Plot detected particle centroids (from coords) on top of the frame.

    Parameters
    ----------
    frame : np.ndarray or cp.ndarray
        Input grayscale image
    coords : np.ndarray or cp.ndarray
        Coordinates of particle centers, shape (N, 2) as (y, x)
    color : str
        Marker color
    marker : str
        Matplotlib marker style (default 'x')
    markersize : float
        Size of the centroid marker
    """

    # Convert to NumPy for matplotlib display
    if isinstance(frame, cp.ndarray):
        frame = frame.get()
    if isinstance(coords, cp.ndarray):
        coords = coords.get()

    plt.figure(figsize=(7, 7))
    plt.imshow(frame, cmap='gray', origin='upper')

    if len(coords) > 0:
        plt.plot(coords[:, 0], coords[:, 1],
                 linestyle='none', marker=marker,
                 color=color, markersize=markersize)

    plt.title("Detected Particle Centroids")
    plt.axis('off')
    plt.show()

idx = 1
# plot_detected_blobs(frame[idx], b[idx])
plot_detected_centroids(frame[idx], coords[idx])