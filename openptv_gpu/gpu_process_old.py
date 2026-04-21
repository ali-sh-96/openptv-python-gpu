import numpy as np
from math import ceil
import cv2
import cupy as cp
from scipy.spatial import cKDTree, Delaunay
from cupyx.scipy.interpolate import interpn
from cupyx.scipy import ndimage as ndi
from . import DTYPE_b, DTYPE_i, DTYPE_u, DTYPE_f
import time
from sklearn.neighbors import NearestNeighbors

PARTICLE_METHOD = "log"
PARTICLE_SIZE = 1
THRESHOLD = 0
SUBPIXEL_METHOD = "gaussian"

SEARCH_SIZE = 8
CLUSTER_SIZE = 8

NUM_SLICES = 1
OVERLAP = 0
NUM_RELAXATION_ITERS = 10
RELAXATION_METHOD = "bidirectional"
BLOCK_SIZE = 64

ALLOWED_SUBPIXEL_METHODS = {"gaussian", "parabolic", "centroid"}
ALLOWED_PARTICLE_METHODS = {"log", "agt"}

class PTVGPU:
    """Bidirectional relaxation PTV algorithm.
    
    The algorithm involves utilizing the estimated displacement field from PIV to search for candidate matches in
    the second frame. For a given window size, multiple iterations can be conducted before the estimated velocity is
    onto a finer mesh. This procedure continues until the desired final mesh and the specified interpolated number of
    iterations are achieved.
    
    Algorithm Details
    -----------------
    Only window sizes that are multiples of 8 and a power of 2 are supported, and the minimum window size is 8.
    By default, windows are shifted symmetrically to reduce bias errors.
    The obtained displacement is the total dispalement for first iteration and the residual displacement dc for
    second iteration onwards.
    The new displacement is computed by adding the residual displacement to the previous estimation.
    Validation may be done by any combination of signal-to-noise ratio, median, median-absolute-deviation (MAD),
    mean, and root-mean-square (RMS) velocities.
    Smoothn can be used between iterations to improve the estimate and replace missing values.
    
    References
    ----------
    Scarano, F., & Riethmuller, M. L. (1999). Iterative multigrid approach in PIV image processing with discrete window
        offset. Experiments in Fluids, 26, 513-523.
        https://doi.org/10.1007/s003480050318
    Meunier, P., & Leweke, T. (2003). Analysis and treatment of errors due to high velocity gradients in particle image
        velocimetry. Experiments in fluids, 35(5), 408-421.
        https://doi.org/10.1007/s00348-003-0673-2
    Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values. Computational
        statistics & data analysis, 54(4), 1167-1178.
        https://doi.org/10.1016/j.csda.2009.09.020
    Shirinzad, A., Jaber, K., Xu, K., & Sullivan, P. E. (2023). An Enhanced Python-Based Open-Source Particle Image
        Velocimetry Software for Use with Central Processing Units. Fluids, 8(11), 285.
        https://doi.org/10.3390/fluids8110285
    
    Parameters
    ----------
    frame_shape : tuple
        Shape of the images in pixels.
    min_search_size : int
        Length of the sides of the square search window. Only supports multiples of 8 and powers of 2.
    search_size_iters : int or tuple, optional
        The length of this tuple represents the number of different window sizes to be used, and each entry specifies
        the number of times a particular window size is used.
    overlap_ratio : float or tuple, optional
        Ratio of the overlap between two windows (between 0 and 1) for different window sizes.
    shrink_ratio : float, optional
        Ratio (between 0 and 1) to shrink the window size for the first frame to use on the first iteration.
    center : bool, optional
        Whether to center the field with respect to the frame edges.
    deforming_order : int or tuple, optional (not active for the current GPU version)
        Order of the interpolation used for window deformation.
    normalize : bool or tuple, optional
        Whether to normalize the window intensity by subtracting the mean intensity.
    mask_zero: bool, optional
        Whether to mask the center of the cross-correlation map.
    subpixel_method : {"gaussian", "centroid", "parabolic"} or tuple, optional
        Method to estimate the subpixel location of the peak at each iteration.
    n_fft : int or tuple, optional
        Size-factor of the 2D FFT. n_fft of 2 is recommended for the smallest window size.
    deforming_par : float or tuple, optional
        Ratio (between 0 and 1) of the previous velocity used to deform each frame at every iteration.
        A default value of 0.5 is recommended to minimize the bias errors. A value of 1 corresponds to only
        second frame deformation.
    batch_size : int or "full" or tuple, optional
        Batch size for cross-correlation at every iteration.
    s2n_method : {"peak2peak", "peak2mean", "peak2energy"} or tuple, optional
        Method of the signal-to-noise ratio measurement.
    s2n_size : int or tuple, optional
        Half size of the region around the first correlation peak to ignore for finding the second peak.
        Default of 2 is only used if s2n_method == "peak2peak".
    validation_size : int or tuple, optional
        Size parameter for the validation kernel, kernel_size = 2 * size + 1.
    s2n_tol : float or None or tuple, optional
        Tolerance for the signal-to-noise (S2N) validation at every iteration.
    median_tol : float or None or tuple, optional
        Tolerance for the median velocity validation at every iteration.
    mad_tol : float or None or tuple, optional
        Tolerance for the median-absolute-deviation (MAD) velocity validation at every iteration.
    mean_tol : float or None or tuple, optional
        Tolerance for the mean velocity validation at every iteration.
    rms_tol : float or None or tuple, optional
        Tolerance for the root-mean-square (RMS) validation at every iteration.
    num_replacing_iters : int or tuple, optional
        Number of iterations per replacement cycle.
    replacing_method : {"spring", "median", "mean"} or tuple, optional
        Method to use for replacement.
    replacing_size : int or tuple, optional
        Size parameter for the replacement kernel, kernel_size = 2 * size + 1.
    revalidate : bool or tuple, optional
        Whether to revalidate the fields after every replecement iteration.
    smooth : bool or tuple, optional
        Whether to smooth the fields. Ignored for the last iteration.
    smoothing_par : float or None or tuple, optional
        Smoothing parameter to pass to smoothn to apply to the velocity fields.
    dt : float, optional
        Time delay separating the two frames.
    scaling_par : int, optional
        Scaling factor to apply to the velocity fields.
    mask : ndarray or None, optional
        2D array with non-zero values indicating the masked locations.
    dtype_f : str, optional
        Float data type. Default of single precision is used if not specified.
    
    Attributes
    ----------
    coords : tuple
        A tuple of 2D arrays, (x, y) coordinates, where the velocity field is computed.
    field_mask : ndarray
        2D boolean array of masked locations for the last iteration.
    s2n_ratio : ndarray
        Signal-to-noise ratio of the cross-correlation map for the last iteration.
    
    """
    def __init__(self,
                 frame_shape,
                 particle_method=PARTICLE_METHOD,
                 particle_size=PARTICLE_SIZE,
                 threshold=THRESHOLD,
                 subpixel_method=SUBPIXEL_METHOD,
                 search_size=SEARCH_SIZE,
                 cluster_size=CLUSTER_SIZE,
                 num_relaxation_iters=NUM_RELAXATION_ITERS,
                 relaxation_method=RELAXATION_METHOD,
                 num_slices=NUM_SLICES,
                 overlap=OVERLAP,
                 dt=1,
                 scaling_par=1,
                 mask=None,
                 dtype_f=DTYPE_f):
        
        # Geometry settings.
        self.frame_shape = frame_shape
        self.n_dims = len(self.frame_shape)
        self.particle_method = particle_method
        
        if particle_method == "log":
            self.size_a, self.size_b = particle_size
        else:
            kernel_size = tuple(2 * size + 1 for size in particle_size)
            self.size_a, self.size_b = kernel_size
        
        self.Ca, self.Cb = threshold
        self.subpixel_method = subpixel_method
        
        # Relaxation settings.
        self.search_size = search_size
        self.cluster_size = cluster_size
        self.n_iters = num_relaxation_iters
        self.relaxation_method = relaxation_method
        self.n_slices = (num_slices,) * self.n_dims if isinstance(num_slices, int) else num_slices
        self.overlap = overlap
        
        # Scaling settings.
        self.dt = dt
        self.scaling_par = scaling_par
        
        # Data type settings.
        self.dtype_f = np.float32 if dtype_f == "float32" else DTYPE_f
        self.dtype_i = np.int32 if dtype_f is not DTYPE_f else DTYPE_i
        self.dtype_b = DTYPE_b
        
        # Convert mask to boolean array.
        self.mask = mask.astype(self.dtype_b) if mask is not None else None
        
        # Initialize the PTV process.
        self.slices = self.get_slices(self.frame_shape, self.n_slices, self.overlap)
        self.init_coords_a, self.init_coords_b = None, None
        
        # Compile the CUDA kernels.
        self.mod_mask_coords = cp.RawModule(code=code_mask_coords)
        self.mod_label_blobs = cp.RawModule(code=code_label_blobs)
        self.mod_get_peak = cp.RawModule(code=code_get_peak)
        self.mod_find_clusters = cp.RawModule(code=code_find_clusters)
        self.mod_get_clusters = cp.RawModule(code=code_get_clusters)
        self.mod_update_probs = cp.RawModule(code=code_update_probs)
        self.mod_median_validation = cp.RawModule(code=code_median_validation)
    
    def __call__(self, frame_a, frame_b, field=None):
        self.field = tuple(cp.asarray(f, dtype=self.dtype_f) for f in field) \
            if field is not None else None
        
        # Mask the frames.
        self.frame_a, self.frame_b = self.mask_frames(frame_a, frame_b, mask=self.mask)
        
        # Initialize the field object.
        self.ptv_field = PTVFIELDGPU(modules=(self.mod_label_blobs, self.mod_get_peak),
                                     particle_method=self.particle_method,
                                     subpixel_method=self.subpixel_method,
                                     dtype_f=self.dtype_f)
        
        # Initialize the relaxation object.
        modules = (self.mod_find_clusters,
                   self.mod_get_clusters,
                   self.mod_update_probs)
        
        self.relaxation = RelaxationGPU(modules=modules,
                                        search_size=self.search_size,
                                        cluster_size=self.cluster_size,
                                        num_relaxation_iters=self.n_iters,
                                        relaxation_method=self.relaxation_method,
                                        dtype_f=self.dtype_f)
        
        # Get the local particle coordinates.
        self.init_coords_a, self.init_coords_b = self.get_coords(frame_a, frame_b, is_gpu=True)
        
        self.coords_a, self.coords_b = [], []
        for self.y_bounds, self.x_bounds in self.slices:
            # Get the coords within the slice.
            # coords_a, coords_b = self.mask_coords(self.init_coords_a, self.init_coords_b)
            coords_a, coords_b = self.init_coords_a, self.init_coords_b
            
            # Perform relaxation.
            coords_a, coords_b = self.relaxation(coords_a, coords_b, field=self.field)
            
            self.coords_a.append(coords_a)
            self.coords_b.append(coords_b)
        
        # Remove all duplicates.
        self.coords_a, self.coords_b = self.get_unique_coords(self.coords_a, self.coords_b)
        
        # Get the displacement field.
        u, v = self.get_displacement(self.coords_a, self.coords_b)
        
        # start_time = time.time()
        # u, v = self.validate_fields(u, v)
        # print(time.time() - start_time)
        
        # return u.get(), v.get()
        return u, v
    
    def get_bounds(self, length, n_slices, overlap):
        "Returns bounds that divide a given length into overlapping slices."
        if n_slices == 1:
            return cp.array([[0, length]], dtype=self.dtype_i)
        
        # Compute the initial step.
        step = max((length - overlap) // n_slices, 1)
        
        # Get a tentative slice size.
        size = step + overlap
        
        starts = np.arange(n_slices) * step
        ends = starts + size
        
        # Clamp last slice exactly to length.
        ends[-1] = length
        starts[-1] = max(0, length - size)
        
        # Ensure starts are strictly increasing.
        starts = np.minimum(starts, ends - 1)
        
        return np.stack((starts, ends), axis=1).astype(self.dtype_i)

    def get_slices(self, frame_shape, n_slices, overlap):
        "Returns bounds for each overlapping 2D slice of an image."
        # Initialize number of slices and minimum overlap.
        ht, wd = frame_shape
        m, n = self.n_slices
        y_overlap, x_overlap = overlap
        
        y_bounds = self.get_bounds(ht, m, y_overlap)
        x_bounds = self.get_bounds(wd, n, x_overlap)
        
        # Update number of slices.
        m = y_bounds.shape[0]
        n = x_bounds.shape[0]
        
        # Generate all slices using meshgrid.
        y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
        y, x = y.ravel(), x.ravel()
        return [(y_bounds[yi], x_bounds[xi]) for yi, xi in zip(y, x)]
    
    def mask_frames(self, frame_a, frame_b, mask=None):
        """Masks the frames."""
        if mask is not None:
            frame_a[mask] = 0
            frame_b[mask] = 0
        
        # Get the global minimum and maximum intensities.
        self.frame_a_min, self.frame_a_max = frame_a.min(), frame_a.max()
        self.frame_b_min, self.frame_b_max = frame_b.min(), frame_b.max()
        
        return frame_a, frame_b
    
    def get_coords(self, frame_a, frame_b, is_gpu=False):
        "Returns the local particle coordinates."
        coords_a = self.ptv_field.get_coords(frame_a,
                                             f_min=self.frame_a_min,
                                             f_max=self.frame_a_max,
                                             size=self.size_a,
                                             C=self.Ca)
        
        coords_b = self.ptv_field.get_coords(frame_b,
                                             f_min=self.frame_b_min,
                                             f_max=self.frame_b_max,
                                             size=self.size_b,
                                             C=self.Cb)
        
        if not is_gpu:
            coords_a, coords_b = coords_a.get(), coords_b.get()
        
        return coords_a, coords_b
    
    def get_unique_coords(self, coords_a, coords_b):
        "Returns all unique particle coordinates."
        coords_a = np.concatenate(coords_a, axis=0)
        coords_b = np.concatenate(coords_b, axis=0)
        pairs = np.unique(np.hstack([coords_a, coords_b]), axis=0)
        
        coords_a = pairs[:, :2]
        coords_b = pairs[:, 2:]
        
        return coords_a, coords_b
    
    def mask_coords(self, coords_a, coords_b):
        yd, yu = map(float, self.y_bounds)
        xl, xr = map(float, self.x_bounds)
        cuda_mask_coords = self.mod_mask_coords.get_function("cuda_mask_coords")
        
        block_size = 256
        Na = coords_a.shape[0]
        Nb = coords_b.shape[0]
        
        # GPU arrays
        mask_a = cp.full((Na,), fill_value=False, dtype=self.dtype_b)
        mask_b = cp.full((Nb,), fill_value=False, dtype=self.dtype_b)
        
        # Launch kernel
        grid_size = ceil(Na / block_size)
        cuda_mask_coords((grid_size,), (block_size,),
                         (coords_a,
                          self.dtype_f(yd), self.dtype_f(yu), self.dtype_f(xl), self.dtype_f(xr),
                          self.dtype_i(Na),
                          mask_a))
        
        grid_size = ceil(Nb / block_size)
        cuda_mask_coords((grid_size,), (block_size,),
                         (coords_b,
                          self.dtype_f(yd), self.dtype_f(yu), self.dtype_f(xl), self.dtype_f(xr),
                          self.dtype_i(Nb),
                          mask_b))
        
        # Filter coords_a and coords_b using mask
        return coords_a[mask_a], coords_b[mask_b]
    
    def validate_fields(self, u, v):
        coords = self.coords_a
        x, y =self.coords
        neighbors = self.find_neighbors(coords)
        
        # Convert to CSR format for GPU.
        counts = [len(n) for n in neighbors]
        nbr_ptr = np.zeros(len(neighbors) + 1, dtype=self.dtype_i)
        np.cumsum(counts, out=nbr_ptr[1:])
        nbr_idx = np.concatenate([np.array(list(n), dtype=np.int32) for n in neighbors])
        
        N = len(u)
        threads = 256
        blocks = (N + threads - 1) // threads
        
        x = cp.asarray(x, dtype=self.dtype_f)
        y = cp.asarray(y, dtype=self.dtype_f)
        u = cp.asarray(u, dtype=self.dtype_f)
        v = cp.asarray(v, dtype=self.dtype_f)
        V = cp.sqrt(u**2 + v**2)
        
        nbr_idx_gpu = cp.asarray(nbr_idx)
        nbr_ptr_gpu = cp.asarray(nbr_ptr)
        mask_u = cp.zeros(N, dtype=self.dtype_b)
        mask_v = cp.zeros(N, dtype=self.dtype_b)
        mask_V = cp.zeros(N, dtype=self.dtype_b)
        
        eps_a = 0.04
        MEDIAN_TOL = 2
        cuda_median_validation = self.mod_median_validation.get_function('cuda_median_validation_ptv')
        
        cuda_median_validation((blocks,), (threads,),
                               (u,
                                x,
                                y,
                                nbr_idx_gpu,
                                nbr_ptr_gpu,
                                self.dtype_f(eps_a),
                                self.dtype_f(MEDIAN_TOL),
                                self.dtype_i(N),
                                mask_u))
        
        cuda_median_validation((blocks,), (threads,),
                               (v,
                                x,
                                y,
                                nbr_idx_gpu,
                                nbr_ptr_gpu,
                                self.dtype_f(eps_a),
                                self.dtype_f(MEDIAN_TOL),
                                self.dtype_i(N),
                                mask_v))
        
        cuda_median_validation((blocks,), (threads,),
                               (V,
                                x,
                                y,
                                nbr_idx_gpu,
                                nbr_ptr_gpu,
                                self.dtype_f(eps_a),
                                self.dtype_f(MEDIAN_TOL),
                                self.dtype_i(N),
                                mask_V))
        
        mask = [mask_u, mask_v, mask_V]
        mask = cp.all(cp.stack(mask, axis=0), axis=0)
        
        # --- Apply mask ---
        u = u[mask]
        v = v[mask]
        self.coords_a = self.coords_a[mask.get()]
        self.coords_b = self.coords_b[mask.get()]
        
        return u, v
    
    # def validate_fields(self, u, v):
    #     u = cp.asarray(u, dtype=self.dtype_f)
    #     v = cp.asarray(v, dtype=self.dtype_f)
    #     xp, yp, up, vp = self.field
    #     xp, yp = xp[0, :], yp[:, 0]
    
    #     # Interpolate predicted field (up, vp) at measured coordinates
    #     up = interpn((yp, xp), up, self.coords_a, bounds_error=False, fill_value=None)
    #     vp = interpn((yp, xp), vp, self.coords_a, bounds_error=False, fill_value=None)
    #     up, vp = up.astype(self.dtype_f), vp.astype(self.dtype_f)
    
    #     eps = 1e-8
    
    #     # Compute norms
    #     norm_u = cp.sqrt(u * u + v * v) + eps
    #     norm_p = cp.sqrt(up * up + vp * vp) + eps
    
    #     # --- Angle residual (radians) ---
    #     dot = u * up + v * vp
    #     cos_theta = dot / (norm_u * norm_p)
    #     cos_theta = cp.clip(cos_theta, -1.0, 1.0)
    #     r_a = cp.arccos(cos_theta)  # radians
    
    #     # --- Magnitude residual (relative difference) ---
    #     r_m = cp.abs(norm_u - norm_p) / norm_p  # dimensionless
    
    #     # --- Combined residual ---
    #     r_star = cp.sqrt(r_a**2 + r_m**2)
    
    #     # --- Mask based on threshold ---
    #     # Example: keep vectors where combined residual < 0.3 (tune as needed)
    #     threshold = 0.3
    #     mask = r_star < threshold
        
    #     # --- Apply mask ---
    #     u = u[mask]
    #     v = v[mask]
    #     self.coords_a = self.coords_a[mask.get()]
    #     self.coords_b = self.coords_b[mask.get()]
        
    #     return u, v
    
    def find_neighbors(self, coords):
        tri = Delaunay(coords)
        simplices = tri.simplices
        
        # Collect all edges (each simplex gives 3 edges).
        edges = np.concatenate([
            simplices[:, [0, 1]],
            simplices[:, [1, 2]],
            simplices[:, [2, 0]]
        ], axis=0)

        # Ensure edges are undirected (small index first).
        edges = np.sort(edges, axis=1)
        
        # Remove duplicates.
        edges = np.unique(edges, axis=0)

        n_points = tri.points.shape[0]
        neighbors = [set() for _ in range(n_points)]
        for i, j in edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
        return neighbors
    
    def find_neighbors_triangulation(self, coords):
        tri = Delaunay(coords)
        simplices = tri.simplices
    
        # Collect edges from all triangles (vectorized)
        edges = np.vstack([
            simplices[:, [0, 1]],
            simplices[:, [1, 2]],
            simplices[:, [2, 0]]
        ])
    
        # Sort within each edge and remove duplicates
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
    
        # Build adjacency (CSR-like)
        n_points = len(coords)
        i, j = edges[:, 0], edges[:, 1]
    
        # Symmetrize (since undirected)
        all_i = np.concatenate([i, j])
        all_j = np.concatenate([j, i])
    
        # Sort by source node
        order = np.argsort(all_i)
        all_i = all_i[order]
        all_j = all_j[order]
    
        # Count neighbors per node
        counts = np.bincount(all_i, minlength=n_points)
        nbr_ptr = np.concatenate([[0], np.cumsum(counts)])
        nbr_idx = all_j
    
        return nbr_idx, nbr_ptr
    
    def get_displacement(self, coords_a, coords_b):
        """Returns the particle displacements."""
        u = coords_b[:, 1] - coords_a[:, 1]
        v = coords_b[:, 0] - coords_a[:, 0]
        return u, v
    
    @property
    def init_coords(self):
        "Returns the initial particle coordinates."
        coords_a, coords_b = self.init_coords_a.get(), self.init_coords_b.get()
        
        return coords_a, coords_b
    
    @property
    def matched_coords(self):
        "Returns the matched particle coordinate pairs."
        coords_a, coords_b = self.coords_a, self.coords_b
        
        return coords_a, coords_b
    
    @property
    def coords(self, is_gpu=False):
        "Returns the x and y components of the matched particle coordinates."
        if self.coords_a is not None:
            x = self.coords_a[:, 1]
            y = self.coords_a[:, 0]
        
        if is_gpu:
            x, y = x.get(), y.get()
        
        return x, y

class PTVFIELDGPU:
    def __init__(self, modules,
                 particle_method=PARTICLE_METHOD,
                 subpixel_method=SUBPIXEL_METHOD,
                 dtype_f=DTYPE_f):
        
        self.mod_label_blobs, self.mod_get_peak = modules
        self.particle_method = particle_method
        self.subpixel_method = subpixel_method
        
        # Small value added to denominator for subpixel approximation.
        self.eps = 1e-6
        
        # Data type settings.
        self.dtype_f = dtype_f
        self.dtype_i = np.int32 if dtype_f is not DTYPE_f else DTYPE_i
        self.dtype_u = DTYPE_u
    
    def binarize_frame(self, f, f_min, f_max, kernel_size=3, C=0):
        """Performs adaptive gaussian thresholding."""
        # Convert to 8 bit image for cv2.
        # Mimics cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype('uint8').
        f = f.astype(self.dtype_f)
        f = (f - f_min) / (f_max - f_min)
        f = np.round(f * 255).astype(np.uint8)
        
        # Apply adaptive Gaussian thresholding.
        thresh = cv2.adaptiveThreshold(f,
                                       maxValue=255,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY,
                                       blockSize=kernel_size,
                                       C=C)
        
        binary = (thresh > 0).astype(np.uint8)
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
        
        # Send labels to device.
        labels = cp.asarray(labels.astype(self.dtype_i))
        
        return labels, n_labels
    
    def blob_log(self, f, sigma, threshold):
        """Performs a single-scale Laplacian of Gaussian blob detection."""
        # Compute Laplacian of Gaussian.
        log = ndi.gaussian_laplace(f, sigma=sigma)
        
        # Normalization for scale invariance (optional).
        log = cp.abs(log) * sigma**2
        
        # Threshold
        mask = log > threshold
        
        # Local maxima detection in 3x3 neighborhood.
        max_filt = ndi.maximum_filter(log, size=3)
        maxima = mask & (log == max_filt)
        
        # Get coordinates.
        coords = cp.argwhere(maxima)
        n_blobs = coords.shape[0]
        
        # Return as (y, x, sigma).
        blobs = cp.column_stack((coords, cp.full((n_blobs,), sigma, dtype=self.dtype_f)))
        return blobs, n_blobs
    
    def get_blobs(self, f, f_min, f_max, size=1, C=0):
        """Returns the labels of the detected blobs by Laplacian of Gaussian."""
        # Normalize to [0, 1].
        f = cp.asarray(f, dtype=self.dtype_f)
        f = (f - f_min) / (f_max - f_min + 1e-8)
        
        # Detect the blobs by LoG.
        blobs, n_labels = self.blob_log(f, sigma=size, threshold=C)
        if blobs.size == 0:
            H, W = f.shape
            return np.zeros((self.ht, self.wd), dtype=self.dtype_i), 1, blobs
        
        # Convert sigma to radius.
        blobs[:, 2] *= cp.sqrt(2)
        
        # Configure CUDA kernel.
        blobs = cp.asarray(blobs, dtype=self.dtype_f)
        labels = cp.zeros((self.ht, self.wd), dtype=self.dtype_i)
        block_size = 256
        grid_size = ceil(self.ht * self.wd / block_size)
        
        cuda_label_blobs = self.mod_label_blobs.get_function("cuda_label_blobs")
        cuda_label_blobs((grid_size,), (block_size,),
                          (blobs,
                          self.dtype_i(n_labels),
                          labels.ravel(),
                          self.dtype_i(self.ht),
                          self.dtype_i(self.wd)))
        
        return labels, n_labels
    
    def get_peak(self, f, labels, n_labels):
        """Returns the row and column of the first peaks in the labeled image."""
        # Convert image for atomic max operation on device.
        f = cp.asarray(f, dtype= self.dtype_u)
        
        # Configure CUDA kernel.
        peak = cp.zeros((n_labels,), dtype=self.dtype_i)
        y_peak = cp.zeros((n_labels,), dtype=self.dtype_i)
        x_peak = cp.zeros((n_labels,), dtype=self.dtype_i)
        counts = cp.zeros((n_labels,), dtype=self.dtype_i)
        block_size = BLOCK_SIZE
        grid_size = ceil(self.ht * self.wd / block_size)
        
        cuda_get_peak = self.mod_get_peak.get_function("cuda_get_peak")
        cuda_get_mask = self.mod_get_peak.get_function("cuda_get_mask")
        cuda_get_peak((grid_size,), (block_size,),
                      (f.ravel(),
                       labels.ravel(),
                       self.dtype_i(self.ht),
                       self.dtype_i(self.wd),
                       self.dtype_i(n_labels),
                       peak,
                       y_peak,
                       x_peak))
        
        cuda_get_mask((grid_size,), (block_size,),
                      (f.ravel(),
                       labels.ravel(),
                       self.dtype_i(self.ht),
                       self.dtype_i(self.wd),
                       self.dtype_i(n_labels),
                       peak,
                       counts))
        
        mask = counts == 1
        # print(cp.count_nonzero(mask))
        x_peak = x_peak[mask]
        y_peak = y_peak[mask]
        # Exclude background (label 0).
        return y_peak, x_peak
    
    def get_subpixel(self, f, y_peak, x_peak):
        """Returns the subpixel estimation of the peak locations."""
        f = cp.asarray(f)
        
        # Ensure indices are not on the border.
        _mask = (y_peak > 0) & (y_peak < self.ht - 1) & (x_peak > 0) & (x_peak < self.wd - 1)
        yc = y_peak[_mask]
        xc = x_peak[_mask]
        
        # Get the center and neighboring values.
        fc  = f[yc, xc]
        fd = f[yc - 1, xc]
        fu = f[yc + 1, xc]
        fl = f[yc, xc - 1]
        fr = f[yc, xc + 1]
        
        # Ensure yc, xc are also float for math operations.
        yc = yc.astype(self.dtype_f)
        xc = xc.astype(self.dtype_f)
        
        if self.subpixel_method == "gaussian":
            _mask = (fc > 0) & (fd > 0) & (fu > 0) & (fl > 0) & (fr > 0)
            yc, xc = yc[_mask], xc[_mask]
            fc, fd, fu, fl, fr = fc[_mask], fd[_mask], fu[_mask], fl[_mask], fr[_mask]
            
            fc, fd, fu, fl, fr = cp.log(fc), cp.log(fd), cp.log(fu), cp.log(fl), cp.log(fr)
            y_sub = yc + 0.5 * (fd - fu) / (fd - 2.0 * fc + fu + self.eps)
            x_sub = xc + 0.5 * (fl - fr) / (fl - 2.0 * fc + fr + self.eps)
        
        elif self.subpixel_method == "parabolic":
            y_sub = yc + 0.5 * (fd - fu) / (fd - 2.0 * fc + fu + self.eps)
            x_sub = xc + 0.5 * (fl - fr) / (fl - 2.0 * fc + fr + self.eps)
        
        elif self.subpixel_method == "centroid":
            y_sub = yc + (fu - fd) / (fd + fc + fu + self.eps)
            x_sub = xc + (fr - fl) / (fl + fc + fr + self.eps)
        
        return cp.column_stack((y_sub, x_sub))
    
    def get_coords(self, f, f_min, f_max, size=1, C=0):
        "Returns the image coordinates of the detected particles."
        self.ht, self.wd = f.shape
        
        # Get the labels using the specified method.
        if self.particle_method == "agt":
            labels, n_labels = self.binarize_frame(f, f_min, f_max, kernel_size=size, C=C)
        elif self.particle_method == "log":
            labels, n_labels = self.get_blobs(f, f_min, f_max, size=size, C=C)
        
        # Get the peak locations inside each label.
        y_peak, x_peak = self.get_peak(f, labels, n_labels)
        coords = self.get_subpixel(f, y_peak, x_peak)
        return coords

class RelaxationGPU:
    def __init__(self, modules,
                 search_size=SEARCH_SIZE,
                 cluster_size=CLUSTER_SIZE,
                 num_relaxation_iters=NUM_RELAXATION_ITERS,
                 relaxation_method=RELAXATION_METHOD,
                 dtype_f=DTYPE_f):
        
        self.search_size = search_size
        self.cluster_size = cluster_size
        self.n_iters = num_relaxation_iters
        self.relaxation_method = relaxation_method
        
        (self.mod_find_clusters,
         self.mod_get_clusters,
         self.mod_update_probs) = modules
        
        # Settings for float and int data types.
        self.dtype_f = dtype_f
        self.dtype_i = np.int32 if dtype_f is not DTYPE_f else DTYPE_i
        self.dtype_b = DTYPE_b
    
    def __call__(self, coords_a, coords_b, field=None):
        self.coords_a = coords_a
        self.coords_b = coords_b
        self.field = field
            
        # Forward particle matching.
        field = self.interpolate_field(coords_a, self.field, direction="forward")
        coords_aa, coords_ab = self.match_particles(coords_a, coords_b, field=field)
        
        # Perform bidirectional validation on CPU.
        if self.relaxation_method == "bidirectional":
            # Backward particle matching.
            field = self.interpolate_field(coords_b, self.field, direction="backward")
            coords_bb, coords_ba = self.match_particles(coords_b, coords_a, field=field)
            
            # Perform bidirectional validation.
            coords_a, coords_b = self.bidirectional_validation(coords_aa.get(), coords_ab.get(),
                                                               coords_ba.get(), coords_bb.get())
        
        else:
            coords_a, coords_b = coords_aa.get(), coords_ab.get()
        
        return coords_a, coords_b
    
    def interpolate_field(self, coords, field, direction="forward"):
        if field is not None:
            x, y, u, v = field
            
            x_grid = x[0, :]
            y_grid = y[:, 0]
            
            up = interpn((y_grid, x_grid), u, coords, bounds_error=False, fill_value=None)
            vp = interpn((y_grid, x_grid), v, coords, bounds_error=False, fill_value=None)
            up, vp = up.astype(self.dtype_f), vp.astype(self.dtype_f)
            
            if direction == "backward":
                up, vp = -up, -vp
            
            field = (vp, up)
        
        return field
    
    def find_clusters(self, coords_a, coords_b, R=0, field=None):
        """
        GPU clustering using a raw CUDA kernel.

        Parameters
        ----------
        coords_a : (Na,2) cp.ndarray, dtype=float32
        coords_b : (Nb,2) cp.ndarray, dtype=float32
        up, vp   : (Na,) cp.ndarray or None (dtype=float32)
        R        : float (search radius)
        maxlen   : int (max cluster size; must be <= blockDim.x * something; we use 32)

        Returns
        -------
        coords_a_kept : (Nkept,2) cp.ndarray
            Filtered coords_a (original coords; displacement was used only for search if provided)
        clusters_list : list of cp.ndarray
            Each entry is an int array of indices into coords_b for that kept coords_a
        kept_indices  : cp.ndarray (Nkept,) int32
            Indices of the kept coords_a (so coords_a[kept_indices] == coords_a_kept)
        raw_clusters  : cp.ndarray (Na, maxlen) int32
            (optional) full clusters buffer with -1 for empty slots (before filtering)
        counts        : cp.ndarray (Na,) int32
            counts per coords_a (before filtering)
        """
        Na = coords_a.shape[0]
        Nb = coords_b.shape[0]
        
        # Create dummy variables if field is not given.
        if field is None:
            up = cp.zeros((1,), dtype=self.dtype_f)
            vp = cp.zeros((1,), dtype=self.dtype_f)
            is_predicted = False
        else:
            vp, up = field
            is_predicted = True
        
        # Allocate outputs on device.
        counts = cp.zeros((Na,), dtype=self.dtype_i)
        
        # kernel launch: one block per coords_a, blockDim.x = BLOCK_SIZE
        block_size = (BLOCK_SIZE, 1, 1)
        grid_size = (Na, 1, 1)
        
        # call kernel
        cuda_find_clusters = self.mod_find_clusters.get_function('cuda_find_clusters')
        cuda_find_clusters(grid_size, block_size,
                           (coords_a.ravel(),
                            coords_b.ravel(),
                            up.ravel(),
                            vp.ravel(),
                            self.dtype_b(is_predicted),
                            self.dtype_i(Na),
                            self.dtype_i(Nb),
                            R**2,
                            self.dtype_i(BLOCK_SIZE),
                            counts))
        
        # Indices of non-empty clusters
        indices = cp.nonzero(counts > 0)[0].astype(self.dtype_i)
        
        return indices
    
    def get_clusters(self, coords_a, coords_b, R=0, field=None):
        Na = coords_a.shape[0]
        Nb = coords_b.shape[0]
        
        # Create dummy variables if field not given.
        if field is None:
            up = cp.zeros((1,), dtype=self.dtype_f)
            vp = cp.zeros((1,), dtype=self.dtype_f)
            is_predicted = False
        else:
            vp, up = field
            is_predicted = True
        
        counts = cp.zeros((Na,), dtype=self.dtype_i)
        indices = cp.full((Na, BLOCK_SIZE), -1, dtype=self.dtype_i)
        clusters = cp.full((Na, BLOCK_SIZE, 2), cp.nan, dtype=self.dtype_f)
        
        block_size = (BLOCK_SIZE,)
        grid_size = (Na,)
        
        cuda_get_clusters = self.mod_get_clusters.get_function('cuda_get_clusters')
        cuda_get_clusters(grid_size, block_size,
                          (coords_a.ravel(),
                           coords_b.ravel(),
                           up.ravel(),
                           vp.ravel(),
                           self.dtype_b(is_predicted),
                           self.dtype_i(Na),
                           self.dtype_i(Nb),
                           R**2,
                           self.dtype_i(BLOCK_SIZE),
                           counts,
                           indices.ravel(),
                           clusters.ravel()))
        
        return clusters, counts, indices
    
    def update_probs(self, delta, Nb, Na, idx, P, n_iters=0, sigma=1, field=None):
        N, max_len, _ = delta.shape
        
        cuda_update_probs = self.mod_update_probs.get_function('cuda_update_probs')
        field = cp.column_stack(field) if field is not None else cp.zeros((1,), dtype=self.dtype_f)
        
        delta = delta.ravel()
        field = field.ravel()
        idx = idx.ravel()
        for _ in range(n_iters):
            P = P.ravel()
            p = P.copy()
            
            block_size = (BLOCK_SIZE,)
            grid_size = (N,)
            
            cuda_update_probs(grid_size, block_size,
                              (delta,
                               field,
                               Nb,
                               Na,
                               idx,
                               P,
                               p,
                               self.dtype_f(sigma),
                               self.dtype_i(N),
                               self.dtype_i(max_len),
                               self.dtype_i(max_len)))
            P = P.reshape(N, max_len)
            P /= cp.sum(P, axis=1, keepdims=True)
        return P
    
    def match_particles(self, coords_a, coords_b, field=None):
        # Filtering particles in coords_a with no nearby candidate matches.
        Rb = cp.full_like(coords_b, fill_value=self.search_size)
        Ra = cp.full_like(coords_a, fill_value=self.cluster_size)
        # indices = self.find_clusters(coords_a, coords_b, R=Rb, field=field)
        # coords_a = coords_a[indices]
        # field = tuple(f[indices] for f in field) if field is not None else None
        N = coords_a.shape[0]
        
        # Clustering coordinates in a and b
        clusters_b, Nb, _ = self.get_clusters(coords_a, coords_b, R=Rb, field=field)
        clusters_a, Na, idx = self.get_clusters(coords_a, coords_a, R=Ra)
        
        # Perform relaxation
        N = coords_a.shape[0]
        delta = clusters_b - coords_a[:, None, :]
        
        # Initialize probability array.
        P = cp.broadcast_to(1.0 / (Nb[:, None] + 1), (N, BLOCK_SIZE)).copy()
        P = P.astype(self.dtype_f)
        mask = cp.arange(BLOCK_SIZE)[None, :] >= (Nb[:, None] + 1)
        P[mask] = 0
        
        # Perform relaxation.
        P = self.update_probs(delta, Nb, Na, idx, P, n_iters=self.n_iters, field=field)
        
        # Return matches with maximum probability.
        jmax = cp.argmax(P, axis=1)
        mb = clusters_b[cp.arange(N), jmax]
        ma = coords_a
        return ma, mb
    
    def bidirectional_validation(self, maa, mab, mba, mbb, tol=1.0):
        tree = cKDTree(mbb)
        dmin, idx_min = tree.query(mab, distance_upper_bound=tol)
        
        valid = dmin < tol
        diff_back = maa[valid] - mba[idx_min[valid]]
        err_back = np.sqrt(np.sum(diff_back**2, axis=1))
        mask = np.zeros_like(valid, dtype=bool)
        mask[valid] = err_back < tol
        return maa[mask], mab[mask]
    
    @property
    def coords(self):
        x = self.ma[:, 1] if self.ma is not None else None
        y = self.ma[:, 0] if self.ma is not None else None
        return x, y

code_mask_coords = """
extern "C" __global__
void cuda_mask_coords(const float* coords,
                      const float yd,
                      const float yu,
                      const float xl,
                      const float xr,
                      const int N,
                      bool* mask)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    float y = coords[2 * idx];
    float x = coords[2 * idx + 1];
    
    mask[idx] = (y >= yd && y < yu && x >= xl && x < xr);
}
"""

code_label_blobs = """
extern "C" __global__
void cuda_label_blobs(
    const float* blobs,  // (N, 3): y, x, r
    const int N,
    int* labels,         // (H, W)
    const int H,
    const int W)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;

    float y = blobs[3 * n + 0];
    float x = blobs[3 * n + 1];
    float r = blobs[3 * n + 2];
    float r2 = r * r;

    int y0 = max(0, (int)floorf(y - r));
    int y1 = min(H - 1, (int)ceilf (y + r));
    int x0 = max(0, (int)floorf(x - r));
    int x1 = min(W - 1, (int)ceilf (x + r));

    for (int yy = y0; yy <= y1; ++yy) {
        float dy = (yy - y);
        float dy2 = dy * dy;
        if (dy2 > r2) continue;
        int offset = yy * W;
        for (int xx = x0; xx <= x1; ++xx) {
            float dx = (xx - x);
            if (dy2 + dx * dx <= r2) {
                labels[offset + xx] = n + 1;  // label starts at 1
            }
        }
    }
}
"""

code_get_peak = """
extern "C" __global__
void cuda_get_peak(
    const unsigned int* f,
    const int* labels,
    const int H,
    const int W,
    const int n_labels,
    unsigned int* max_vals,
    int* y_peak,
    int* x_peak
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = H * W;
    if (idx >= size) return;
    
    int label = labels[idx];
    if (label < 1 || label >= n_labels) return;
    
    unsigned int val = f[idx];
    unsigned int old = atomicMax(&max_vals[label], val);
    
    // If this thread set a new max, update coordinates and reset count
    if (val > old) {
        __threadfence();
        if (max_vals[label] == val) {
            y_peak[label] = idx / W;
            x_peak[label] = idx % W;
        }
    }
}

extern "C" __global__
void cuda_get_mask(
    const unsigned int* f,
    const int* labels,
    int H,
    int W,
    int n_labels,
    const unsigned int* max_vals,
    int* counts
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = H * W;
    if (idx >= size) return;

    int label = labels[idx];
    if (label < 1 || label >= n_labels) return;

    unsigned int val = f[idx];

    if (val == max_vals[label]) {
        atomicAdd(&counts[label], 1);
    }
}
"""

code_filter_bounds = """
extern "C" __global__
void cuda_filter_bounds(const int* y_peak, const int* x_peak,
                        const int ht, const int wd,
                        int* out_i, int* out_j, int* count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ht * wd) return;
    int i = y_peak[idx];
    int j = x_peak[idx];
    if (i > 0 && i < ht - 1 && j > 0 && j < wd - 1) {
        int pos = atomicAdd(count, 1);
        out_i[pos] = i;
        out_j[pos] = j;
    }
}
"""

code_find_clusters = """
extern "C" __global__
void cuda_find_clusters(
    const float* coords_a,   // (Na,2)
    const float* coords_b,   // (Nb,2)
    const float* up,         // (Na,) or NULL
    const float* vp,         // (Na,) or NULL
    const bool is_predicted,
    const int Na,
    const int Nb,
    const float* R2,         // radius squared
    const int maxlen,
    int* counts              // (Na,)
){
    int r = blockIdx.x;                 // region / coords_a index
    if (r >= Na) return;

    // compute predicted coords (apply displacement if requested)
    float ax = coords_a[2*r + 0];
    float ay = coords_a[2*r + 1];
    if (is_predicted) {
        ax += vp[r];
        ay += up[r];
    }

    // threads in block scan coords_b in a strided loop
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int b = tid; b < Nb; b += stride) {
        float bx = coords_b[2*b + 0];
        float by = coords_b[2*b + 1];
        float dx = ax - bx;
        float dy = ay - by;
        float d2 = dx*dx + dy*dy;
        if (d2 <= R2[b]) {
            // append index b to cluster r using atomicAdd
            int pos = atomicAdd(&counts[r], 1);
            if (pos >= maxlen) {
                // overflow: undo increment and ignore this match
                atomicSub(&counts[r], 1);
            }
        }
    }
}
"""

code_get_clusters = """
extern "C" __global__
void cuda_get_clusters(
    const float* coords_a,      // (Na,2)
    const float* coords_b,      // (Nb,2)
    const float* up,            // (Na,) or NULL
    const float* vp,            // (Na,) or NULL
    const bool is_predicted,
    const int Na,
    const int Nb,
    const float* R2,            // radius squared
    const int maxlen,
    int* counts,                // (Na,)
    int* clusters,              // (Na * maxlen,) -> indices of coords_b
    float* cluster_coords       // (Na * maxlen * 2,) -> x,y coords of matches
){
    int r = blockIdx.x;  // region index
    if (r >= Na) return;
    
    // compute predicted coord (apply displacement if requested)
    float ax = coords_a[2*r + 0];
    float ay = coords_a[2*r + 1];
    if (is_predicted) {
        ax += vp[r];
        ay += up[r];
    }
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int b = tid; b < Nb; b += stride) {
        float bx = coords_b[2*b + 0];
        float by = coords_b[2*b + 1];
        float dx = ax - bx;
        float dy = ay - by;
        float d2 = dx*dx + dy*dy;
        
        if (d2 <= R2[b]) {
            int pos = atomicAdd(&counts[r], 1);
            if (pos < maxlen) {
                int base_idx = r * maxlen + pos;
                clusters[base_idx] = b;             // store index
                cluster_coords[2*base_idx + 0] = bx; // store x
                cluster_coords[2*base_idx + 1] = by; // store y
            } else {
                atomicSub(&counts[r], 1);
            }
        }
    }
}
"""

code_update_probs = """
extern "C" __global__
void cuda_update_probs(
    const float* delta,   // (N * max_Nb * 2)
    const float* field,
    const int* Nb,
    const int* Na,
    const int* idx,
    float* P,
    const float* p,
    const float sigma,
    const int N,
    const int max_Nb,
    const int max_Na)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    if (i >= N || j >= Nb[i]) return;
    
    // Compute r[j]
    float dx = delta[(i * max_Nb + j) * 2 + 0];
    float dy = delta[(i * max_Nb + j) * 2 + 1];
    float rj = sqrtf(dx * dx + dy * dy);
    
    float P_sum = 0.0f;
    
    // Loop over neighbors
    for (int k = 0; k < Na[i]; ++k) {
        int kk = idx[i * max_Na + k];
        int Nbk = Nb[kk];
        
        for (int l = 0; l < Nbk; ++l) {
            float dx2 = delta[(kk * max_Nb + l) * 2 + 0];
            float dy2 = delta[(kk * max_Nb + l) * 2 + 1];
            
            float ddx = dx - dx2;
            float ddy = dy - dy2;
            
            float r2 = ddx * ddx + ddy * ddy;
            float W = expf(-r2 / (2.0f * sigma * sigma));
            P_sum += W * p[kk * max_Nb + l];
        }
    }
    
    // Update P entry
    float newP = p[i * max_Nb + j] * (1.0f + P_sum);
    P[i * max_Nb + j] = newP;
}
"""

code_median_validation = """
extern "C" __global__
void cuda_median_validation(
    const float* u,
    const int* nbr_idx,
    const int* nbr_ptr,
    const float tol,
    const int N,
    bool* mask_out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    int start = nbr_ptr[i];
    int end   = nbr_ptr[i + 1];
    int n     = end - start;
    if (n == 0) {
        mask_out[i] = false;
        return;
    }

    // local buffer for neighbor u-values (max 64 neighbors)
    float vals[64];
    if (n > 64) n = 64;  // safeguard

    for (int j = 0; j < n; j++) {
        int idx = nbr_idx[start + j];
        vals[j] = u[idx];
    }

    // sort (simple insertion sort since n small)
    for (int a = 1; a < n; a++) {
        float key = vals[a];
        int b = a - 1;
        while (b >= 0 && vals[b] > key) {
            vals[b + 1] = vals[b];
            b--;
        }
        vals[b + 1] = key;
    }

    // median
    float med = (n % 2 == 1) ? vals[n / 2] : 0.5f * (vals[n/2 - 1] + vals[n/2]);

    // check
    mask_out[i] = fabsf(u[i] - med) > tol;
}


extern "C" __global__
void cuda_median_validation_ptv(
    const float* u,            // velocity component (can run separately for u and v)
    const float* x,            // x-coordinate of particles
    const float* y,            // y-coordinate of particles
    const int* nbr_idx,        // neighbor indices
    const int* nbr_ptr,        // neighbor pointer array
    const float eps_a,         // adaptive tolerance (e.g., 0.1)
    const float r_thresh,      // threshold (e.g., 2.0)
    const int N,
    bool* mask_out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    int start = nbr_ptr[i];
    int end   = nbr_ptr[i + 1];
    int n     = end - start;
    if (n == 0) {
        mask_out[i] = false;
        return;
    }

    // local buffers (max 64 neighbors)
    float vals[64];
    float dists[64];
    if (n > 64) n = 64;

    float x0 = x[i];
    float y0 = y[i];

    // collect neighbor values and distances
    for (int j = 0; j < n; j++) {
        int idx = nbr_idx[start + j];
        vals[j]  = u[idx];
        float dx = x[idx] - x0;
        float dy = y[idx] - y0;
        dists[j] = sqrtf(dx * dx + dy * dy);
    }

    // sort velocities and distances (insertion sort)
    for (int a = 1; a < n; a++) {
        float key_v = vals[a];
        float key_d = dists[a];
        int b = a - 1;
        while (b >= 0 && vals[b] > key_v) {
            vals[b + 1] = vals[b];
            dists[b + 1] = dists[b];
            b--;
        }
        vals[b + 1] = key_v;
        dists[b + 1] = key_d;
    }

    // median of neighbor velocities
    float med_u = (n % 2 == 1) ? vals[n/2] : 0.5f * (vals[n/2 - 1] + vals[n/2]);

    // median distance
    float med_d = (n % 2 == 1) ? dists[n/2] : 0.5f * (dists[n/2 - 1] + dists[n/2]);

    // compute |Ui - med(Ui)|
    for (int j = 0; j < n; j++) {
        vals[j] = fabsf(vals[j] - med_u);
    }

    // sort again to find median absolute deviation
    for (int a = 1; a < n; a++) {
        float key = vals[a];
        int b = a - 1;
        while (b >= 0 && vals[b] > key) {
            vals[b + 1] = vals[b];
            b--;
        }
        vals[b + 1] = key;
    }

    float med_abs = (n % 2 == 1) ? vals[n/2] : 0.5f * (vals[n/2 - 1] + vals[n/2]);

    // compute r0*
    float num = fabsf(u[i] - med_u);
    float denom = med_abs + eps_a * (med_d + eps_a);
    float r0 = num / denom;

    mask_out[i] = (r0 <= r_thresh);
}
"""