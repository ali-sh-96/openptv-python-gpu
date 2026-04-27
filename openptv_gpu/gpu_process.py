"""This module contains algorithms for PTV analysis on an NVIDIA GPU."""

import numpy as np
import cupy as cp
from math import ceil, log2
import cv2

from cupyx.scipy.interpolate import interpn
from cupyx.scipy.ndimage import gaussian_laplace, maximum_filter, label

from . import DTYPE_b, DTYPE_i, DTYPE_u, DTYPE_f
from .gpu_validation import ValidationGPU, Num_VALIDATION_ITERS, VALIDATION_SIZE, MAX_VALIDATION_SIZE
from .gpu_validation import MEDIAN_TOL, MAD_TOL, EPSILON

# Default settings.
PARTICLE_METHOD = "log"
SUBPIXEL_METHOD = "gaussian"
THRESHOLD = 0
PARTICLE_SIZE = 1
SEARCH_SIZE = 8
CLUSTER_SIZE = 8
KERNEL_SIZE = 128
NUM_RELAXATION_ITERS = 1
SIGMA = 1
RELAXATION_METHOD = "unidirectional"
FIELD_TOL = None
BATCH_SIZE = 1
BLOCK_SIZE = 32

# Allowed settings.
ALLOWED_PARTICLE_METHODS = {"log", "agt"}
ALLOWED_SUBPIXEL_METHODS = {"gaussian", "parabolic", "centroid"}
ALLOWED_RELAXATION_METHODS = {"unidirectional", "bidirectional"}

class ptv_gpu:
    """Wrapper-class for PTVGPU that further applies input validation and provides user inetrface.
    
    Parameters
    ----------
    frame_shape : tuple
        Shape of the images in pixels.
    **kwargs
        PTV settings. See PTVGPU.
    
    Attributes
    ----------
    init_coords : tuple of ndarray
        Arrays (x, y) of initially detected particle positions.
    coords : tuple of ndarray
        Arrays (x, y) of matched particle positions after tracking.
    
    """
    def __init__(self, frame_shape, **kwargs):
        particle_method = kwargs["particle_method"] if "particle_method" in kwargs else PARTICLE_METHOD
        subpixel_method = kwargs["subpixel_method"] if "subpixel_method" in kwargs else SUBPIXEL_METHOD
        threshold = kwargs["threshold"] if "threshold" in kwargs else THRESHOLD
        particle_size = kwargs["particle_size"] if "particle_size" in kwargs else PARTICLE_SIZE
        search_size = kwargs["search_size"] if "search_size" in kwargs else SEARCH_SIZE
        cluster_size = kwargs["cluster_size"] if "cluster_size" in kwargs else CLUSTER_SIZE
        kernel_size = kwargs["kernel_size"] if "kernel_size" in kwargs else KERNEL_SIZE
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else BATCH_SIZE
        num_relaxation_iters = kwargs["num_relaxation_iters"] if "num_relaxation_iters" in kwargs else NUM_RELAXATION_ITERS
        relaxation_constant = kwargs["relaxation_constant"] if "relaxation_constant" in kwargs else SIGMA
        relaxation_method = kwargs["relaxation_method"] if "relaxation_method" in kwargs else RELAXATION_METHOD
        num_validation_iters = kwargs["num_validation_iters"] if "num_validation_iters" in kwargs else Num_VALIDATION_ITERS
        validation_size = kwargs["validation_size"] if "validation_size" in kwargs else VALIDATION_SIZE
        max_validation_size = kwargs["max_validation_size"] if "max_validation_size" in kwargs else MAX_VALIDATION_SIZE
        field_tol = kwargs["field_tol"] if "field_tol" in kwargs else FIELD_TOL
        median_tol = kwargs["median_tol"] if "median_tol" in kwargs else MEDIAN_TOL
        mad_tol = kwargs["mad_tol"] if "mad_tol" in kwargs else MAD_TOL
        epsilon = kwargs["epsilon"] if "epsilon" in kwargs else EPSILON
        dt = kwargs["dt"] if "dt" in kwargs else 1
        scaling_par = kwargs["scaling_par"] if "scaling_par" in kwargs else 1
        mask = kwargs["mask"] if "mask" in kwargs else None
        dtype_f = "float32"
        
        # Check the geometry settings.
        self.frame_shape = frame_shape
        assert isinstance(self.frame_shape, tuple) and \
            len(self.frame_shape) == 2 and \
                all(isinstance(item, int) for item in self.frame_shape), \
                    "{} must be a tuple of {} numbers.".format("frame_shape", "int")
        
        self.n_dims = len(self.frame_shape)
        self.particle_method = particle_method
        assert self.particle_method in ALLOWED_PARTICLE_METHODS, \
            "{} must be one of {}.".format("particle_method", ALLOWED_PARTICLE_METHODS)
        
        self.subpixel_method = subpixel_method
        assert self.subpixel_method in ALLOWED_SUBPIXEL_METHODS, \
            "{} must be one of {}.".format("subpixel_method", ALLOWED_SUBPIXEL_METHODS)
        
        self.threshold = (threshold,) * self.n_dims if isinstance(threshold, int) or \
            isinstance(threshold, float) else threshold
        assert isinstance(self.threshold, tuple) and \
            all(isinstance(item, int) or isinstance(item, float) for item in self.threshold), \
                "{} must be a tuple of {} numbers.".format("threshold", "real")
        
        self.particle_size = (particle_size,) * self.n_dims if isinstance(particle_size, int) \
            else particle_size
        assert isinstance(self.particle_size, tuple) and \
            all(isinstance(item, int) and item >= 1 for item in self.particle_size), \
                "{} must be a tuple of {} numbers.".format("particle_size", "integer")
        
        # Check the relaxation settings.
        self.search_size = search_size
        assert isinstance(self.search_size, int) and self.search_size >= 1, \
            "{} must be an {} number greater than zero.".format("search_size", "integer")
        
        self.cluster_size = cluster_size
        assert isinstance(self.cluster_size, int) and self.cluster_size >= 1, \
            "{} must be an {} greater than zero.".format("cluster_size", "integer")
        
        self.kernel_size = kernel_size
        assert isinstance(self.kernel_size, int) and 32 <= self.kernel_size <= 1024 and\
            (self.kernel_size & (self.kernel_size - 1)) == 0,\
                "{} must be a positive power of 2 between 32 and 1024.".format("kernel_size")
        
        self.num_relaxation_iters = num_relaxation_iters
        assert isinstance(self.num_relaxation_iters, int) and self.num_relaxation_iters >= 1, \
            "{} must be an {} greater than zero.".format("num_relaxation_iters", "integer")
        
        self.relaxation_constant = relaxation_constant
        assert (isinstance(self.relaxation_constant, int) or isinstance(self.relaxation_constant, float)) \
            and self.relaxation_constant > 0, \
                "{} must be a {} number greater than zero.".format("relaxation_constant", "real")
        
        self.relaxation_method = relaxation_method
        assert self.relaxation_method in ALLOWED_RELAXATION_METHODS, \
            "{} must be one of {}.".format("relaxation_method", ALLOWED_RELAXATION_METHODS)
        
        # Check the validation settings.
        self.num_validation_iters = num_validation_iters
        assert isinstance(self.num_validation_iters, int) and self.num_validation_iters >= 1, \
            "{} must be an {} greater than zero.".format("num_validation_iters", "integer")
        
        self.validation_size = validation_size
        assert isinstance(self.validation_size, int) and self.validation_size >= 1, \
            "{} must be an {} greater than zero.".format("validation_size", "integer")
        
        self.max_validation_size = max_validation_size
        assert isinstance(self.max_validation_size, int) and self.validation_size >= 1, \
            "{} must be an {} greater than zero.".format("max_validation_size", "integer")
        if self.max_validation_size < self.validation_size:
            self.max_validation_size = self.validation_size
        
        self.field_tol = field_tol
        assert self.field_tol is None or isinstance(self.field_tol, int) or \
            isinstance(self.field_tol, float), \
                    "{} must be a {} number or None.".format("field_tol", "real")
        
        self.median_tol = median_tol
        assert self.median_tol is None or isinstance(self.median_tol, int) or \
            isinstance(self.median_tol, float), \
                "{} must be a {} number or None.".format("median_tol", "real")
        
        self.mad_tol = mad_tol
        assert self.mad_tol is None or isinstance(self.mad_tol, int) or \
            isinstance(self.mad_tol, float), \
                    "{} must be a {} number or None.".format("mad_tol", "real")
        
        self.eps = epsilon
        assert (isinstance(self.mad_tol, int) or isinstance(self.eps, float)) and \
            self.eps >= 0, "{} must be a {} number.".format("epsilon", "positive real")
        
        # Check the scaling settings.
        self.dt = dt
        assert isinstance(self.dt, int) or isinstance(self.dt, float) and self.dt > 0, \
            "{} must be a {} number greater than 0.".format("dt", "real")
        
        self.scaling_par = scaling_par
        assert isinstance(self.scaling_par, int) or isinstance(self.scaling_par, float) and \
            self.scaling_par > 0, \
                "{} must be a {} number greater than 0.".format("scaling_par", "real")
        
        # Check the masking settings.
        self.mask = mask
        assert self.mask is None or \
            (isinstance(self.mask, np.ndarray) and \
             self.mask.shape == self.frame_shape and \
                 (np.issubdtype(self.mask.dtype, np.number) or mask.dtype == bool)), \
                    "{} must be an ndarray of {} values with shape {}.".format("mask", "real", self.frame_shape)
        
        self.mask = mask.astype(bool) if mask is not None else None
        self.frame_mask = self.mask if self.mask is not None else np.full(self.frame_shape, fill_value=False, dtype=bool)
        
        # Data type settings.
        self.dtype_f = DTYPE_f if dtype_f == "float64" else np.float32
        
        # Initialize the process.
        self.gpu_process = PTVGPU(frame_shape, **kwargs)
    
    def __call__(self, frame_a, frame_b, field=None):
        """Computes velocity field from an image pair.
        
        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D arrays containing grey levels of the frames.
        
        field : ndarray
            2D predictor field.
        
        Returns
        -------
        u, v : ndarray
            2D arrays, horizontal/vertical components of the velocity field.
        
        """
        frames = [frame_a, frame_b]

        assert all(isinstance(frame, np.ndarray) for frame in frames) \
            and all(frame.shape == self.frame_shape for frame in frames) \
                and all(np.issubdtype(frame.dtype, np.number) for frame in frames) \
                    and all(not np.iscomplexobj(frame) for frame in frames), \
                        "Both frames must be ndarrays of {} values with shape {}.".format("real", self.frame_shape)
        
        if field is not None:
            assert isinstance(field, tuple) and len(field) == 4 \
                and all(isinstance(f, np.ndarray) for f in field) \
                    and all(f.shape == field[0].shape for f in field) \
                        and all(np.issubdtype(f.dtype, np.number) for f in field) \
                            and all(not np.iscomplexobj(f) for f in field), \
                                "field must be a tuple of four ndarrays (x, y, u, v) of real numbers."
        
        u, v = self.gpu_process(frame_a, frame_b, field=field)
        return u.get(), v.get()
    
    def get_coords(self, frame_a, frame_b):
        """Returns the local particle coordinates."""
        return self.gpu_process.get_coords(frame_a, frame_b, is_gpu=False)
    
    @property
    def init_coords(self):
        """Returns the initial particle coordinates."""
        coords_a, coords_b = self.gpu_process.init_coords
        
        return coords_a, coords_b
    
    @property
    def coords(self):
        """Returns the x and y components of the matched particle coordinates."""
        x, y = self.gpu_process.coords
        
        return x, y

class PTVGPU:
    """Relaxation-based PTV algorithm.
    
    Algorithm Details
    -----------------
    This algorithm estimates an instantaneous two-dimensional velocity field from two consecutive
    image frames using a probabilistic particle-matching approach. For each particle in the first
    frame, candidate matches are identified within a specified search radius in the second frame.
    Match probabilities are iteratively updated using a relaxation scheme. At each iteration,
    the probability of a candidate increases if its displacement is more consistent with those of
    neighboring particles, promoting spatial coherence in the resulting velocity field. Upon
    reaching the maximum number of iterations, the candidate with the highest probability is
    selected as the final match, yielding the particle displacement.
    
    References
    ----------
    Baek, S. J., & Lee, S. J. (1996). A new two-frame particle tracking algorithm using match probability.
        Experiments in Fluids, 22, 23-32.
        https://doi.org/10.1007/BF01893303
    Westerweel, J., Scarano, F. (2005). Universal outlier detection for PIV data.
        Experiments in Fluids, 39, 1096–1100.
        https://doi.org/10.1007/s00348-005-0016-6
    
    Parameters
    ----------
    frame_shape : tuple of int
        Image dimensions in pixels (height, width).
    particle_method : {"log", "agt"}, optional
        Method used for particle detection.
    subpixel_method : {"gaussian", "centroid", "parabolic"}, optional
        Method used for subpixel peak localization.
    threshold : float or tuple, optional
        Threshold for particle detection (Laplacian of Gaussian or adaptive Gaussian).
    particle_size : int or tuple, optional
        Half-size of the kernel used for particle detection.
    search_size : int, optional
        Radius for searching candidate particles in the second frame.
    cluster_size : int, optional
        Radius for identifying neighboring particles in the first frame.
    kernel_size : int, optional
        Maximum number of particles (power of 2) for candidate or neighbor sets.
    batch_size : int, optional
        Batch size for relaxation (not active for the current GPU version).
    num_relaxation_iters : int, optional
        Number of iterations for probability relaxation.
    relaxation_constant : float, optional
        Relaxation parameter (typically ~3) controlling convergence rate.
    relaxation_method : {"unidirectional", "bidirectional"}, optional
        Tracking direction, forward only or bidirectional.
    num_validation_iters : int, optional
        Number of iterations in the validation cycle.
    validation_size : int, optional
        Initial radius for the validation process.
    max_validation_size : int, optional
        Maximum radius for the adaptive validation process.
    field_tol : float or None, optional
        Tolerance for validation by predictor field.
    median_tol : float or None, optional
        Tolerance for median-based velocity validation.
    mad_tol : float or None, optional
        Tolerance for median-absolute-deviation (MAD) validation.
    epsilon : float, optional
        Small constant used in MAD validation (see Westerweel & Scarano, 2005).
    dt : float, optional
        Time delay between frames.
    scaling_par : float, optional
        Scaling factor applied to the velocity field.
    mask : ndarray or None, optional
        2D array where non-zero values indicate masked regions.
    dtype_f : str, optional
        Float data type (not active for the current GPU version).
    
    Attributes
    ----------
    init_coords : tuple of ndarray
        Arrays (x, y) of initially detected particle positions.
    coords : tuple of ndarray
        Arrays (x, y) of matched particle positions after tracking.
    
    """
    def __init__(self,
                 frame_shape,
                 particle_method=PARTICLE_METHOD,
                 subpixel_method=SUBPIXEL_METHOD,
                 threshold=THRESHOLD,
                 particle_size=PARTICLE_SIZE,
                 search_size=SEARCH_SIZE,
                 cluster_size=CLUSTER_SIZE,
                 kernel_size=KERNEL_SIZE,
                 batch_size=BATCH_SIZE,
                 num_relaxation_iters=NUM_RELAXATION_ITERS,
                 relaxation_constant=SIGMA,
                 relaxation_method=RELAXATION_METHOD,
                 num_validation_iters=Num_VALIDATION_ITERS,
                 validation_size=VALIDATION_SIZE,
                 max_validation_size=MAX_VALIDATION_SIZE,
                 field_tol=FIELD_TOL,
                 median_tol=MEDIAN_TOL,
                 mad_tol=MAD_TOL,
                 epsilon=EPSILON,
                 dt=1,
                 scaling_par=1,
                 mask=None,
                 dtype_f=DTYPE_f):
        
        # Geometry settings.
        self.f_shape = frame_shape
        self.n_dims = len(self.f_shape)
        self.particle_method = particle_method
        self.subpixel_method = subpixel_method
        self.threshold = (threshold,) * self.n_dims if isinstance(threshold, int) or \
            isinstance(threshold, float) else threshold
        self.particle_size = (particle_size,) * self.n_dims if isinstance(particle_size, int) \
            else particle_size
        
        # Relaxation settings.
        self.search_size = search_size
        self.cluster_size = cluster_size
        self.kernel_size = kernel_size
        self.n_iters = num_relaxation_iters
        self.sigma = relaxation_constant
        self.relaxation_method = relaxation_method
        
        # Validation settings.
        self.n_ietrs = num_validation_iters
        self.validation_size = validation_size
        self.max_validation_size = max_validation_size
        self.field_tol = field_tol
        self.median_tol = median_tol
        self.mad_tol = mad_tol
        self.eps = epsilon
        
        # Scaling settings.
        self.dt = dt
        self.scaling_par = scaling_par
        
        # Data type settings.
        self.dtype_f = np.float32 if dtype_f == "float32" else DTYPE_f
        self.dtype_i = np.int32 if dtype_f is not DTYPE_f else DTYPE_i
        self.dtype_b = DTYPE_b
        
        # Convert mask to boolean array.
        self.mask = mask.astype(self.dtype_b) if mask is not None else None
        
        # Compile the CUDA kernels.
        self.mod_get_peak = cp.RawModule(code=code_get_peak)
        self.mod_fill_matrix = cp.RawModule(code=code_fill_matrix)
        self.mod_update_probs = cp.RawModule(code=code_update_probs)
        
        # Initialize the field object.
        self.ptv_field = PTVFIELDGPU(self.f_shape,
                                     modules=self.mod_get_peak,
                                     particle_method=self.particle_method,
                                     subpixel_method=self.subpixel_method,
                                     dtype_f=self.dtype_f)
        
        # Initialize the relaxation object.
        self.relaxation = RelaxationGPU(modules=(self.mod_fill_matrix, self.mod_update_probs),
                                        search_size=self.search_size,
                                        cluster_size=self.cluster_size,
                                        kernel_size=self.kernel_size,
                                        num_relaxation_iters=self.n_iters,
                                        sigma=self.sigma,
                                        dtype_f=self.dtype_f)
        
        # Initialize the validation object.
        self.validation = ValidationGPU(size=self.validation_size,
                                        max_size=self.max_validation_size,
                                        kernel_size=self.kernel_size,
                                        median_tol=self.median_tol,
                                        mad_tol=self.mad_tol,
                                        epsilon=self.eps,
                                        dtype_f=self.dtype_f)
    
    def __call__(self, frame_a, frame_b, field=None):
        """Computes velocity field from an image pair.
        
        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D arrays containing grey levels of the frames.
        
        Returns
        -------
        u, v : ndarray
            Arrays of horizontal/vertical components of the velocity field.
        
        """
        self.field = tuple(cp.asarray(f, dtype=self.dtype_f) for f in field) \
            if field is not None else None
        
        # Get the particle coordinates.
        self.coords_a, self.coords_b = self.get_coords(frame_a, frame_b, is_gpu=True)
        
        # Perform relaxation.
        u, v = self.relaxation(ptv_field=self.ptv_field,
                               field=self.field,
                               relaxation_method=self.relaxation_method)
        
        # Validate the fields.
        u, v = self.validate_fields(u, v)
        
        return u, v
    
    def get_coords(self, frame_a, frame_b, is_gpu=False):
        """Returns the local particle coordinates."""
        # Mask the frames.
        self.frame_a, self.frame_b = self.mask_frames(frame_a, frame_b, mask=self.mask)
        
        coords_a, coords_b = self.ptv_field(frame_a, frame_b,
                                            threshold=self.threshold,
                                            particle_size=self.particle_size)
        
        if not is_gpu:
            coords_a, coords_b = coords_a.get(), coords_b.get()
        
        return coords_a, coords_b
    
    def mask_frames(self, frame_a, frame_b, mask=None):
        """Masks the frames."""
        if mask is not None:
            frame_a[mask] = 0
            frame_b[mask] = 0
        
        return frame_a, frame_b
    
    def validate_fields(self, u, v):
        """Returns the validated velocity field with outliers removed."""
        mask = self.validate_by_fields(u, v, field=self.field) \
            if self.field is not None and self.field_tol is not None \
                else self.validation(u, v, self.ptv_field, n_iters=self.n_ietrs)
        
        if mask is not None:
            u = u[mask]
            v = v[mask]
            self.coords_a = self.coords_a[mask]
        
        return u, v
    
    def validate_by_fields(self, u, v, field):
        """Performs validation using the predictor field."""
        xp, yp, up, vp = field
        xp, yp = xp[0, :], yp[:, 0]
        
        # Interpolate field (up, vp) at measured coordinates.
        up = interpn((yp, xp), up, self.coords_a, bounds_error=False, fill_value=None)
        vp = interpn((yp, xp), vp, self.coords_a, bounds_error=False, fill_value=None)
        up, vp = up.astype(self.dtype_f), vp.astype(self.dtype_f)
        
        # Compute norms.
        eps = 1e-8
        norm_u = cp.sqrt(u * u + v * v) + eps
        norm_p = cp.sqrt(up * up + vp * vp) + eps
        
        # Angle residual.
        dot = u * up + v * vp
        cos_theta = dot / (norm_u * norm_p)
        cos_theta = cp.clip(cos_theta, -1.0, 1.0)
        
        # Combined angle and magnitude residuals.
        r_a = cp.arccos(cos_theta)
        r_m = cp.abs(norm_u - norm_p) / norm_p
        r_star = cp.sqrt(r_a**2 + r_m**2)
        
        # Mask based on tolerance.
        mask = r_star < self.field_tol
        
        return mask
    
    @property
    def init_coords(self):
        """Returns the initial particle coordinates."""
        coords_a, coords_b = self.coords_a.get(), self.coords_b.get()
        
        return coords_a, coords_b
    
    @property
    def coords(self, is_gpu=True):
        """Returns the x and y components of the matched particle coordinates."""
        if self.coords_a is not None:
            x = self.coords_a[:, 1]
            y = self.coords_a[:, 0]
        
        if is_gpu:
            x, y = x.get(), y.get()
        
        return x, y

class PTVFIELDGPU:
    """Contains geometric information of PTV field.
    
    Parameters
    ----------
    f_shape : tuple
        Shape of the frames, (ht, wd).
    modules: RawModule
        Raw CUDA kernels for peak and subpixel identification.
    particle_method : {"log", "agt"}, optional
        Method used for particle detection.
    subpixel_method : {"gaussian", "centroid", "parabolic"}, optional
        Method to approximate the subpixel location of the peaks.
    dtype_f : str, optional
        Float data type (not active).
    
    """
    def __init__(self, f_shape,
                 modules,
                 particle_method=PARTICLE_METHOD,
                 subpixel_method=SUBPIXEL_METHOD,
                 dtype_f=DTYPE_f):
        
        self.f_shape = f_shape
        self.ht, self.wd = f_shape
        self.N = self.ht * self.wd
        self.mod_get_peak = modules
        self.particle_method = particle_method
        self.subpixel_method = subpixel_method
        
        # Small value added to denominator for subpixel approximation.
        self.eps = 1e-6
        
        # Data type settings.
        self.dtype_f = dtype_f
        self.dtype_i = np.int32 if dtype_f is not DTYPE_f else DTYPE_i
        self.dtype_u = DTYPE_u
        self.dtype_b = DTYPE_b
    
    def __call__(self, frame_a, frame_b,
                 threshold=THRESHOLD,
                 particle_size=PARTICLE_SIZE):
        """Returns the locations of the subpixel peaks using the specified size and threshold.
        
        Parameters
        ----------
        frame_a, frame_b : ndarray
            Image pair.
        threshold : tuple, optional
            Threshold for particle detection.
        particle_size : tuple, optional
            Half-size of the kernel used for particle detection.
        
        Returns
        -------
        coords_a, coords_b : ndarray
            Arrays, containing image coordinates of the detected particles.
        
        """
        threshold_a, threshold_b = threshold
        
        if self.particle_method == "log":
            size_a, size_b = particle_size
        else:
            kernel_size = tuple(2 * size + 1 for size in particle_size)
            size_a, size_b = kernel_size
        
        coords_a, mask_a = self.get_coords(frame_a,
                                           f_min=frame_a.min(),
                                           f_max=frame_a.max(),
                                           size=size_a,
                                           C=threshold_a)
        
        coords_b, mask_b = self.get_coords(frame_b,
                                           f_min=frame_b.min(),
                                           f_max=frame_b.max(),
                                           size=size_b,
                                           C=threshold_b)
        
        self.mask_a, self.mask_b = mask_a, mask_b
        self.coords_a, self.coords_b = coords_a, coords_b
        self.Na = cp.count_nonzero(~mask_a).get()
        self.Nb = cp.count_nonzero(~mask_b).get()
        self.offset_a = cp.cumsum(mask_a, dtype=self.dtype_i)
        self.offset_b = cp.cumsum(mask_b, dtype=self.dtype_i)
        return coords_a[~mask_a], coords_b[~mask_b]
    
    def get_labels_agt(self, f, f_min, f_max, kernel_size=3, C=0):
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
    
    def get_labels_log(self, f, f_min, f_max, size=1, C=0):
        """Performs a single-scale Laplacian of Gaussian blob detection."""
        # Normalize the image.
        f = cp.asarray(f, dtype=self.dtype_f)
        f = (f - f_min) / (f_max - f_min + 1e-8)
        
        # Laplacian of Gaussian.
        sigma = size
        log = gaussian_laplace(f, sigma=sigma)
        
        # Scale normalization.
        log = cp.abs(log) * sigma**2
        
        # Perform thresholding.
        mask = log > C
        
        # Local maxima in 3×3 window.
        peak = maximum_filter(log, size=3)
        mask &= (log == peak)
        
        # Connected-component labeling on GPU.
        labels, n_labels = label(mask)
        
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
        x_peak = x_peak[mask]
        y_peak = y_peak[mask]
        
        return y_peak, x_peak
    
    def get_subpixel(self, f, y_peak, x_peak):
        """Returns the subpixel estimation of the peak locations."""
        f = cp.asarray(f)
        
        # Ensure indices are not on the border.
        mask = (y_peak > 0) & (y_peak < self.ht - 1) & (x_peak > 0) & (x_peak < self.wd - 1)
        yc = y_peak[mask]
        xc = x_peak[mask]
        
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
            mask = (fc > 0) & (fd > 0) & (fu > 0) & (fl > 0) & (fr > 0)
            yc, xc = yc[mask], xc[mask]
            fc, fd, fu, fl, fr = fc[mask], fd[mask], fu[mask], fl[mask], fr[mask]
            
            fc, fd, fu, fl, fr = cp.log(fc), cp.log(fd), cp.log(fu), cp.log(fl), cp.log(fr)
            y_sub = yc + 0.5 * (fd - fu) / (fd - 2.0 * fc + fu + self.eps)
            x_sub = xc + 0.5 * (fl - fr) / (fl - 2.0 * fc + fr + self.eps)
        
        elif self.subpixel_method == "parabolic":
            y_sub = yc + 0.5 * (fd - fu) / (fd - 2.0 * fc + fu + self.eps)
            x_sub = xc + 0.5 * (fl - fr) / (fl - 2.0 * fc + fr + self.eps)
        
        elif self.subpixel_method == "centroid":
            y_sub = yc + (fu - fd) / (fd + fc + fu + self.eps)
            x_sub = xc + (fr - fl) / (fl + fc + fr + self.eps)
        
        # Remove invalid subpixel approximations.
        mask = (y_sub >= 0) & (y_sub <= self.ht - 1) & (x_sub >= 0) & (x_sub <= self.wd - 1)
        mask = ~mask | cp.isinf(y_sub) | cp.isinf(x_sub)
        y_sub[mask] = yc[mask]
        x_sub[mask] = xc[mask]
        
        # Fill arrays that hold the coordinates.
        mask = cp.full((self.ht, self.wd), fill_value=True, dtype=self.dtype_b)
        coords = cp.full((self.ht, self.wd, 2), fill_value=cp.nan, dtype=self.dtype_f)
        y_peak, x_peak = y_sub.astype(self.dtype_i), x_sub.astype(self.dtype_i)
        mask[y_peak, x_peak] = False
        coords[y_peak, x_peak] = cp.column_stack((y_sub, x_sub))
        
        return coords, mask
    
    def get_coords(self, f, f_min, f_max, size=1, C=0):
        """Returns the image coordinates of the detected particles."""
        # Get the labels using the specified method.
        if self.particle_method == "agt":
            labels, n_labels = self.get_labels_agt(f, f_min, f_max, kernel_size=size, C=C)
        elif self.particle_method == "log":
            labels, n_labels = self.get_labels_log(f, f_min, f_max, size=size, C=C)
        
        # Get the peak locations inside each label.
        y_peak, x_peak = self.get_peak(f, labels, n_labels)
        return self.get_subpixel(f, y_peak, x_peak)

class RelaxationGPU:
    """Performs particle matching using probability relaxation.
    
    Parameters
    ----------
    modules: RawModule
        Raw CUDA kernels for neighbor search, candidate selection, and probability updates.
    search_size : int, optional
        Radius for searching candidate particles.
    cluster_size : int, optional
        Radius for identifying neighboring particles.
    kernel_size : int, optional
        Maximum number of particles (power of 2) for candidate or neighbor sets.
    num_relaxation_iters : int, optional
        Number of iterations for probability relaxation.
    sigma : float, optional
        Relaxation parameter (typically ~3) controlling convergence rate.
    dtype_f : str, optional
        Float data type (not active).
    
    Attributes
    ----------
    coords : tuple of ndarray
        Arrays (x, y) of matched particle positions after tracking.
    
    """
    def __init__(self, modules,
                 search_size=SEARCH_SIZE,
                 cluster_size=CLUSTER_SIZE,
                 kernel_size=KERNEL_SIZE,
                 num_relaxation_iters=NUM_RELAXATION_ITERS,
                 sigma=SIGMA,
                 dtype_f=DTYPE_f):
        
        self.search_size = search_size
        self.cluster_size = cluster_size
        self.kernel_size = kernel_size
        self.n_iters = num_relaxation_iters
        self.sigma = sigma
        
        (self.mod_fill_matrix,
         self.mod_update_probs) = modules
        
        # Settings for float and int data types.
        self.dtype_f = dtype_f
        self.dtype_i = np.int32 if dtype_f is not DTYPE_f else DTYPE_i
        self.dtype_b = DTYPE_b
    
    def __call__(self, ptv_field, field=None, relaxation_method=RELAXATION_METHOD):
        """Returns the displacement field using the specified direction.
        
        Parameters
        ----------
        ptv_field : PTVFieldGPU
            Geometric information for the particle tracking field.
        field : tuple or None, optional
            Predictor field used to shift the search region.
        relaxation_method : {"unidirectional", "bidirectional"}, optional
            Tracking direction, forward only or bidirectional.
        
        Returns
        -------
        u, v : ndarray
            2D arrays, displacement components of the matched particle coordinates.
        
        """
        self.ptv_field = ptv_field
        self.field = field
        self.relaxation_method = relaxation_method
        
        # Forward particle matching.
        field = self.interpolate_field(self.ptv_field.coords_a,
                                       self.ptv_field.mask_a,
                                       self.field,
                                       direction="forward")
        
        self.N = self.ptv_field.Na
        self.offset = self.ptv_field.offset_a
        ua, va = self.match_particles(self.ptv_field.coords_a,
                                      self.ptv_field.coords_b,
                                      self.ptv_field.mask_a,
                                      self.ptv_field.mask_b,
                                      self.cluster_size,
                                      self.search_size,
                                      field=field)
        
        # Perform bidirectional validation on CPU.
        if self.relaxation_method == "bidirectional":
            # Backward particle matching.
            field = self.interpolate_field(self.ptv_field.coords_b,
                                           self.ptv_field.mask_b,
                                           self.field,
                                           direction="backward")
            
            self.N = self.ptv_field.Nb
            self.offset = self.ptv_field.offset_b
            ub, vb = self.match_particles(self.ptv_field.coords_b,
                                          self.ptv_field.coords_a,
                                          self.ptv_field.mask_b,
                                          self.ptv_field.mask_a,
                                          self.cluster_size,
                                          self.search_size,
                                          field=field)
            
            # Perform bidirectional validation.
            u, v = self.bidirectional_validation(ua, va, ub, vb)
        else:
            u, v = ua, va
        
        return u, v
    
    def interpolate_field(self, coords, mask, field, direction="forward"):
        """Interpolates the predictor field onto particle coordinates."""
        if field is not None:
            coords = coords[~mask]
            x, y, u, v = field
            field = cp.zeros((self.ptv_field.ht, self.ptv_field.wd, 2), dtype=self.dtype_f)
            
            x_grid = x[0, :]
            y_grid = y[:, 0]
            
            up = interpn((y_grid, x_grid), u, coords, bounds_error=False, fill_value=None)
            vp = interpn((y_grid, x_grid), v, coords, bounds_error=False, fill_value=None)
            up, vp = up.astype(self.dtype_f), vp.astype(self.dtype_f)
            
            if direction == "backward":
                up, vp = -up, -vp
            
            field[~mask] = cp.column_stack((vp, up))
        
        return field
    
    def get_candidates(self, coords_a, coords_b, mask_a, mask_b, Ra, Rb, field=None):
        """Groups neighboring and candidate particles into clusters."""
        block_size = BLOCK_SIZE
        
        # Create dummy variables if field not given.
        is_predicted = True
        if field is None:
            field = cp.zeros((1,), dtype=self.dtype_f)
            is_predicted = False
        
        Na = cp.ones((self.N,), dtype=self.dtype_i)
        indices = cp.full((self.N, self.kernel_size), -1, dtype=self.dtype_i)
        
        window_size = 2 ** ceil(log2(2 * Ra))
        grid_size = ceil(window_size / block_size)
        cuda_get_neighbors = self.mod_fill_matrix.get_function('cuda_get_neighbors')
        cuda_get_neighbors((self.ptv_field.N, grid_size, grid_size), (1, block_size, block_size),
                           (coords_a,
                            mask_a,
                            self.offset,
                            self.dtype_i(self.ptv_field.ht),
                            self.dtype_i(self.ptv_field.wd),
                            self.dtype_i(Ra),
                            self.dtype_i(window_size),
                            self.dtype_i(self.kernel_size),
                            self.dtype_i(self.ptv_field.N),
                            Na,
                            indices))
        
        N0 = cp.zeros((self.N,), dtype=self.dtype_i)
        u = cp.full((self.N, self.kernel_size), cp.nan, dtype=self.dtype_f)
        v = cp.full((self.N, self.kernel_size), cp.nan, dtype=self.dtype_f)
        window_size = 2 ** ceil(log2(2 * Rb))
        grid_size = ceil(window_size / block_size)
        cuda_get_candidates = self.mod_fill_matrix.get_function('cuda_get_candidates')
        cuda_get_candidates((self.ptv_field.N, grid_size, grid_size), (1, block_size, block_size),
                            (coords_a,
                             coords_b,
                             mask_a,
                             mask_b,
                             self.offset,
                             self.dtype_b(is_predicted),
                             field,
                             self.dtype_i(self.ptv_field.ht),
                             self.dtype_i(self.ptv_field.wd),
                             self.dtype_i(Rb),
                             self.dtype_i(window_size),
                             self.dtype_i(self.kernel_size),
                             self.dtype_i(self.ptv_field.N),
                             N0,
                             u, v))
        
        Nb = cp.zeros((self.N, self.kernel_size), dtype=self.dtype_i)
        grid_size = ceil(self.kernel_size / block_size)
        cuda_count_candidates = self.mod_fill_matrix.get_function('cuda_count_candidates')
        cuda_count_candidates((self.N, grid_size), (1, block_size),
                              (self.dtype_i(self.kernel_size),
                               self.dtype_i(self.N),
                               Na,
                               indices,
                               N0,
                               Nb))
        
        delta = cp.full((self.N, self.kernel_size, self.kernel_size, 2), cp.nan, dtype=self.dtype_f)
        cuda_fill_matrix = self.mod_fill_matrix.get_function('cuda_fill_matrix')
        cuda_fill_matrix((self.N, grid_size, grid_size), (1, block_size, block_size),
                         (self.dtype_i(self.kernel_size),
                          self.dtype_i(self.N),
                          Na,
                          Nb,
                          indices,
                          u,
                          v,
                          delta))
        
        if int(Na.max()) >= self.kernel_size or int(Nb.max()) >= self.kernel_size:
            raise ValueError("Overflow detected: kernel_size is too small.")
        
        return delta, indices, Nb, Na
    
    def update_probs(self, delta, indices, Nb, Na, n_iters=0, sigma=1):
        """Updates the probability matrix using relaxation."""
        block_size = BLOCK_SIZE
        grid_size = ceil(self.kernel_size / block_size)
        
        P = cp.zeros((self.N, self.kernel_size, self.kernel_size), dtype=self.dtype_f)
        cuda_init_probs = self.mod_update_probs.get_function('cuda_init_probs')
        cuda_init_probs((self.N, grid_size, grid_size), (1, block_size, block_size),
                        (self.dtype_i(self.kernel_size),
                         self.dtype_i(self.N),
                         Na,
                         Nb,
                         P))
        
        cuda_update_probs = self.mod_update_probs.get_function('cuda_update_probs')
        cuda_reset_probs = self.mod_update_probs.get_function('cuda_reset_probs')
        for k in range(n_iters):
            cuda_update_probs((self.N, 1), (1, self.kernel_size),
                              (delta,
                               Nb,
                               Na,
                               self.dtype_i(self.N),
                               self.dtype_f(sigma),
                               self.dtype_i(self.kernel_size),
                               P))
            
            P0 = P[:, 0, :].copy()
            P0 /= cp.sum(P0, axis=1, keepdims=True)
            cuda_reset_probs((self.N, grid_size, grid_size), (1, block_size, block_size),
                             (self.dtype_i(self.kernel_size),
                              self.dtype_i(self.N),
                              Na,
                              Nb,
                              indices,
                              P0,
                              P))
        
        return P[:, 0, :]
    
    def match_particles(self, coords_a, coords_b, mask_a, mask_b, Ra, Rb, field=None):
        """Performs particle matching using probability relaxation."""
        # Clustering coordinates in instance a and b.
        delta, indices, Nb, Na = self.get_candidates(coords_a,
                                                     coords_b,
                                                     mask_a,
                                                     mask_b,
                                                     Ra,
                                                     Rb,
                                                     field=field)
        
        P = self.update_probs(delta, indices, Nb, Na, n_iters=self.n_iters, sigma=self.sigma)
        
        # Return displacement with maximum probability.
        u, v = self.get_displacement(P, delta)
        
        return u, v
    
    def bidirectional_validation(self, ua, va, ub, vb):
        """Performs bidirectional validation."""
        # Forward and backward vectors
        forward = cp.column_stack((ua, va))
        backward = cp.column_stack((-ub, -vb))
        
        mask = ~cp.all(cp.isin(forward, backward), axis=1)
        ua[mask], va[mask] = cp.nan, cp.nan
        return ua, va
    
    def get_displacement(self, P, delta):
        """Returns the displacement of the matched particles."""
        j_peak = cp.argmax(P, axis=1)
        i_peak = cp.arange(self.N)
        
        u, v = delta[:, 0, :, 1], delta[:, 0, :, 0]
        u = u[i_peak, j_peak]
        v = v[i_peak, j_peak]
        
        # Ensure the peak values are unique.
        P_peak = cp.max(P, axis=1, keepdims=True)
        n_peak = cp.sum(P == P_peak, axis=1)
        mask = n_peak != 1
        u[mask], v[mask] = cp.nan, cp.nan
        return u, v
    
    @property
    def coords(self):
        """Returns the x and y components of the matched particle coordinates."""
        x = self.ma[:, 1] if self.ma is not None else None
        y = self.ma[:, 0] if self.ma is not None else None
        
        return x, y

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

code_fill_matrix = """
extern "C" __global__
void cuda_get_neighbors(
    const float* coords,
    const bool* mask,
    const int* offset,
    const int ht,
    const int wd,
    const int R,
    const int window_size,
    const int kernel_size,
    const int N,
    int* counts,
    int* indices
)
{
    // x blocks are nodes, and y and z blocks are dimensions.
    int pos = blockIdx.x;
    int j_wins = blockIdx.y * blockDim.y + threadIdx.y;
    int i_wins = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Ensure indices are valid.
    if (pos >= N || j_wins >= window_size || i_wins >= window_size) return;
    if (mask[pos]) return;
    
    // Get the coordinates.
    float xa = coords[2 * pos + 1];
    float ya = coords[2 * pos + 0];
    
    // Map the indices.
    int j = (int) xa + j_wins - window_size / 2;
    int i = (int) ya + i_wins - window_size / 2;
    
    // Ensure all the indices are inside the domain.
    if (j < 0 || j >= wd || i < 0 || i >= ht) return;
    int idx = i * wd + j;
    if (mask[idx]) return;
    
    // Get the candidate coordinates.
    float xb = coords[2 * idx + 1];
    float yb = coords[2 * idx + 0];
    
    // Ensure the neighbor is inside the circular kernel.
    float dx = xb - xa;
    float dy = yb - ya;
    if (dx * dx + dy * dy > R * R) return;
    
    // Use atomic to count the valid neighbors.
    int k;
    pos -= offset[pos];
    if (yb == ya && xb == xa) {k = 0;} else {
        k = atomicAdd(&counts[pos], 1);
    }
    
    // Fill the output array.
    int ik = pos * kernel_size + k;
    indices[ik] = idx - offset[idx];
}

extern "C" __global__
void cuda_get_candidates(
    const float* coords_a,
    const float* coords_b,
    const bool* mask_a,
    const bool* mask_b,
    const int* offset,
    const bool is_predicted,
    float* field,
    const int ht,
    const int wd,
    const int R,
    const int window_size,
    const int kernel_size,
    const int N,
    int* Nb,
    float* u,
    float* v
)
{
    // x blocks are nodes, and y and z blocks are dimensions.
    int pos = blockIdx.x;
    int j_wins = blockIdx.y * blockDim.y + threadIdx.y;
    int i_wins = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Ensure indices are valid.
    if (pos >= N || j_wins >= window_size || i_wins >= window_size) return;
    if (mask_a[pos]) return;
    
    // Get the coordinates.
    float xa = coords_a[2 * pos + 1];
    float ya = coords_a[2 * pos + 0];
    
    // Shift the center coordinates.
    float xc = xa;
    float yc = ya;
    if (is_predicted) {
        xc += field[2 * pos + 1];
        yc += field[2 * pos + 0];
    }
    
    // Map the indices.
    int j = (int) xc + j_wins - window_size / 2;
    int i = (int) yc + i_wins - window_size / 2;
    
    // Ensure all the indices are inside the domain.
    if (j < 0 || j >= wd || i < 0 || i >= ht) return;
    int idx = i * wd + j;
    if (mask_b[idx]) return;
    
    // Get the candidate coordinates.
    float xb = coords_b[2 * idx + 1];
    float yb = coords_b[2 * idx + 0];
    
    // Ensure the neighbor is inside the circular kernel.
    float dx = xb - xc;
    float dy = yb - yc;
    if (dx * dx + dy * dy > R * R) return;
    
    // Use atomic to count the valid candidates.
    pos -= offset[pos];
    int k = atomicAdd(&Nb[pos], 1);
    
    // Fill the output arrays.
    int ik = pos * kernel_size + k;
    u[ik] = xb - xa;
    v[ik] = yb - ya;
}

extern "C" __global__
void cuda_count_candidates(
    const int kernel_size,
    const int N,
    const int* Na,
    const int* idx,
    const int* N0,
    int* Nb
)
{
    // i is node, k is neighbor, and l is candidate indices.
    int i = blockIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Ensure indices are valid.
    if (i >= N) return;
    if (k >= Na[i]) return;
    
    // Fill the array.
    int ik = i * kernel_size + k;
    Nb[ik] = N0[idx[ik]];
}

extern "C" __global__
void cuda_fill_matrix(
    const int kernel_size,
    const int N,
    const int* Na,
    const int* Nb,
    const int* idx,
    const float* u,
    const float* v,
    float* delta
)
{
    // i is node, k is neighbor, and l is candidate indices.
    int i = blockIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Ensure indices are valid.
    if (i >= N) return;
    if (k >= Na[i]) return;
    int ik = i * kernel_size + k;
    if (l >= Nb[ik]) return;
    
    // Fill the displacement array.
    int ij = idx[ik] * kernel_size + l;
    int kl = ik * kernel_size + l;
    delta[2 * kl + 1] = u[ij];
    delta[2 * kl + 0] = v[ij];
}
"""

code_update_probs = """
extern "C" __global__
void cuda_init_probs(
    const int kernel_size,
    const int N,
    const int* Na,
    const int* Nb,
    float* P
)
{
    // i is particle, k is neighbor, and l is candidate indices.
    int i = blockIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Ensure indices are valid.
    if (i >= N) return;
    if (k >= Na[i]) return;
    int ik = i * kernel_size + k;
    if (l > Nb[ik]) return;
    
    // Fill the probability array.
    int kl = ik * kernel_size + l;
    P[kl] = 1.0f / (Nb[ik] + 1);
}

extern "C" __global__
void cuda_update_probs(
    const float* delta,
    const int* Nb,
    const int* Na,
    const int N,
    const float sigma,
    const int kernel_size,
    float* P
){
    // i is node, k is neighbor, and j and l are candidate indices.
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ii = i * kernel_size;
    
    // Ensure indices are valid.
    if (i >= N) return;
    if (j >= Nb[ii]) return;
    
    // Get the candidate displacement.
    int ij = ii * kernel_size + j;
    float dx_ij = delta[2 * ij + 1];
    float dy_ij = delta[2 * ij + 0];
    
    // Loop over neighbors and candidates.
    int ik;
    int kl;
    float P_kl = 0.0f;
    
    for (int k = 1; k < Na[i]; ++k) {
        ik = i * kernel_size + k;
        for (int l = 0; l < Nb[ik]; ++l) {
            kl = ik * kernel_size + l;
            float dx_kl = delta[2 * kl + 1];
            float dy_kl = delta[2 * kl + 0];
            
            float dx = dx_ij - dx_kl;
            float dy = dy_ij - dy_kl;
            float d2 = dx * dx + dy * dy;
            
            float W = expf(-d2 / (2.0f * sigma * sigma));
            P_kl += W * P[kl];
        }
    }
    
    // Update the candidate probability.
    P[ij] *= (1.0f + P_kl);
}

extern "C" __global__
void cuda_reset_probs(
    const int kernel_size,
    const int N,
    const int* Na,
    const int* Nb,
    const int* idx,
    const float* P0,
    float* P
)
{
    // i is node, k is neighbor, and l is candidate indices.
    int i = blockIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Ensure indices are valid.
    if (i >= N) return;
    if (k >= Na[i]) return;
    int ik = i * kernel_size + k;
    if (l >= Nb[ik]) return;
    
    // Fill the probability array.
    int ij = idx[ik] * kernel_size + l;
    int kl = ik * kernel_size + l;
    P[kl] = P0[ij];
}
"""