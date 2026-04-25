"""This module contains GPU-accelerated validation algorithms."""

import numpy as np
import cupy as cp
from math import ceil, log2
from cupy.linalg import norm

from . import DTYPE_i, DTYPE_f

# Default validation settings.
Num_VALIDATION_ITERS = 1
VALIDATION_SIZE = 1
MAX_VALIDATION_SIZE = 1
MEDIAN_TOL = None
MAD_TOL = 2
EPSILON = 0
BLOCK_SIZE = 8
KERNEL_SIZE = 128

class ValidationGPU:
    """Validates velocity and returns an array indicating which locations need to be removed.
    
    Parameters
    ----------
    size : int, optional
        Initial radius for the validation process.
    max_size : int, optional
        Maximum radius for the adaptive validation process.
    kernel_size : int, optional
        Maximum number of particles (power of 2) for neighbor sets.
    median_tol : float or None, optional
        Tolerance for median-based velocity validation.
    mad_tol : float or None, optional
        Tolerance for median-absolute-deviation (MAD) validation.
    epsilon : float, optional
        Small constant used in MAD validation.
    dtype_f : str, optional
        Float data type (not active).
    
    """
    def __init__(self, size=VALIDATION_SIZE,
                 max_size=MAX_VALIDATION_SIZE,
                 kernel_size=KERNEL_SIZE,
                 median_tol=MEDIAN_TOL,
                 mad_tol=MAD_TOL,
                 epsilon=EPSILON,
                 dtype_f=DTYPE_f):
        
        self.mod_fill_kernel = cp.RawModule(code=code_fill_kernel)
        
        self.validation_tols = {"median": median_tol, "mad": mad_tol}
        self.eps = epsilon
        self.size = size
        self.max_size = max_size
        self.kernel_size = kernel_size
        
        self.dtype_f = dtype_f
        self.dtype_i = np.int32 if dtype_f is not DTYPE_f else DTYPE_i
        self.init_data()
    
    def __call__(self, u, v, ptv_field, n_iters=Num_VALIDATION_ITERS):
        """Returns an array indicating which positions need to be removed.
        
        Parameters
        ----------
        u, v : ndarray
            Input velocity field to be validated.
        ptv_field : PTVFieldGPU
            Geometric information for the particle tracking field.
        n_iters : int, optional
            Number of iterations in the validation cycle.
        
        Returns
        -------
        mask : ndarray
            2D boolean array of locations that need to be removed.
        
        """
        self.init_data()
        self.ptv_field = ptv_field
        self.n_iters = n_iters
        
        if not all(tol is None for tol in self.validation_tols.values()):
            delta = cp.full((self.ptv_field.ht, self.ptv_field.wd, 2), fill_value=cp.nan, dtype=self.dtype_f)
            delta[~self.ptv_field.mask_a] = cp.column_stack((v, u))
            self.f = [v, u, norm(delta)]
            self.n_fields = 3
            self.fk = self.fill_kernel(delta)
        
        # Perform the validations.
        mask = {}
        if self.validation_tols["median"] is not None:
            mask["median"] = self.median_validation()
        if self.validation_tols["mad"] is not None:
            mask["mad"] = self.mad_validation()
        
        # Get all mask locations.
        mask = None if not mask else ~cp.any(cp.stack(list(mask.values()), axis=0), axis=0)
        
        return mask
    
    def init_data(self):
        """Initializes the field variables."""
        self.f = None
        self.f_median = None
        self.f_mad = None
    
    def median_validation(self):
        """Performs median validation on each field."""
        self.f_median = self.get_stats(method="median")
        median_tol = self.validation_tols["median"]
        mask = [abs(self.f[k] - self.f_median[k]) > median_tol for k in range(self.n_fields)]
        return cp.any(cp.stack(mask, axis=0), axis=0)
    
    def mad_validation(self):
        """Performs MAD validation on each field."""
        self.f_median = self.get_stats(method="median") if self.f_median is None else self.f_median
        self.f_mad = self.get_stats(method="mad")
        mad_tol = self.validation_tols["mad"]
        mask = [abs(self.f[k] - self.f_median[k]) > mad_tol * self.f_mad[k] for k in range(self.n_fields)]
        return cp.any(cp.stack(mask, axis=0), axis=0)
    
    def get_stats(self, method="median"):
        """Returns a field containing the statistics of the neighbouring points in each cluster."""
        fm = [cp.full(self.ptv_field.Na, fill_value=cp.nan, dtype=self.dtype_f) for k in range(self.n_fields)]
        
        # Select the method.
        if method=="median":
            mf = cp.nanmedian
        else:
            mf = self.nanmad
        
        # Get the statistics.
        self.fk[:, 0, :] = cp.nan
        for k in range(self.n_fields):
            fm[k] = mf(self.fk[:, :, k], axis=(1, 2))
        return fm
    
    def fill_kernel(self, delta):
        """Identifies neighboring particles and populates the neighbor velocity matrix."""
        Na = cp.ones((self.ptv_field.Na,), dtype=self.dtype_i)
        f = cp.full((self.ptv_field.Na, self.kernel_size, self.n_fields), cp.nan, dtype=self.dtype_f)
        R = cp.full((self.ptv_field.Na,), fill_value=self.size, dtype=self.dtype_i)
        
        block_size = BLOCK_SIZE
        size = self.size
        cuda_fill_kernel = self.mod_fill_kernel.get_function('cuda_fill_kernel')
        
        # Iteratively increase the neighborhood radius.
        for _ in range(self.n_iters):
            window_size = 2 ** ceil(log2(2 * size))
            grid_size = ceil(window_size / block_size)
            cuda_fill_kernel((self.ptv_field.N, grid_size, grid_size), (1, block_size, block_size),
                             (self.ptv_field.coords_a,
                              self.ptv_field.mask_a,
                              self.ptv_field.offset_a,
                              delta,
                              self.dtype_i(self.ptv_field.ht),
                              self.dtype_i(self.ptv_field.wd),
                              R,
                              self.dtype_i(window_size),
                              self.dtype_i(self.kernel_size),
                              self.dtype_i(self.ptv_field.N),
                              Na,
                              f))
            R[Na < self.max_size] += 1
            size += 1
        
        return f
    
    def nanmad(self, f, axis=(1, 2)):
        """Returns the median-absolute-deviation of a 3D array."""
        f_median = cp.nanmedian(f, axis=axis, keepdims=True)
        return cp.nanmedian(abs(f - f_median), axis=axis) + self.eps

code_fill_kernel = """
extern "C" __global__
void cuda_fill_kernel(
    const float* coords,
    const bool* mask,
    const int* offset,
    float* fi,
    const int ht,
    const int wd,
    const int* R,
    const int window_size,
    const int kernel_size,
    const int N,
    int* Na,
    float* f
)
{
    // x blocks are particles, and y and z blocks are dimensions.
    int k_wins = blockIdx.x;
    int j_wins = blockIdx.y * blockDim.y + threadIdx.y;
    int i_wins = blockIdx.z * blockDim.z + threadIdx.z;
    int k_offset = k_wins - offset[k_wins];
    if (Na[k_offset] >= 20) return;
    
    // Ensure indices are valid.
    if (k_wins >= N || j_wins >= window_size || i_wins >= window_size) return;
    if (mask[k_wins]) return;
    
    
    
    // Map the indices.
    int r = R[k_offset];
    int j = k_wins % wd;
    int i = k_wins / wd;
    j_wins -= r;
    i_wins -= r;
    j += j_wins;
    i += i_wins;
    
    // Ensure all the indices are inside the domain.
    if (j < 0 || j >= wd || i < 0 || i >= ht) return;
    int idx = i * wd + j;
    if (mask[idx]) return;
    
    // Get the center and neighbor vectors.
    float xa = coords[2 * k_wins + 1];
    float ya = coords[2 * k_wins + 0];
    float xb = coords[2 * idx + 1];
    float yb = coords[2 * idx + 0];
    
    // Ensure the neighbor is inside the circular kernel.
    float dx = xb - xa;
    float dy = yb - ya;
    if (dx * dx + dy * dy > r * r) return;
    
    // Use atomic to count the valid neighbors.
    int k;
    if (j_wins == 0 && i_wins == 0) {k = 0;} else {
        k = atomicAdd(&Na[k_offset], 1);
    }
    
    // Fill the output array.
    int ik = k_offset * kernel_size + k;
    float u = fi[2 * idx + 1];
    float v = fi[2 * idx + 0];
    
    f[3 * ik + 0] = v;
    f[3 * ik + 1] = u;
    f[3 * ik + 2] = sqrt(u * u + v * v);
}
"""