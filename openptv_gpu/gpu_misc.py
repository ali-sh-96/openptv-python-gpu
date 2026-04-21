"""This module contains miscellaneous functions for GPU-accelerated PTV processing."""

code_fill_kernel = """
extern "C" __global__
void cuda_fill_kernel(
    const float* coords,
    const bool* mask,
    const int* k_offset,
    const bool is_predicted,
    float* field,
    const int ht,
    const int wd,
    const int R,
    const int window_size,
    const int kernel_size,
    const int N,
    int* Na,
    int* indices,
    float* kernel,
    float* f
)
{
    // x blocks are particles, and y and z blocks are dimensions.
    int k_wins = blockIdx.x;
    int j_wins = blockIdx.y * blockDim.y + threadIdx.y;
    int i_wins = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Ensure indices are valid.
    if (k_wins >= N || j_wins >= window_size || i_wins >= window_size) return;
    if (mask[k_wins]) return;
    
    // Map the indices.
    int j = k_wins % wd;
    int i = k_wins / wd;
    j_wins -= R;
    i_wins -= R;
    j += j_wins;
    i += i_wins;
    
    // Ensure all the indices are inside the domain.
    if (j < 0 || j >= wd || i < 0 || i >= ht) return;
    int idx = i * wd + j;
    if (mask[idx]) return;
    
    // Get the center and neighbor coordinates.
    float xa = coords[2 * k_wins + 1];
    float ya = coords[2 * k_wins + 0];
    float xb = coords[2 * idx + 1];
    float yb = coords[2 * idx + 0];
    
    // Ensure the neighbor is inside the circular kernel.
    float dx = xb - xa;
    float dy = yb - ya;
    if (dx * dx + dy * dy > R * R) return;
    
    // Use atomic to count the valid neighbors.
    int k;
    k_wins -= k_offset[k_wins];
    if (j_wins == 0 && i_wins == 0) {k = 0;} else {
        k = atomicAdd(&Na[k_wins], 1);
    }
    
    // Fill the output arrays.
    int ik = k_wins * kernel_size + k;
    kernel[2 * ik + 1] = xb;
    kernel[2 * ik + 0] = yb;
    indices[ik] = idx - k_offset[idx];
    
    if (is_predicted) {
        f[2 * ik + 1] = field[2 * idx + 1];
        f[2 * ik + 0] = field[2 * idx + 0];
    }
}
"""