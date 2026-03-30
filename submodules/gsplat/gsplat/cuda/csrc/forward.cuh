#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
);


__global__ void rasterize_forward_sum(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3* __restrict__ background
);
