#include "forward2d.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
namespace cg = cooperative_groups;


__global__ void project_gaussians_2d_scale_rot_forward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    // if (idx == 100) {
    //     printf("hello from %d\n", idx);
    // }
    
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 2D Gaussian parameters
    float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
                     0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};

    glm::mat2 R = rotmat2d(rotation[idx]);
    glm::mat2 S = scale_to_mat2d(scales2d[idx]);
    glm::mat2 M = R * S;
    glm::mat2 tmp = M * glm::transpose(M);
    // glm::mat2 tmp = R * S * glm::transpose(R);

    float3 cov2d = make_float3(tmp[0][0], tmp[0][1], tmp[1][1]);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    // if (idx == 10) {
    //     printf("center %.2f %.2f\n", center.x, center.y);
    //     printf("R %.2f %.2f %.2f %.2f\n", R[0][0], R[1][0], R[0][1], R[1][1]);
    //     printf("S %.2f %.2f %.2f %.2f\n", S[0][0], S[1][0], S[0][1], S[1][1]);
    //     printf("M %.2f %.2f %.2f %.2f\n", M[0][0], M[1][0], M[0][1], M[1][1]);
    //     printf("M.T %.2f %.2f %.2f %.2f\n", glm::transpose(M)[0][0], glm::transpose(M)[1][0], glm::transpose(M)[0][1], glm::transpose(M)[1][1]);
    //     printf("tmp %.2f %.2f %.2f %.2f\n", tmp[0][0], tmp[1][0], tmp[0][1], tmp[1][1]);
    //     printf("conv2d %.2f %.2f %.2f\n", cov2d.x, cov2d.y, cov2d.z);
    //     printf("conic %.2f %.2f %.2f\n", conic.x, conic.y, conic.z);
    //     printf("radius %.2f\n", radius);
    // }
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    // if (tile_area <= 0) {
    //     printf("%d point bbox outside of bounds\n", idx);
    //     return;
    // }
    num_tiles_hit[idx] = tile_area;
    // if (idx == 10) {
    //     printf("tile_area %d\n", tile_area);
    // }
    // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;

}