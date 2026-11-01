#include <metal_stdlib>
using namespace metal;

kernel void embed_rows_1024(
    device uint *b0 [[buffer(0)]],
    device uint *b1 [[buffer(1)]],
    device uint *b2 [[buffer(2)]],
    device uint *b3 [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint v2 = b0[3u];
    uint v3 = v2 * 512u;
    bool v5 = tpig.x < v3;
    if (v5) {
        uint v6 = tpig.x / 512u;
        uint v7 = tpig.x % 512u;
        uint v8 = b1[v6];
        uint v9 = v8 * 512u;
        uint v10 = v9 + v7;
        uint v11 = b2[v10];
        b3[tpig.x] = v11;
    }
}
