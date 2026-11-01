#include <metal_stdlib>
using namespace metal;

kernel void embed_1024(
    device uint *b0 [[buffer(0)]],
    device uint *b1 [[buffer(1)]],
    device uint *b2 [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    bool v2 = tpig.x < 512u;
    if (v2) {
        uint v4 = b0[0u];
        uint v5 = v4 * 512u;
        uint v6 = v5 + tpig.x;
        uint v7 = b1[v6];
        b2[tpig.x] = v7;
    }
}
