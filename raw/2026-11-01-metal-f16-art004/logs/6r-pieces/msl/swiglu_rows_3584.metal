#include <metal_stdlib>
using namespace metal;

kernel void swiglu_rows_3584(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint v1 = b0[3u];
    uint v3 = v1 * 3584u;
    bool v5 = tpig.x < v3;
    if (v5) {
        uint v6 = tpig.x / 3584u;
        uint v7 = tpig.x % 3584u;
        uint v9 = v6 * 7168u;
        uint v10 = v9 + v7;
        uint v11 = v10 + 3584u;
        uint v12 = uint(b1[v10]);
        uint v13 = uint(b1[v11]);
        float v14 = float(as_type<half>(ushort(v12)));
        float v15 = float(as_type<half>(ushort(v13)));
        float v16 = -v14;
        float v17 = exp(v16);
        float v19 = as_type<float>(0x3f800000u) + v17;
        float v20 = as_type<float>(0x3f800000u) / v19;
        float v21 = v14 * v20;
        float v22 = v21 * v15;
        uint v23 = uint(as_type<ushort>(half(v22)));
        b2[tpig.x] = ushort(v23);
    }
}
