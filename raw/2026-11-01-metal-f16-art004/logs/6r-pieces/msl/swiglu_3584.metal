#include <metal_stdlib>
using namespace metal;

kernel void swiglu_3584(
    device ushort *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    bool v2 = tpig.x < 3584u;
    if (v2) {
        uint v3 = uint(b0[tpig.x]);
        uint v4 = tpig.x + 3584u;
        uint v5 = uint(b0[v4]);
        float v6 = float(as_type<half>(ushort(v3)));
        float v7 = float(as_type<half>(ushort(v5)));
        float v8 = -v6;
        float v9 = exp(v8);
        float v11 = as_type<float>(0x3f800000u) + v9;
        float v12 = as_type<float>(0x3f800000u) / v11;
        float v13 = v6 * v12;
        float v14 = v13 * v7;
        uint v15 = uint(as_type<ushort>(half(v14)));
        b1[tpig.x] = ushort(v15);
    }
}
