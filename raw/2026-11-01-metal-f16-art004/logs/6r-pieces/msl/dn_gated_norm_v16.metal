#include <metal_stdlib>
using namespace metal;

kernel void dn_gated_norm_v16(
    device ushort *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[5];
    uint v3 = tgid.x * 128u;
    uint v4 = v3 + lid;
    uint v5 = uint(b0[v4]);
    float v6 = float(as_type<half>(ushort(v5)));
    float v7 = v6 * v6;
    float v8 = simd_sum(v7);
    bool v12 = lane == 0u;
    if (v12) {
        s0[sgid] = v8;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v14 = lid == 0u;
    if (v14) {
        float v16 = s0[0u];
        float m0 = v16;
        for (uint v19 = 1u; v19 < 4u; v19 += 1u) {
            float v20 = m0;
            float v21 = s0[v19];
            float v22 = v20 + v21;
            m0 = v22;
        }
        float v23 = m0;
        s0[4u] = v23;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v26 = s0[4u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v29 = v26 * as_type<float>(0x3c000000u);
    float v30 = v29 + as_type<float>(0x358637bdu);
    float v31 = sqrt(v30);
    float v33 = as_type<float>(0x3f800000u) / v31;
    uint v34 = uint(b2[lid]);
    float v35 = float(as_type<half>(ushort(v34)));
    uint v37 = 6144u + v4;
    uint v38 = uint(b1[v37]);
    float v39 = float(as_type<half>(ushort(v38)));
    float v40 = -v39;
    float v41 = exp(v40);
    float v43 = as_type<float>(0x3f800000u) + v41;
    float v44 = as_type<float>(0x3f800000u) / v43;
    float v45 = v39 * v44;
    float v46 = v6 * v33;
    float v47 = v46 * v35;
    float v48 = v47 * v45;
    uint v49 = uint(as_type<ushort>(half(v48)));
    b3[v4] = ushort(v49);
}
