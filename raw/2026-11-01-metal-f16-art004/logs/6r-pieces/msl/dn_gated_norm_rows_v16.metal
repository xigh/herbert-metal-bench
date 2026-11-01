#include <metal_stdlib>
using namespace metal;

kernel void dn_gated_norm_rows_v16(
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
    uint v2 = tgid.x / 16u;
    uint v5 = tgid.x * 128u;
    uint v6 = v5 + lid;
    uint v7 = uint(b0[v6]);
    float v8 = float(as_type<half>(ushort(v7)));
    float v9 = v8 * v8;
    float v10 = simd_sum(v9);
    bool v14 = lane == 0u;
    if (v14) {
        s0[sgid] = v10;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v16 = lid == 0u;
    if (v16) {
        float v18 = s0[0u];
        float m0 = v18;
        for (uint v21 = 1u; v21 < 4u; v21 += 1u) {
            float v22 = m0;
            float v23 = s0[v21];
            float v24 = v22 + v23;
            m0 = v24;
        }
        float v25 = m0;
        s0[4u] = v25;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v28 = s0[4u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v31 = v28 * as_type<float>(0x3c000000u);
    float v32 = v31 + as_type<float>(0x358637bdu);
    float v33 = sqrt(v32);
    float v35 = as_type<float>(0x3f800000u) / v33;
    uint v36 = uint(b2[lid]);
    float v37 = float(as_type<half>(ushort(v36)));
    uint v40 = v2 * 8224u;
    uint v42 = v2 * 2048u;
    uint v43 = v6 - v42;
    uint v44 = v40 + 6144u;
    uint v45 = v44 + v43;
    uint v46 = uint(b1[v45]);
    float v47 = float(as_type<half>(ushort(v46)));
    float v48 = -v47;
    float v49 = exp(v48);
    float v51 = as_type<float>(0x3f800000u) + v49;
    float v52 = as_type<float>(0x3f800000u) / v51;
    float v53 = v47 * v52;
    float v54 = v8 * v35;
    float v55 = v54 * v37;
    float v56 = v55 * v53;
    uint v57 = uint(as_type<ushort>(half(v56)));
    b3[v6] = ushort(v57);
}
