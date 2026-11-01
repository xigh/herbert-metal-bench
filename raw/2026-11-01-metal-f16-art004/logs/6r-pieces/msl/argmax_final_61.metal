#include <metal_stdlib>
using namespace metal;

kernel void argmax_final_61(
    device uint *b0 [[buffer(0)]],
    device uint *b1 [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[9];
    threadgroup uint s1[9];
    float m0 = as_type<float>(0xff800000u);
    uint m1 = 4294967295u;
    for (uint v5 = lid; v5 < 61u; v5 += 256u) {
        uint v7 = v5 * 2u;
        uint v8 = b0[v7];
        float v9 = as_type<float>(v8);
        uint v11 = v7 + 1u;
        uint v12 = b0[v11];
        float v13 = m0;
        bool v14 = v9 > v13;
        if (v14) {
            m0 = v9;
            m1 = v12;
        }
    }
    float v15 = m0;
    float v16 = simd_max(v15);
    bool v20 = lane == 0u;
    if (v20) {
        s0[sgid] = v16;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v22 = lid == 0u;
    if (v22) {
        float v24 = s0[0u];
        float m2 = v24;
        for (uint v27 = 1u; v27 < 8u; v27 += 1u) {
            float v28 = m2;
            float v29 = s0[v27];
            float v30 = fmax(v28, v29);
            m2 = v30;
        }
        float v31 = m2;
        s0[8u] = v31;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v34 = s0[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v35 = m1;
    bool v36 = v15 == v34;
    uint v37 = v36 ? v35 : 4294967295u;
    uint v38 = simd_min(v37);
    bool v42 = lane == 0u;
    if (v42) {
        s1[sgid] = v38;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v44 = lid == 0u;
    if (v44) {
        uint v46 = s1[0u];
        uint m3 = v46;
        for (uint v49 = 1u; v49 < 8u; v49 += 1u) {
            uint v50 = m3;
            uint v51 = s1[v49];
            bool v52 = v51 < v50;
            uint v53 = v52 ? v51 : v50;
            m3 = v53;
        }
        uint v54 = m3;
        s1[8u] = v54;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v57 = s1[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v59 = lid == 0u;
    if (v59) {
        b1[2u] = v57;
    }
}
