#include <metal_stdlib>
using namespace metal;

kernel void argmax_partial_248320(
    device ushort *b0 [[buffer(0)]],
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
    uint v3 = tgid.x * 4096u;
    uint v4 = v3 + lid;
    uint v5 = v3 + 4096u;
    bool v7 = v5 < 248320u;
    uint v8 = v7 ? v5 : 248320u;
    float m0 = as_type<float>(0xff800000u);
    uint m1 = 4294967295u;
    for (uint v12 = v4; v12 < v8; v12 += 256u) {
        uint v13 = uint(b0[v12]);
        float v14 = float(as_type<half>(ushort(v13)));
        float v15 = m0;
        bool v16 = v14 > v15;
        if (v16) {
            m0 = v14;
            m1 = v12;
        }
    }
    float v17 = m0;
    float v18 = simd_max(v17);
    bool v22 = lane == 0u;
    if (v22) {
        s0[sgid] = v18;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v24 = lid == 0u;
    if (v24) {
        float v26 = s0[0u];
        float m2 = v26;
        for (uint v29 = 1u; v29 < 8u; v29 += 1u) {
            float v30 = m2;
            float v31 = s0[v29];
            float v32 = fmax(v30, v31);
            m2 = v32;
        }
        float v33 = m2;
        s0[8u] = v33;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v36 = s0[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v37 = m1;
    bool v38 = v17 == v36;
    uint v39 = v38 ? v37 : 4294967295u;
    uint v40 = simd_min(v39);
    bool v44 = lane == 0u;
    if (v44) {
        s1[sgid] = v40;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v46 = lid == 0u;
    if (v46) {
        uint v48 = s1[0u];
        uint m3 = v48;
        for (uint v51 = 1u; v51 < 8u; v51 += 1u) {
            uint v52 = m3;
            uint v53 = s1[v51];
            bool v54 = v53 < v52;
            uint v55 = v54 ? v53 : v52;
            m3 = v55;
        }
        uint v56 = m3;
        s1[8u] = v56;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v59 = s1[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v61 = lid == 0u;
    if (v61) {
        uint v63 = tgid.x * 2u;
        uint v64 = as_type<uint>(v36);
        b1[v63] = v64;
        uint v66 = v63 + 1u;
        b1[v66] = v59;
    }
}
