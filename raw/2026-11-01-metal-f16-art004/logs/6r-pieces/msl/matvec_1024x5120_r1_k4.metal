#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x5120_r1_k4(
    device uint4 *b0 [[buffer(0)]],
    device uint4 *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[4];
    uint v3 = sgid / 4u;
    uint v4 = sgid % 4u;
    uint v6 = tgid.x * 1u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 1u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 5120u;
    uint v17 = v16 ? v15 : 5119u;
    uint v18 = v17 * 128u;
    uint v21 = v4 * 32u;
    uint v22 = v21 + lane;
    float m0 = as_type<float>(0x00000000u);
    for (uint v25 = v22; v25 < 128u; v25 += 128u) {
        uint4 v27 = b1[v25];
        uint v28 = v27.x;
        float v29 = float(as_type<half>(ushort(v28)));
        float v30 = float(as_type<half>(ushort(v28 >> 16)));
        uint v31 = v27.y;
        float v32 = float(as_type<half>(ushort(v31)));
        float v33 = float(as_type<half>(ushort(v31 >> 16)));
        uint v34 = v27.z;
        float v35 = float(as_type<half>(ushort(v34)));
        float v36 = float(as_type<half>(ushort(v34 >> 16)));
        uint v37 = v27.w;
        float v38 = float(as_type<half>(ushort(v37)));
        float v39 = float(as_type<half>(ushort(v37 >> 16)));
        uint v40 = v18 + v25;
        uint4 v41 = b0[v40];
        float v42 = m0;
        uint v43 = v41.x;
        float v44 = float(as_type<half>(ushort(v43)));
        float v45 = fma(v44, v29, v42);
        float v46 = float(as_type<half>(ushort(v43 >> 16)));
        float v47 = fma(v46, v30, v45);
        uint v48 = v41.y;
        float v49 = float(as_type<half>(ushort(v48)));
        float v50 = fma(v49, v32, v47);
        float v51 = float(as_type<half>(ushort(v48 >> 16)));
        float v52 = fma(v51, v33, v50);
        uint v53 = v41.z;
        float v54 = float(as_type<half>(ushort(v53)));
        float v55 = fma(v54, v35, v52);
        float v56 = float(as_type<half>(ushort(v53 >> 16)));
        float v57 = fma(v56, v36, v55);
        uint v58 = v41.w;
        float v59 = float(as_type<half>(ushort(v58)));
        float v60 = fma(v59, v38, v57);
        float v61 = float(as_type<half>(ushort(v58 >> 16)));
        float v62 = fma(v61, v39, v60);
        m0 = v62;
    }
    bool v64 = lane == 0u;
    uint v65 = v3 * 1u;
    float v66 = m0;
    float v67 = simd_sum(v66);
    uint v69 = v65 + 0u;
    uint v70 = v69 * 4u;
    uint v71 = v70 + v4;
    if (v64) {
        s0[v71] = v67;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v72 = v4 == 0u;
    bool v73 = v64 && v72;
    uint v75 = v10 + 0u;
    bool v76 = v75 < 5120u;
    bool v77 = v73 && v76;
    if (v77) {
        uint v79 = v65 + 0u;
        uint v80 = v79 * 4u;
        float v81 = s0[v80];
        uint v83 = v80 + 1u;
        float v84 = s0[v83];
        float v85 = v81 + v84;
        uint v87 = v80 + 2u;
        float v88 = s0[v87];
        float v89 = v85 + v88;
        uint v91 = v80 + 3u;
        float v92 = s0[v91];
        float v93 = v89 + v92;
        uint v94 = uint(as_type<ushort>(half(v93)));
        b2[v75] = ushort(v94);
    }
}
