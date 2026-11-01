#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x5120_r2_k2(
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
    threadgroup float s0[8];
    uint v3 = sgid / 2u;
    uint v4 = sgid % 2u;
    uint v6 = tgid.x * 2u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 2u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 5120u;
    uint v17 = v16 ? v15 : 5119u;
    uint v18 = v17 * 128u;
    uint v20 = v10 + 1u;
    bool v21 = v20 < 5120u;
    uint v22 = v21 ? v20 : 5119u;
    uint v23 = v22 * 128u;
    uint v26 = v4 * 32u;
    uint v27 = v26 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    for (uint v30 = v27; v30 < 128u; v30 += 64u) {
        uint4 v32 = b1[v30];
        uint v33 = v32.x;
        float v34 = float(as_type<half>(ushort(v33)));
        float v35 = float(as_type<half>(ushort(v33 >> 16)));
        uint v36 = v32.y;
        float v37 = float(as_type<half>(ushort(v36)));
        float v38 = float(as_type<half>(ushort(v36 >> 16)));
        uint v39 = v32.z;
        float v40 = float(as_type<half>(ushort(v39)));
        float v41 = float(as_type<half>(ushort(v39 >> 16)));
        uint v42 = v32.w;
        float v43 = float(as_type<half>(ushort(v42)));
        float v44 = float(as_type<half>(ushort(v42 >> 16)));
        uint v45 = v18 + v30;
        uint4 v46 = b0[v45];
        float v47 = m0;
        uint v48 = v46.x;
        float v49 = float(as_type<half>(ushort(v48)));
        float v50 = fma(v49, v34, v47);
        float v51 = float(as_type<half>(ushort(v48 >> 16)));
        float v52 = fma(v51, v35, v50);
        uint v53 = v46.y;
        float v54 = float(as_type<half>(ushort(v53)));
        float v55 = fma(v54, v37, v52);
        float v56 = float(as_type<half>(ushort(v53 >> 16)));
        float v57 = fma(v56, v38, v55);
        uint v58 = v46.z;
        float v59 = float(as_type<half>(ushort(v58)));
        float v60 = fma(v59, v40, v57);
        float v61 = float(as_type<half>(ushort(v58 >> 16)));
        float v62 = fma(v61, v41, v60);
        uint v63 = v46.w;
        float v64 = float(as_type<half>(ushort(v63)));
        float v65 = fma(v64, v43, v62);
        float v66 = float(as_type<half>(ushort(v63 >> 16)));
        float v67 = fma(v66, v44, v65);
        m0 = v67;
        uint v68 = v23 + v30;
        uint4 v69 = b0[v68];
        float v70 = m1;
        uint v71 = v69.x;
        float v72 = float(as_type<half>(ushort(v71)));
        float v73 = fma(v72, v34, v70);
        float v74 = float(as_type<half>(ushort(v71 >> 16)));
        float v75 = fma(v74, v35, v73);
        uint v76 = v69.y;
        float v77 = float(as_type<half>(ushort(v76)));
        float v78 = fma(v77, v37, v75);
        float v79 = float(as_type<half>(ushort(v76 >> 16)));
        float v80 = fma(v79, v38, v78);
        uint v81 = v69.z;
        float v82 = float(as_type<half>(ushort(v81)));
        float v83 = fma(v82, v40, v80);
        float v84 = float(as_type<half>(ushort(v81 >> 16)));
        float v85 = fma(v84, v41, v83);
        uint v86 = v69.w;
        float v87 = float(as_type<half>(ushort(v86)));
        float v88 = fma(v87, v43, v85);
        float v89 = float(as_type<half>(ushort(v86 >> 16)));
        float v90 = fma(v89, v44, v88);
        m1 = v90;
    }
    bool v92 = lane == 0u;
    uint v93 = v3 * 2u;
    float v94 = m0;
    float v95 = simd_sum(v94);
    uint v97 = v93 + 0u;
    uint v98 = v97 * 2u;
    uint v99 = v98 + v4;
    if (v92) {
        s0[v99] = v95;
    }
    float v100 = m1;
    float v101 = simd_sum(v100);
    uint v103 = v93 + 1u;
    uint v104 = v103 * 2u;
    uint v105 = v104 + v4;
    if (v92) {
        s0[v105] = v101;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v106 = v4 == 0u;
    bool v107 = v92 && v106;
    uint v109 = v10 + 0u;
    bool v110 = v109 < 5120u;
    bool v111 = v107 && v110;
    if (v111) {
        uint v113 = v93 + 0u;
        uint v114 = v113 * 2u;
        float v115 = s0[v114];
        uint v117 = v114 + 1u;
        float v118 = s0[v117];
        float v119 = v115 + v118;
        uint v120 = uint(as_type<ushort>(half(v119)));
        b2[v109] = ushort(v120);
    }
    uint v122 = v10 + 1u;
    bool v123 = v122 < 5120u;
    bool v124 = v107 && v123;
    if (v124) {
        uint v126 = v93 + 1u;
        uint v127 = v126 * 2u;
        float v128 = s0[v127];
        uint v130 = v127 + 1u;
        float v131 = s0[v130];
        float v132 = v128 + v131;
        uint v133 = uint(as_type<ushort>(half(v132)));
        b2[v122] = ushort(v133);
    }
}
