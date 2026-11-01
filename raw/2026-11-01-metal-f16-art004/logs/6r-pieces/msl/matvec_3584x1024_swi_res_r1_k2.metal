#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_swi_res_r1_k2(
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
    uint v3 = sgid / 2u;
    uint v4 = sgid % 2u;
    uint v6 = tgid.x * 2u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 1u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 1024u;
    uint v17 = v16 ? v15 : 1023u;
    uint v18 = v17 * 448u;
    uint v21 = v4 * 32u;
    uint v22 = v21 + lane;
    float m0 = as_type<float>(0x00000000u);
    for (uint v25 = v22; v25 < 448u; v25 += 64u) {
        uint4 v27 = b1[v25];
        uint v28 = v25 + 448u;
        uint4 v29 = b1[v28];
        uint v30 = v27.x;
        float v31 = float(as_type<half>(ushort(v30)));
        float v32 = float(as_type<half>(ushort(v30 >> 16)));
        uint v33 = v29.x;
        float v34 = float(as_type<half>(ushort(v33)));
        float v35 = float(as_type<half>(ushort(v33 >> 16)));
        float v36 = -v31;
        float v37 = exp(v36);
        float v39 = as_type<float>(0x3f800000u) + v37;
        float v40 = as_type<float>(0x3f800000u) / v39;
        float v41 = v31 * v40;
        float v42 = v41 * v34;
        float v43 = float(half(v42));
        float v44 = -v32;
        float v45 = exp(v44);
        float v47 = as_type<float>(0x3f800000u) + v45;
        float v48 = as_type<float>(0x3f800000u) / v47;
        float v49 = v32 * v48;
        float v50 = v49 * v35;
        float v51 = float(half(v50));
        uint v52 = v27.y;
        float v53 = float(as_type<half>(ushort(v52)));
        float v54 = float(as_type<half>(ushort(v52 >> 16)));
        uint v55 = v29.y;
        float v56 = float(as_type<half>(ushort(v55)));
        float v57 = float(as_type<half>(ushort(v55 >> 16)));
        float v58 = -v53;
        float v59 = exp(v58);
        float v61 = as_type<float>(0x3f800000u) + v59;
        float v62 = as_type<float>(0x3f800000u) / v61;
        float v63 = v53 * v62;
        float v64 = v63 * v56;
        float v65 = float(half(v64));
        float v66 = -v54;
        float v67 = exp(v66);
        float v69 = as_type<float>(0x3f800000u) + v67;
        float v70 = as_type<float>(0x3f800000u) / v69;
        float v71 = v54 * v70;
        float v72 = v71 * v57;
        float v73 = float(half(v72));
        uint v74 = v27.z;
        float v75 = float(as_type<half>(ushort(v74)));
        float v76 = float(as_type<half>(ushort(v74 >> 16)));
        uint v77 = v29.z;
        float v78 = float(as_type<half>(ushort(v77)));
        float v79 = float(as_type<half>(ushort(v77 >> 16)));
        float v80 = -v75;
        float v81 = exp(v80);
        float v83 = as_type<float>(0x3f800000u) + v81;
        float v84 = as_type<float>(0x3f800000u) / v83;
        float v85 = v75 * v84;
        float v86 = v85 * v78;
        float v87 = float(half(v86));
        float v88 = -v76;
        float v89 = exp(v88);
        float v91 = as_type<float>(0x3f800000u) + v89;
        float v92 = as_type<float>(0x3f800000u) / v91;
        float v93 = v76 * v92;
        float v94 = v93 * v79;
        float v95 = float(half(v94));
        uint v96 = v27.w;
        float v97 = float(as_type<half>(ushort(v96)));
        float v98 = float(as_type<half>(ushort(v96 >> 16)));
        uint v99 = v29.w;
        float v100 = float(as_type<half>(ushort(v99)));
        float v101 = float(as_type<half>(ushort(v99 >> 16)));
        float v102 = -v97;
        float v103 = exp(v102);
        float v105 = as_type<float>(0x3f800000u) + v103;
        float v106 = as_type<float>(0x3f800000u) / v105;
        float v107 = v97 * v106;
        float v108 = v107 * v100;
        float v109 = float(half(v108));
        float v110 = -v98;
        float v111 = exp(v110);
        float v113 = as_type<float>(0x3f800000u) + v111;
        float v114 = as_type<float>(0x3f800000u) / v113;
        float v115 = v98 * v114;
        float v116 = v115 * v101;
        float v117 = float(half(v116));
        uint v118 = v18 + v25;
        uint4 v119 = b0[v118];
        float v120 = m0;
        uint v121 = v119.x;
        float v122 = float(as_type<half>(ushort(v121)));
        float v123 = fma(v122, v43, v120);
        float v124 = float(as_type<half>(ushort(v121 >> 16)));
        float v125 = fma(v124, v51, v123);
        uint v126 = v119.y;
        float v127 = float(as_type<half>(ushort(v126)));
        float v128 = fma(v127, v65, v125);
        float v129 = float(as_type<half>(ushort(v126 >> 16)));
        float v130 = fma(v129, v73, v128);
        uint v131 = v119.z;
        float v132 = float(as_type<half>(ushort(v131)));
        float v133 = fma(v132, v87, v130);
        float v134 = float(as_type<half>(ushort(v131 >> 16)));
        float v135 = fma(v134, v95, v133);
        uint v136 = v119.w;
        float v137 = float(as_type<half>(ushort(v136)));
        float v138 = fma(v137, v109, v135);
        float v139 = float(as_type<half>(ushort(v136 >> 16)));
        float v140 = fma(v139, v117, v138);
        m0 = v140;
    }
    bool v142 = lane == 0u;
    uint v143 = v3 * 1u;
    float v144 = m0;
    float v145 = simd_sum(v144);
    uint v147 = v143 + 0u;
    uint v148 = v147 * 2u;
    uint v149 = v148 + v4;
    if (v142) {
        s0[v149] = v145;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v150 = v4 == 0u;
    bool v151 = v142 && v150;
    uint v153 = v10 + 0u;
    bool v154 = v153 < 1024u;
    bool v155 = v151 && v154;
    if (v155) {
        uint v157 = v143 + 0u;
        uint v158 = v157 * 2u;
        float v159 = s0[v158];
        uint v161 = v158 + 1u;
        float v162 = s0[v161];
        float v163 = v159 + v162;
        float v164 = float(half(v163));
        uint v165 = uint(b2[v153]);
        float v166 = float(as_type<half>(ushort(v165)));
        float v167 = v166 + v164;
        uint v168 = uint(as_type<ushort>(half(v167)));
        b2[v153] = ushort(v168);
    }
}
