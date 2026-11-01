#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_swi_res_r1_k4_p1280(
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
    for (uint v7 = tgid.x; v7 < 1024u; v7 += 1280u) {
        uint v9 = v7 * 1u;
        uint v10 = v9 + v3;
        uint v13 = v10 * 1u;
        uint v18 = v13 + 0u;
        bool v19 = v18 < 1024u;
        uint v20 = v19 ? v18 : 1023u;
        uint v21 = v20 * 448u;
        uint v24 = v4 * 32u;
        uint v25 = v24 + lane;
        float m0 = as_type<float>(0x00000000u);
        for (uint v28 = v25; v28 < 448u; v28 += 128u) {
            uint4 v30 = b1[v28];
            uint v31 = v28 + 448u;
            uint4 v32 = b1[v31];
            uint v33 = v30.x;
            float v34 = float(as_type<half>(ushort(v33)));
            float v35 = float(as_type<half>(ushort(v33 >> 16)));
            uint v36 = v32.x;
            float v37 = float(as_type<half>(ushort(v36)));
            float v38 = float(as_type<half>(ushort(v36 >> 16)));
            float v39 = -v34;
            float v40 = exp(v39);
            float v42 = as_type<float>(0x3f800000u) + v40;
            float v43 = as_type<float>(0x3f800000u) / v42;
            float v44 = v34 * v43;
            float v45 = v44 * v37;
            float v46 = float(half(v45));
            float v47 = -v35;
            float v48 = exp(v47);
            float v50 = as_type<float>(0x3f800000u) + v48;
            float v51 = as_type<float>(0x3f800000u) / v50;
            float v52 = v35 * v51;
            float v53 = v52 * v38;
            float v54 = float(half(v53));
            uint v55 = v30.y;
            float v56 = float(as_type<half>(ushort(v55)));
            float v57 = float(as_type<half>(ushort(v55 >> 16)));
            uint v58 = v32.y;
            float v59 = float(as_type<half>(ushort(v58)));
            float v60 = float(as_type<half>(ushort(v58 >> 16)));
            float v61 = -v56;
            float v62 = exp(v61);
            float v64 = as_type<float>(0x3f800000u) + v62;
            float v65 = as_type<float>(0x3f800000u) / v64;
            float v66 = v56 * v65;
            float v67 = v66 * v59;
            float v68 = float(half(v67));
            float v69 = -v57;
            float v70 = exp(v69);
            float v72 = as_type<float>(0x3f800000u) + v70;
            float v73 = as_type<float>(0x3f800000u) / v72;
            float v74 = v57 * v73;
            float v75 = v74 * v60;
            float v76 = float(half(v75));
            uint v77 = v30.z;
            float v78 = float(as_type<half>(ushort(v77)));
            float v79 = float(as_type<half>(ushort(v77 >> 16)));
            uint v80 = v32.z;
            float v81 = float(as_type<half>(ushort(v80)));
            float v82 = float(as_type<half>(ushort(v80 >> 16)));
            float v83 = -v78;
            float v84 = exp(v83);
            float v86 = as_type<float>(0x3f800000u) + v84;
            float v87 = as_type<float>(0x3f800000u) / v86;
            float v88 = v78 * v87;
            float v89 = v88 * v81;
            float v90 = float(half(v89));
            float v91 = -v79;
            float v92 = exp(v91);
            float v94 = as_type<float>(0x3f800000u) + v92;
            float v95 = as_type<float>(0x3f800000u) / v94;
            float v96 = v79 * v95;
            float v97 = v96 * v82;
            float v98 = float(half(v97));
            uint v99 = v30.w;
            float v100 = float(as_type<half>(ushort(v99)));
            float v101 = float(as_type<half>(ushort(v99 >> 16)));
            uint v102 = v32.w;
            float v103 = float(as_type<half>(ushort(v102)));
            float v104 = float(as_type<half>(ushort(v102 >> 16)));
            float v105 = -v100;
            float v106 = exp(v105);
            float v108 = as_type<float>(0x3f800000u) + v106;
            float v109 = as_type<float>(0x3f800000u) / v108;
            float v110 = v100 * v109;
            float v111 = v110 * v103;
            float v112 = float(half(v111));
            float v113 = -v101;
            float v114 = exp(v113);
            float v116 = as_type<float>(0x3f800000u) + v114;
            float v117 = as_type<float>(0x3f800000u) / v116;
            float v118 = v101 * v117;
            float v119 = v118 * v104;
            float v120 = float(half(v119));
            uint v121 = v21 + v28;
            uint4 v122 = b0[v121];
            float v123 = m0;
            uint v124 = v122.x;
            float v125 = float(as_type<half>(ushort(v124)));
            float v126 = fma(v125, v46, v123);
            float v127 = float(as_type<half>(ushort(v124 >> 16)));
            float v128 = fma(v127, v54, v126);
            uint v129 = v122.y;
            float v130 = float(as_type<half>(ushort(v129)));
            float v131 = fma(v130, v68, v128);
            float v132 = float(as_type<half>(ushort(v129 >> 16)));
            float v133 = fma(v132, v76, v131);
            uint v134 = v122.z;
            float v135 = float(as_type<half>(ushort(v134)));
            float v136 = fma(v135, v90, v133);
            float v137 = float(as_type<half>(ushort(v134 >> 16)));
            float v138 = fma(v137, v98, v136);
            uint v139 = v122.w;
            float v140 = float(as_type<half>(ushort(v139)));
            float v141 = fma(v140, v112, v138);
            float v142 = float(as_type<half>(ushort(v139 >> 16)));
            float v143 = fma(v142, v120, v141);
            m0 = v143;
        }
        bool v145 = lane == 0u;
        uint v146 = v3 * 1u;
        float v147 = m0;
        float v148 = simd_sum(v147);
        uint v150 = v146 + 0u;
        uint v151 = v150 * 4u;
        uint v152 = v151 + v4;
        if (v145) {
            s0[v152] = v148;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v153 = v4 == 0u;
        bool v154 = v145 && v153;
        uint v156 = v13 + 0u;
        bool v157 = v156 < 1024u;
        bool v158 = v154 && v157;
        if (v158) {
            uint v160 = v146 + 0u;
            uint v161 = v160 * 4u;
            float v162 = s0[v161];
            uint v164 = v161 + 1u;
            float v165 = s0[v164];
            float v166 = v162 + v165;
            uint v168 = v161 + 2u;
            float v169 = s0[v168];
            float v170 = v166 + v169;
            uint v172 = v161 + 3u;
            float v173 = s0[v172];
            float v174 = v170 + v173;
            float v175 = float(half(v174));
            uint v176 = uint(b2[v156]);
            float v177 = float(as_type<half>(ushort(v176)));
            float v178 = v177 + v175;
            uint v179 = uint(as_type<ushort>(half(v178)));
            b2[v156] = ushort(v179);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
