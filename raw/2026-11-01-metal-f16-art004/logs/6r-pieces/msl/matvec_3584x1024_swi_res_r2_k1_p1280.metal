#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_swi_res_r2_k1_p1280(
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
    uint v3 = sgid / 1u;
    uint v4 = sgid % 1u;
    for (uint v7 = tgid.x; v7 < 128u; v7 += 1280u) {
        uint v9 = v7 * 4u;
        uint v10 = v9 + v3;
        uint v13 = v10 * 2u;
        uint v18 = v13 + 0u;
        bool v19 = v18 < 1024u;
        uint v20 = v19 ? v18 : 1023u;
        uint v21 = v20 * 448u;
        uint v23 = v13 + 1u;
        bool v24 = v23 < 1024u;
        uint v25 = v24 ? v23 : 1023u;
        uint v26 = v25 * 448u;
        uint v29 = v4 * 32u;
        uint v30 = v29 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        for (uint v33 = v30; v33 < 448u; v33 += 32u) {
            uint4 v35 = b1[v33];
            uint v36 = v33 + 448u;
            uint4 v37 = b1[v36];
            uint v38 = v35.x;
            float v39 = float(as_type<half>(ushort(v38)));
            float v40 = float(as_type<half>(ushort(v38 >> 16)));
            uint v41 = v37.x;
            float v42 = float(as_type<half>(ushort(v41)));
            float v43 = float(as_type<half>(ushort(v41 >> 16)));
            float v44 = -v39;
            float v45 = exp(v44);
            float v47 = as_type<float>(0x3f800000u) + v45;
            float v48 = as_type<float>(0x3f800000u) / v47;
            float v49 = v39 * v48;
            float v50 = v49 * v42;
            float v51 = float(half(v50));
            float v52 = -v40;
            float v53 = exp(v52);
            float v55 = as_type<float>(0x3f800000u) + v53;
            float v56 = as_type<float>(0x3f800000u) / v55;
            float v57 = v40 * v56;
            float v58 = v57 * v43;
            float v59 = float(half(v58));
            uint v60 = v35.y;
            float v61 = float(as_type<half>(ushort(v60)));
            float v62 = float(as_type<half>(ushort(v60 >> 16)));
            uint v63 = v37.y;
            float v64 = float(as_type<half>(ushort(v63)));
            float v65 = float(as_type<half>(ushort(v63 >> 16)));
            float v66 = -v61;
            float v67 = exp(v66);
            float v69 = as_type<float>(0x3f800000u) + v67;
            float v70 = as_type<float>(0x3f800000u) / v69;
            float v71 = v61 * v70;
            float v72 = v71 * v64;
            float v73 = float(half(v72));
            float v74 = -v62;
            float v75 = exp(v74);
            float v77 = as_type<float>(0x3f800000u) + v75;
            float v78 = as_type<float>(0x3f800000u) / v77;
            float v79 = v62 * v78;
            float v80 = v79 * v65;
            float v81 = float(half(v80));
            uint v82 = v35.z;
            float v83 = float(as_type<half>(ushort(v82)));
            float v84 = float(as_type<half>(ushort(v82 >> 16)));
            uint v85 = v37.z;
            float v86 = float(as_type<half>(ushort(v85)));
            float v87 = float(as_type<half>(ushort(v85 >> 16)));
            float v88 = -v83;
            float v89 = exp(v88);
            float v91 = as_type<float>(0x3f800000u) + v89;
            float v92 = as_type<float>(0x3f800000u) / v91;
            float v93 = v83 * v92;
            float v94 = v93 * v86;
            float v95 = float(half(v94));
            float v96 = -v84;
            float v97 = exp(v96);
            float v99 = as_type<float>(0x3f800000u) + v97;
            float v100 = as_type<float>(0x3f800000u) / v99;
            float v101 = v84 * v100;
            float v102 = v101 * v87;
            float v103 = float(half(v102));
            uint v104 = v35.w;
            float v105 = float(as_type<half>(ushort(v104)));
            float v106 = float(as_type<half>(ushort(v104 >> 16)));
            uint v107 = v37.w;
            float v108 = float(as_type<half>(ushort(v107)));
            float v109 = float(as_type<half>(ushort(v107 >> 16)));
            float v110 = -v105;
            float v111 = exp(v110);
            float v113 = as_type<float>(0x3f800000u) + v111;
            float v114 = as_type<float>(0x3f800000u) / v113;
            float v115 = v105 * v114;
            float v116 = v115 * v108;
            float v117 = float(half(v116));
            float v118 = -v106;
            float v119 = exp(v118);
            float v121 = as_type<float>(0x3f800000u) + v119;
            float v122 = as_type<float>(0x3f800000u) / v121;
            float v123 = v106 * v122;
            float v124 = v123 * v109;
            float v125 = float(half(v124));
            uint v126 = v21 + v33;
            uint4 v127 = b0[v126];
            float v128 = m0;
            uint v129 = v127.x;
            float v130 = float(as_type<half>(ushort(v129)));
            float v131 = fma(v130, v51, v128);
            float v132 = float(as_type<half>(ushort(v129 >> 16)));
            float v133 = fma(v132, v59, v131);
            uint v134 = v127.y;
            float v135 = float(as_type<half>(ushort(v134)));
            float v136 = fma(v135, v73, v133);
            float v137 = float(as_type<half>(ushort(v134 >> 16)));
            float v138 = fma(v137, v81, v136);
            uint v139 = v127.z;
            float v140 = float(as_type<half>(ushort(v139)));
            float v141 = fma(v140, v95, v138);
            float v142 = float(as_type<half>(ushort(v139 >> 16)));
            float v143 = fma(v142, v103, v141);
            uint v144 = v127.w;
            float v145 = float(as_type<half>(ushort(v144)));
            float v146 = fma(v145, v117, v143);
            float v147 = float(as_type<half>(ushort(v144 >> 16)));
            float v148 = fma(v147, v125, v146);
            m0 = v148;
            uint v149 = v26 + v33;
            uint4 v150 = b0[v149];
            float v151 = m1;
            uint v152 = v150.x;
            float v153 = float(as_type<half>(ushort(v152)));
            float v154 = fma(v153, v51, v151);
            float v155 = float(as_type<half>(ushort(v152 >> 16)));
            float v156 = fma(v155, v59, v154);
            uint v157 = v150.y;
            float v158 = float(as_type<half>(ushort(v157)));
            float v159 = fma(v158, v73, v156);
            float v160 = float(as_type<half>(ushort(v157 >> 16)));
            float v161 = fma(v160, v81, v159);
            uint v162 = v150.z;
            float v163 = float(as_type<half>(ushort(v162)));
            float v164 = fma(v163, v95, v161);
            float v165 = float(as_type<half>(ushort(v162 >> 16)));
            float v166 = fma(v165, v103, v164);
            uint v167 = v150.w;
            float v168 = float(as_type<half>(ushort(v167)));
            float v169 = fma(v168, v117, v166);
            float v170 = float(as_type<half>(ushort(v167 >> 16)));
            float v171 = fma(v170, v125, v169);
            m1 = v171;
        }
        bool v173 = lane == 0u;
        uint v174 = v3 * 2u;
        float v175 = m0;
        float v176 = simd_sum(v175);
        uint v178 = v174 + 0u;
        uint v179 = v178 * 1u;
        uint v180 = v179 + v4;
        if (v173) {
            s0[v180] = v176;
        }
        float v181 = m1;
        float v182 = simd_sum(v181);
        uint v184 = v174 + 1u;
        uint v185 = v184 * 1u;
        uint v186 = v185 + v4;
        if (v173) {
            s0[v186] = v182;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v187 = v4 == 0u;
        bool v188 = v173 && v187;
        uint v190 = v13 + 0u;
        bool v191 = v190 < 1024u;
        bool v192 = v188 && v191;
        if (v192) {
            uint v194 = v174 + 0u;
            uint v195 = v194 * 1u;
            float v196 = s0[v195];
            float v197 = float(half(v196));
            uint v198 = uint(b2[v190]);
            float v199 = float(as_type<half>(ushort(v198)));
            float v200 = v199 + v197;
            uint v201 = uint(as_type<ushort>(half(v200)));
            b2[v190] = ushort(v201);
        }
        uint v203 = v13 + 1u;
        bool v204 = v203 < 1024u;
        bool v205 = v188 && v204;
        if (v205) {
            uint v207 = v174 + 1u;
            uint v208 = v207 * 1u;
            float v209 = s0[v208];
            float v210 = float(half(v209));
            uint v211 = uint(b2[v203]);
            float v212 = float(as_type<half>(ushort(v211)));
            float v213 = v212 + v210;
            uint v214 = uint(as_type<ushort>(half(v213)));
            b2[v203] = ushort(v214);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
