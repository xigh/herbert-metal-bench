#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_swi_res_r2_k4(
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
    uint v3 = sgid / 4u;
    uint v4 = sgid % 4u;
    uint v6 = tgid.x * 1u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 2u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 1024u;
    uint v17 = v16 ? v15 : 1023u;
    uint v18 = v17 * 448u;
    uint v20 = v10 + 1u;
    bool v21 = v20 < 1024u;
    uint v22 = v21 ? v20 : 1023u;
    uint v23 = v22 * 448u;
    uint v26 = v4 * 32u;
    uint v27 = v26 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    for (uint v30 = v27; v30 < 448u; v30 += 128u) {
        uint4 v32 = b1[v30];
        uint v33 = v30 + 448u;
        uint4 v34 = b1[v33];
        uint v35 = v32.x;
        float v36 = float(as_type<half>(ushort(v35)));
        float v37 = float(as_type<half>(ushort(v35 >> 16)));
        uint v38 = v34.x;
        float v39 = float(as_type<half>(ushort(v38)));
        float v40 = float(as_type<half>(ushort(v38 >> 16)));
        float v41 = -v36;
        float v42 = exp(v41);
        float v44 = as_type<float>(0x3f800000u) + v42;
        float v45 = as_type<float>(0x3f800000u) / v44;
        float v46 = v36 * v45;
        float v47 = v46 * v39;
        float v48 = float(half(v47));
        float v49 = -v37;
        float v50 = exp(v49);
        float v52 = as_type<float>(0x3f800000u) + v50;
        float v53 = as_type<float>(0x3f800000u) / v52;
        float v54 = v37 * v53;
        float v55 = v54 * v40;
        float v56 = float(half(v55));
        uint v57 = v32.y;
        float v58 = float(as_type<half>(ushort(v57)));
        float v59 = float(as_type<half>(ushort(v57 >> 16)));
        uint v60 = v34.y;
        float v61 = float(as_type<half>(ushort(v60)));
        float v62 = float(as_type<half>(ushort(v60 >> 16)));
        float v63 = -v58;
        float v64 = exp(v63);
        float v66 = as_type<float>(0x3f800000u) + v64;
        float v67 = as_type<float>(0x3f800000u) / v66;
        float v68 = v58 * v67;
        float v69 = v68 * v61;
        float v70 = float(half(v69));
        float v71 = -v59;
        float v72 = exp(v71);
        float v74 = as_type<float>(0x3f800000u) + v72;
        float v75 = as_type<float>(0x3f800000u) / v74;
        float v76 = v59 * v75;
        float v77 = v76 * v62;
        float v78 = float(half(v77));
        uint v79 = v32.z;
        float v80 = float(as_type<half>(ushort(v79)));
        float v81 = float(as_type<half>(ushort(v79 >> 16)));
        uint v82 = v34.z;
        float v83 = float(as_type<half>(ushort(v82)));
        float v84 = float(as_type<half>(ushort(v82 >> 16)));
        float v85 = -v80;
        float v86 = exp(v85);
        float v88 = as_type<float>(0x3f800000u) + v86;
        float v89 = as_type<float>(0x3f800000u) / v88;
        float v90 = v80 * v89;
        float v91 = v90 * v83;
        float v92 = float(half(v91));
        float v93 = -v81;
        float v94 = exp(v93);
        float v96 = as_type<float>(0x3f800000u) + v94;
        float v97 = as_type<float>(0x3f800000u) / v96;
        float v98 = v81 * v97;
        float v99 = v98 * v84;
        float v100 = float(half(v99));
        uint v101 = v32.w;
        float v102 = float(as_type<half>(ushort(v101)));
        float v103 = float(as_type<half>(ushort(v101 >> 16)));
        uint v104 = v34.w;
        float v105 = float(as_type<half>(ushort(v104)));
        float v106 = float(as_type<half>(ushort(v104 >> 16)));
        float v107 = -v102;
        float v108 = exp(v107);
        float v110 = as_type<float>(0x3f800000u) + v108;
        float v111 = as_type<float>(0x3f800000u) / v110;
        float v112 = v102 * v111;
        float v113 = v112 * v105;
        float v114 = float(half(v113));
        float v115 = -v103;
        float v116 = exp(v115);
        float v118 = as_type<float>(0x3f800000u) + v116;
        float v119 = as_type<float>(0x3f800000u) / v118;
        float v120 = v103 * v119;
        float v121 = v120 * v106;
        float v122 = float(half(v121));
        uint v123 = v18 + v30;
        uint4 v124 = b0[v123];
        float v125 = m0;
        uint v126 = v124.x;
        float v127 = float(as_type<half>(ushort(v126)));
        float v128 = fma(v127, v48, v125);
        float v129 = float(as_type<half>(ushort(v126 >> 16)));
        float v130 = fma(v129, v56, v128);
        uint v131 = v124.y;
        float v132 = float(as_type<half>(ushort(v131)));
        float v133 = fma(v132, v70, v130);
        float v134 = float(as_type<half>(ushort(v131 >> 16)));
        float v135 = fma(v134, v78, v133);
        uint v136 = v124.z;
        float v137 = float(as_type<half>(ushort(v136)));
        float v138 = fma(v137, v92, v135);
        float v139 = float(as_type<half>(ushort(v136 >> 16)));
        float v140 = fma(v139, v100, v138);
        uint v141 = v124.w;
        float v142 = float(as_type<half>(ushort(v141)));
        float v143 = fma(v142, v114, v140);
        float v144 = float(as_type<half>(ushort(v141 >> 16)));
        float v145 = fma(v144, v122, v143);
        m0 = v145;
        uint v146 = v23 + v30;
        uint4 v147 = b0[v146];
        float v148 = m1;
        uint v149 = v147.x;
        float v150 = float(as_type<half>(ushort(v149)));
        float v151 = fma(v150, v48, v148);
        float v152 = float(as_type<half>(ushort(v149 >> 16)));
        float v153 = fma(v152, v56, v151);
        uint v154 = v147.y;
        float v155 = float(as_type<half>(ushort(v154)));
        float v156 = fma(v155, v70, v153);
        float v157 = float(as_type<half>(ushort(v154 >> 16)));
        float v158 = fma(v157, v78, v156);
        uint v159 = v147.z;
        float v160 = float(as_type<half>(ushort(v159)));
        float v161 = fma(v160, v92, v158);
        float v162 = float(as_type<half>(ushort(v159 >> 16)));
        float v163 = fma(v162, v100, v161);
        uint v164 = v147.w;
        float v165 = float(as_type<half>(ushort(v164)));
        float v166 = fma(v165, v114, v163);
        float v167 = float(as_type<half>(ushort(v164 >> 16)));
        float v168 = fma(v167, v122, v166);
        m1 = v168;
    }
    bool v170 = lane == 0u;
    uint v171 = v3 * 2u;
    float v172 = m0;
    float v173 = simd_sum(v172);
    uint v175 = v171 + 0u;
    uint v176 = v175 * 4u;
    uint v177 = v176 + v4;
    if (v170) {
        s0[v177] = v173;
    }
    float v178 = m1;
    float v179 = simd_sum(v178);
    uint v181 = v171 + 1u;
    uint v182 = v181 * 4u;
    uint v183 = v182 + v4;
    if (v170) {
        s0[v183] = v179;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v184 = v4 == 0u;
    bool v185 = v170 && v184;
    uint v187 = v10 + 0u;
    bool v188 = v187 < 1024u;
    bool v189 = v185 && v188;
    if (v189) {
        uint v191 = v171 + 0u;
        uint v192 = v191 * 4u;
        float v193 = s0[v192];
        uint v195 = v192 + 1u;
        float v196 = s0[v195];
        float v197 = v193 + v196;
        uint v199 = v192 + 2u;
        float v200 = s0[v199];
        float v201 = v197 + v200;
        uint v203 = v192 + 3u;
        float v204 = s0[v203];
        float v205 = v201 + v204;
        float v206 = float(half(v205));
        uint v207 = uint(b2[v187]);
        float v208 = float(as_type<half>(ushort(v207)));
        float v209 = v208 + v206;
        uint v210 = uint(as_type<ushort>(half(v209)));
        b2[v187] = ushort(v210);
    }
    uint v212 = v10 + 1u;
    bool v213 = v212 < 1024u;
    bool v214 = v185 && v213;
    if (v214) {
        uint v216 = v171 + 1u;
        uint v217 = v216 * 4u;
        float v218 = s0[v217];
        uint v220 = v217 + 1u;
        float v221 = s0[v220];
        float v222 = v218 + v221;
        uint v224 = v217 + 2u;
        float v225 = s0[v224];
        float v226 = v222 + v225;
        uint v228 = v217 + 3u;
        float v229 = s0[v228];
        float v230 = v226 + v229;
        float v231 = float(half(v230));
        uint v232 = uint(b2[v212]);
        float v233 = float(as_type<half>(ushort(v232)));
        float v234 = v233 + v231;
        uint v235 = uint(as_type<ushort>(half(v234)));
        b2[v212] = ushort(v235);
    }
}
