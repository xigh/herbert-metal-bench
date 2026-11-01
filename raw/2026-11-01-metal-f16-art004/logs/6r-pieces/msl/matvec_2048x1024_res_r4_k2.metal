#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_r4_k2(
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
    threadgroup float s0[16];
    uint v3 = sgid / 2u;
    uint v4 = sgid % 2u;
    uint v6 = tgid.x * 2u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 4u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 1024u;
    uint v17 = v16 ? v15 : 1023u;
    uint v18 = v17 * 256u;
    uint v20 = v10 + 1u;
    bool v21 = v20 < 1024u;
    uint v22 = v21 ? v20 : 1023u;
    uint v23 = v22 * 256u;
    uint v25 = v10 + 2u;
    bool v26 = v25 < 1024u;
    uint v27 = v26 ? v25 : 1023u;
    uint v28 = v27 * 256u;
    uint v30 = v10 + 3u;
    bool v31 = v30 < 1024u;
    uint v32 = v31 ? v30 : 1023u;
    uint v33 = v32 * 256u;
    uint v36 = v4 * 32u;
    uint v37 = v36 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    for (uint v40 = v37; v40 < 256u; v40 += 64u) {
        uint4 v42 = b1[v40];
        uint v43 = v42.x;
        float v44 = float(as_type<half>(ushort(v43)));
        float v45 = float(as_type<half>(ushort(v43 >> 16)));
        uint v46 = v42.y;
        float v47 = float(as_type<half>(ushort(v46)));
        float v48 = float(as_type<half>(ushort(v46 >> 16)));
        uint v49 = v42.z;
        float v50 = float(as_type<half>(ushort(v49)));
        float v51 = float(as_type<half>(ushort(v49 >> 16)));
        uint v52 = v42.w;
        float v53 = float(as_type<half>(ushort(v52)));
        float v54 = float(as_type<half>(ushort(v52 >> 16)));
        uint v55 = v18 + v40;
        uint4 v56 = b0[v55];
        float v57 = m0;
        uint v58 = v56.x;
        float v59 = float(as_type<half>(ushort(v58)));
        float v60 = fma(v59, v44, v57);
        float v61 = float(as_type<half>(ushort(v58 >> 16)));
        float v62 = fma(v61, v45, v60);
        uint v63 = v56.y;
        float v64 = float(as_type<half>(ushort(v63)));
        float v65 = fma(v64, v47, v62);
        float v66 = float(as_type<half>(ushort(v63 >> 16)));
        float v67 = fma(v66, v48, v65);
        uint v68 = v56.z;
        float v69 = float(as_type<half>(ushort(v68)));
        float v70 = fma(v69, v50, v67);
        float v71 = float(as_type<half>(ushort(v68 >> 16)));
        float v72 = fma(v71, v51, v70);
        uint v73 = v56.w;
        float v74 = float(as_type<half>(ushort(v73)));
        float v75 = fma(v74, v53, v72);
        float v76 = float(as_type<half>(ushort(v73 >> 16)));
        float v77 = fma(v76, v54, v75);
        m0 = v77;
        uint v78 = v23 + v40;
        uint4 v79 = b0[v78];
        float v80 = m1;
        uint v81 = v79.x;
        float v82 = float(as_type<half>(ushort(v81)));
        float v83 = fma(v82, v44, v80);
        float v84 = float(as_type<half>(ushort(v81 >> 16)));
        float v85 = fma(v84, v45, v83);
        uint v86 = v79.y;
        float v87 = float(as_type<half>(ushort(v86)));
        float v88 = fma(v87, v47, v85);
        float v89 = float(as_type<half>(ushort(v86 >> 16)));
        float v90 = fma(v89, v48, v88);
        uint v91 = v79.z;
        float v92 = float(as_type<half>(ushort(v91)));
        float v93 = fma(v92, v50, v90);
        float v94 = float(as_type<half>(ushort(v91 >> 16)));
        float v95 = fma(v94, v51, v93);
        uint v96 = v79.w;
        float v97 = float(as_type<half>(ushort(v96)));
        float v98 = fma(v97, v53, v95);
        float v99 = float(as_type<half>(ushort(v96 >> 16)));
        float v100 = fma(v99, v54, v98);
        m1 = v100;
        uint v101 = v28 + v40;
        uint4 v102 = b0[v101];
        float v103 = m2;
        uint v104 = v102.x;
        float v105 = float(as_type<half>(ushort(v104)));
        float v106 = fma(v105, v44, v103);
        float v107 = float(as_type<half>(ushort(v104 >> 16)));
        float v108 = fma(v107, v45, v106);
        uint v109 = v102.y;
        float v110 = float(as_type<half>(ushort(v109)));
        float v111 = fma(v110, v47, v108);
        float v112 = float(as_type<half>(ushort(v109 >> 16)));
        float v113 = fma(v112, v48, v111);
        uint v114 = v102.z;
        float v115 = float(as_type<half>(ushort(v114)));
        float v116 = fma(v115, v50, v113);
        float v117 = float(as_type<half>(ushort(v114 >> 16)));
        float v118 = fma(v117, v51, v116);
        uint v119 = v102.w;
        float v120 = float(as_type<half>(ushort(v119)));
        float v121 = fma(v120, v53, v118);
        float v122 = float(as_type<half>(ushort(v119 >> 16)));
        float v123 = fma(v122, v54, v121);
        m2 = v123;
        uint v124 = v33 + v40;
        uint4 v125 = b0[v124];
        float v126 = m3;
        uint v127 = v125.x;
        float v128 = float(as_type<half>(ushort(v127)));
        float v129 = fma(v128, v44, v126);
        float v130 = float(as_type<half>(ushort(v127 >> 16)));
        float v131 = fma(v130, v45, v129);
        uint v132 = v125.y;
        float v133 = float(as_type<half>(ushort(v132)));
        float v134 = fma(v133, v47, v131);
        float v135 = float(as_type<half>(ushort(v132 >> 16)));
        float v136 = fma(v135, v48, v134);
        uint v137 = v125.z;
        float v138 = float(as_type<half>(ushort(v137)));
        float v139 = fma(v138, v50, v136);
        float v140 = float(as_type<half>(ushort(v137 >> 16)));
        float v141 = fma(v140, v51, v139);
        uint v142 = v125.w;
        float v143 = float(as_type<half>(ushort(v142)));
        float v144 = fma(v143, v53, v141);
        float v145 = float(as_type<half>(ushort(v142 >> 16)));
        float v146 = fma(v145, v54, v144);
        m3 = v146;
    }
    bool v148 = lane == 0u;
    uint v149 = v3 * 4u;
    float v150 = m0;
    float v151 = simd_sum(v150);
    uint v153 = v149 + 0u;
    uint v154 = v153 * 2u;
    uint v155 = v154 + v4;
    if (v148) {
        s0[v155] = v151;
    }
    float v156 = m1;
    float v157 = simd_sum(v156);
    uint v159 = v149 + 1u;
    uint v160 = v159 * 2u;
    uint v161 = v160 + v4;
    if (v148) {
        s0[v161] = v157;
    }
    float v162 = m2;
    float v163 = simd_sum(v162);
    uint v165 = v149 + 2u;
    uint v166 = v165 * 2u;
    uint v167 = v166 + v4;
    if (v148) {
        s0[v167] = v163;
    }
    float v168 = m3;
    float v169 = simd_sum(v168);
    uint v171 = v149 + 3u;
    uint v172 = v171 * 2u;
    uint v173 = v172 + v4;
    if (v148) {
        s0[v173] = v169;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v174 = v4 == 0u;
    bool v175 = v148 && v174;
    uint v177 = v10 + 0u;
    bool v178 = v177 < 1024u;
    bool v179 = v175 && v178;
    if (v179) {
        uint v181 = v149 + 0u;
        uint v182 = v181 * 2u;
        float v183 = s0[v182];
        uint v185 = v182 + 1u;
        float v186 = s0[v185];
        float v187 = v183 + v186;
        float v188 = float(half(v187));
        uint v189 = uint(b2[v177]);
        float v190 = float(as_type<half>(ushort(v189)));
        float v191 = v190 + v188;
        uint v192 = uint(as_type<ushort>(half(v191)));
        b2[v177] = ushort(v192);
    }
    uint v194 = v10 + 1u;
    bool v195 = v194 < 1024u;
    bool v196 = v175 && v195;
    if (v196) {
        uint v198 = v149 + 1u;
        uint v199 = v198 * 2u;
        float v200 = s0[v199];
        uint v202 = v199 + 1u;
        float v203 = s0[v202];
        float v204 = v200 + v203;
        float v205 = float(half(v204));
        uint v206 = uint(b2[v194]);
        float v207 = float(as_type<half>(ushort(v206)));
        float v208 = v207 + v205;
        uint v209 = uint(as_type<ushort>(half(v208)));
        b2[v194] = ushort(v209);
    }
    uint v211 = v10 + 2u;
    bool v212 = v211 < 1024u;
    bool v213 = v175 && v212;
    if (v213) {
        uint v215 = v149 + 2u;
        uint v216 = v215 * 2u;
        float v217 = s0[v216];
        uint v219 = v216 + 1u;
        float v220 = s0[v219];
        float v221 = v217 + v220;
        float v222 = float(half(v221));
        uint v223 = uint(b2[v211]);
        float v224 = float(as_type<half>(ushort(v223)));
        float v225 = v224 + v222;
        uint v226 = uint(as_type<ushort>(half(v225)));
        b2[v211] = ushort(v226);
    }
    uint v228 = v10 + 3u;
    bool v229 = v228 < 1024u;
    bool v230 = v175 && v229;
    if (v230) {
        uint v232 = v149 + 3u;
        uint v233 = v232 * 2u;
        float v234 = s0[v233];
        uint v236 = v233 + 1u;
        float v237 = s0[v236];
        float v238 = v234 + v237;
        float v239 = float(half(v238));
        uint v240 = uint(b2[v228]);
        float v241 = float(as_type<half>(ushort(v240)));
        float v242 = v241 + v239;
        uint v243 = uint(as_type<ushort>(half(v242)));
        b2[v228] = ushort(v243);
    }
}
