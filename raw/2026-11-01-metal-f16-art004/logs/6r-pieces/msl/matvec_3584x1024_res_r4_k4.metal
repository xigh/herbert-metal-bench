#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_res_r4_k4(
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
    uint v3 = sgid / 4u;
    uint v4 = sgid % 4u;
    uint v6 = tgid.x * 1u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 4u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 1024u;
    uint v17 = v16 ? v15 : 1023u;
    uint v18 = v17 * 448u;
    uint v20 = v10 + 1u;
    bool v21 = v20 < 1024u;
    uint v22 = v21 ? v20 : 1023u;
    uint v23 = v22 * 448u;
    uint v25 = v10 + 2u;
    bool v26 = v25 < 1024u;
    uint v27 = v26 ? v25 : 1023u;
    uint v28 = v27 * 448u;
    uint v30 = v10 + 3u;
    bool v31 = v30 < 1024u;
    uint v32 = v31 ? v30 : 1023u;
    uint v33 = v32 * 448u;
    uint v36 = v4 * 32u;
    uint v37 = v36 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    for (uint v40 = v37; v40 < 448u; v40 += 128u) {
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
    uint v154 = v153 * 4u;
    uint v155 = v154 + v4;
    if (v148) {
        s0[v155] = v151;
    }
    float v156 = m1;
    float v157 = simd_sum(v156);
    uint v159 = v149 + 1u;
    uint v160 = v159 * 4u;
    uint v161 = v160 + v4;
    if (v148) {
        s0[v161] = v157;
    }
    float v162 = m2;
    float v163 = simd_sum(v162);
    uint v165 = v149 + 2u;
    uint v166 = v165 * 4u;
    uint v167 = v166 + v4;
    if (v148) {
        s0[v167] = v163;
    }
    float v168 = m3;
    float v169 = simd_sum(v168);
    uint v171 = v149 + 3u;
    uint v172 = v171 * 4u;
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
        uint v182 = v181 * 4u;
        float v183 = s0[v182];
        uint v185 = v182 + 1u;
        float v186 = s0[v185];
        float v187 = v183 + v186;
        uint v189 = v182 + 2u;
        float v190 = s0[v189];
        float v191 = v187 + v190;
        uint v193 = v182 + 3u;
        float v194 = s0[v193];
        float v195 = v191 + v194;
        float v196 = float(half(v195));
        uint v197 = uint(b2[v177]);
        float v198 = float(as_type<half>(ushort(v197)));
        float v199 = v198 + v196;
        uint v200 = uint(as_type<ushort>(half(v199)));
        b2[v177] = ushort(v200);
    }
    uint v202 = v10 + 1u;
    bool v203 = v202 < 1024u;
    bool v204 = v175 && v203;
    if (v204) {
        uint v206 = v149 + 1u;
        uint v207 = v206 * 4u;
        float v208 = s0[v207];
        uint v210 = v207 + 1u;
        float v211 = s0[v210];
        float v212 = v208 + v211;
        uint v214 = v207 + 2u;
        float v215 = s0[v214];
        float v216 = v212 + v215;
        uint v218 = v207 + 3u;
        float v219 = s0[v218];
        float v220 = v216 + v219;
        float v221 = float(half(v220));
        uint v222 = uint(b2[v202]);
        float v223 = float(as_type<half>(ushort(v222)));
        float v224 = v223 + v221;
        uint v225 = uint(as_type<ushort>(half(v224)));
        b2[v202] = ushort(v225);
    }
    uint v227 = v10 + 2u;
    bool v228 = v227 < 1024u;
    bool v229 = v175 && v228;
    if (v229) {
        uint v231 = v149 + 2u;
        uint v232 = v231 * 4u;
        float v233 = s0[v232];
        uint v235 = v232 + 1u;
        float v236 = s0[v235];
        float v237 = v233 + v236;
        uint v239 = v232 + 2u;
        float v240 = s0[v239];
        float v241 = v237 + v240;
        uint v243 = v232 + 3u;
        float v244 = s0[v243];
        float v245 = v241 + v244;
        float v246 = float(half(v245));
        uint v247 = uint(b2[v227]);
        float v248 = float(as_type<half>(ushort(v247)));
        float v249 = v248 + v246;
        uint v250 = uint(as_type<ushort>(half(v249)));
        b2[v227] = ushort(v250);
    }
    uint v252 = v10 + 3u;
    bool v253 = v252 < 1024u;
    bool v254 = v175 && v253;
    if (v254) {
        uint v256 = v149 + 3u;
        uint v257 = v256 * 4u;
        float v258 = s0[v257];
        uint v260 = v257 + 1u;
        float v261 = s0[v260];
        float v262 = v258 + v261;
        uint v264 = v257 + 2u;
        float v265 = s0[v264];
        float v266 = v262 + v265;
        uint v268 = v257 + 3u;
        float v269 = s0[v268];
        float v270 = v266 + v269;
        float v271 = float(half(v270));
        uint v272 = uint(b2[v252]);
        float v273 = float(as_type<half>(ushort(v272)));
        float v274 = v273 + v271;
        uint v275 = uint(as_type<ushort>(half(v274)));
        b2[v252] = ushort(v275);
    }
}
