#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_swi_res_r4_k1_p5120(
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
    uint v3 = sgid / 1u;
    uint v4 = sgid % 1u;
    for (uint v7 = tgid.x; v7 < 64u; v7 += 5120u) {
        uint v9 = v7 * 4u;
        uint v10 = v9 + v3;
        uint v13 = v10 * 4u;
        uint v18 = v13 + 0u;
        bool v19 = v18 < 1024u;
        uint v20 = v19 ? v18 : 1023u;
        uint v21 = v20 * 448u;
        uint v23 = v13 + 1u;
        bool v24 = v23 < 1024u;
        uint v25 = v24 ? v23 : 1023u;
        uint v26 = v25 * 448u;
        uint v28 = v13 + 2u;
        bool v29 = v28 < 1024u;
        uint v30 = v29 ? v28 : 1023u;
        uint v31 = v30 * 448u;
        uint v33 = v13 + 3u;
        bool v34 = v33 < 1024u;
        uint v35 = v34 ? v33 : 1023u;
        uint v36 = v35 * 448u;
        uint v39 = v4 * 32u;
        uint v40 = v39 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        float m2 = as_type<float>(0x00000000u);
        float m3 = as_type<float>(0x00000000u);
        for (uint v43 = v40; v43 < 448u; v43 += 32u) {
            uint4 v45 = b1[v43];
            uint v46 = v43 + 448u;
            uint4 v47 = b1[v46];
            uint v48 = v45.x;
            float v49 = float(as_type<half>(ushort(v48)));
            float v50 = float(as_type<half>(ushort(v48 >> 16)));
            uint v51 = v47.x;
            float v52 = float(as_type<half>(ushort(v51)));
            float v53 = float(as_type<half>(ushort(v51 >> 16)));
            float v54 = -v49;
            float v55 = exp(v54);
            float v57 = as_type<float>(0x3f800000u) + v55;
            float v58 = as_type<float>(0x3f800000u) / v57;
            float v59 = v49 * v58;
            float v60 = v59 * v52;
            float v61 = float(half(v60));
            float v62 = -v50;
            float v63 = exp(v62);
            float v65 = as_type<float>(0x3f800000u) + v63;
            float v66 = as_type<float>(0x3f800000u) / v65;
            float v67 = v50 * v66;
            float v68 = v67 * v53;
            float v69 = float(half(v68));
            uint v70 = v45.y;
            float v71 = float(as_type<half>(ushort(v70)));
            float v72 = float(as_type<half>(ushort(v70 >> 16)));
            uint v73 = v47.y;
            float v74 = float(as_type<half>(ushort(v73)));
            float v75 = float(as_type<half>(ushort(v73 >> 16)));
            float v76 = -v71;
            float v77 = exp(v76);
            float v79 = as_type<float>(0x3f800000u) + v77;
            float v80 = as_type<float>(0x3f800000u) / v79;
            float v81 = v71 * v80;
            float v82 = v81 * v74;
            float v83 = float(half(v82));
            float v84 = -v72;
            float v85 = exp(v84);
            float v87 = as_type<float>(0x3f800000u) + v85;
            float v88 = as_type<float>(0x3f800000u) / v87;
            float v89 = v72 * v88;
            float v90 = v89 * v75;
            float v91 = float(half(v90));
            uint v92 = v45.z;
            float v93 = float(as_type<half>(ushort(v92)));
            float v94 = float(as_type<half>(ushort(v92 >> 16)));
            uint v95 = v47.z;
            float v96 = float(as_type<half>(ushort(v95)));
            float v97 = float(as_type<half>(ushort(v95 >> 16)));
            float v98 = -v93;
            float v99 = exp(v98);
            float v101 = as_type<float>(0x3f800000u) + v99;
            float v102 = as_type<float>(0x3f800000u) / v101;
            float v103 = v93 * v102;
            float v104 = v103 * v96;
            float v105 = float(half(v104));
            float v106 = -v94;
            float v107 = exp(v106);
            float v109 = as_type<float>(0x3f800000u) + v107;
            float v110 = as_type<float>(0x3f800000u) / v109;
            float v111 = v94 * v110;
            float v112 = v111 * v97;
            float v113 = float(half(v112));
            uint v114 = v45.w;
            float v115 = float(as_type<half>(ushort(v114)));
            float v116 = float(as_type<half>(ushort(v114 >> 16)));
            uint v117 = v47.w;
            float v118 = float(as_type<half>(ushort(v117)));
            float v119 = float(as_type<half>(ushort(v117 >> 16)));
            float v120 = -v115;
            float v121 = exp(v120);
            float v123 = as_type<float>(0x3f800000u) + v121;
            float v124 = as_type<float>(0x3f800000u) / v123;
            float v125 = v115 * v124;
            float v126 = v125 * v118;
            float v127 = float(half(v126));
            float v128 = -v116;
            float v129 = exp(v128);
            float v131 = as_type<float>(0x3f800000u) + v129;
            float v132 = as_type<float>(0x3f800000u) / v131;
            float v133 = v116 * v132;
            float v134 = v133 * v119;
            float v135 = float(half(v134));
            uint v136 = v21 + v43;
            uint4 v137 = b0[v136];
            float v138 = m0;
            uint v139 = v137.x;
            float v140 = float(as_type<half>(ushort(v139)));
            float v141 = fma(v140, v61, v138);
            float v142 = float(as_type<half>(ushort(v139 >> 16)));
            float v143 = fma(v142, v69, v141);
            uint v144 = v137.y;
            float v145 = float(as_type<half>(ushort(v144)));
            float v146 = fma(v145, v83, v143);
            float v147 = float(as_type<half>(ushort(v144 >> 16)));
            float v148 = fma(v147, v91, v146);
            uint v149 = v137.z;
            float v150 = float(as_type<half>(ushort(v149)));
            float v151 = fma(v150, v105, v148);
            float v152 = float(as_type<half>(ushort(v149 >> 16)));
            float v153 = fma(v152, v113, v151);
            uint v154 = v137.w;
            float v155 = float(as_type<half>(ushort(v154)));
            float v156 = fma(v155, v127, v153);
            float v157 = float(as_type<half>(ushort(v154 >> 16)));
            float v158 = fma(v157, v135, v156);
            m0 = v158;
            uint v159 = v26 + v43;
            uint4 v160 = b0[v159];
            float v161 = m1;
            uint v162 = v160.x;
            float v163 = float(as_type<half>(ushort(v162)));
            float v164 = fma(v163, v61, v161);
            float v165 = float(as_type<half>(ushort(v162 >> 16)));
            float v166 = fma(v165, v69, v164);
            uint v167 = v160.y;
            float v168 = float(as_type<half>(ushort(v167)));
            float v169 = fma(v168, v83, v166);
            float v170 = float(as_type<half>(ushort(v167 >> 16)));
            float v171 = fma(v170, v91, v169);
            uint v172 = v160.z;
            float v173 = float(as_type<half>(ushort(v172)));
            float v174 = fma(v173, v105, v171);
            float v175 = float(as_type<half>(ushort(v172 >> 16)));
            float v176 = fma(v175, v113, v174);
            uint v177 = v160.w;
            float v178 = float(as_type<half>(ushort(v177)));
            float v179 = fma(v178, v127, v176);
            float v180 = float(as_type<half>(ushort(v177 >> 16)));
            float v181 = fma(v180, v135, v179);
            m1 = v181;
            uint v182 = v31 + v43;
            uint4 v183 = b0[v182];
            float v184 = m2;
            uint v185 = v183.x;
            float v186 = float(as_type<half>(ushort(v185)));
            float v187 = fma(v186, v61, v184);
            float v188 = float(as_type<half>(ushort(v185 >> 16)));
            float v189 = fma(v188, v69, v187);
            uint v190 = v183.y;
            float v191 = float(as_type<half>(ushort(v190)));
            float v192 = fma(v191, v83, v189);
            float v193 = float(as_type<half>(ushort(v190 >> 16)));
            float v194 = fma(v193, v91, v192);
            uint v195 = v183.z;
            float v196 = float(as_type<half>(ushort(v195)));
            float v197 = fma(v196, v105, v194);
            float v198 = float(as_type<half>(ushort(v195 >> 16)));
            float v199 = fma(v198, v113, v197);
            uint v200 = v183.w;
            float v201 = float(as_type<half>(ushort(v200)));
            float v202 = fma(v201, v127, v199);
            float v203 = float(as_type<half>(ushort(v200 >> 16)));
            float v204 = fma(v203, v135, v202);
            m2 = v204;
            uint v205 = v36 + v43;
            uint4 v206 = b0[v205];
            float v207 = m3;
            uint v208 = v206.x;
            float v209 = float(as_type<half>(ushort(v208)));
            float v210 = fma(v209, v61, v207);
            float v211 = float(as_type<half>(ushort(v208 >> 16)));
            float v212 = fma(v211, v69, v210);
            uint v213 = v206.y;
            float v214 = float(as_type<half>(ushort(v213)));
            float v215 = fma(v214, v83, v212);
            float v216 = float(as_type<half>(ushort(v213 >> 16)));
            float v217 = fma(v216, v91, v215);
            uint v218 = v206.z;
            float v219 = float(as_type<half>(ushort(v218)));
            float v220 = fma(v219, v105, v217);
            float v221 = float(as_type<half>(ushort(v218 >> 16)));
            float v222 = fma(v221, v113, v220);
            uint v223 = v206.w;
            float v224 = float(as_type<half>(ushort(v223)));
            float v225 = fma(v224, v127, v222);
            float v226 = float(as_type<half>(ushort(v223 >> 16)));
            float v227 = fma(v226, v135, v225);
            m3 = v227;
        }
        bool v229 = lane == 0u;
        uint v230 = v3 * 4u;
        float v231 = m0;
        float v232 = simd_sum(v231);
        uint v234 = v230 + 0u;
        uint v235 = v234 * 1u;
        uint v236 = v235 + v4;
        if (v229) {
            s0[v236] = v232;
        }
        float v237 = m1;
        float v238 = simd_sum(v237);
        uint v240 = v230 + 1u;
        uint v241 = v240 * 1u;
        uint v242 = v241 + v4;
        if (v229) {
            s0[v242] = v238;
        }
        float v243 = m2;
        float v244 = simd_sum(v243);
        uint v246 = v230 + 2u;
        uint v247 = v246 * 1u;
        uint v248 = v247 + v4;
        if (v229) {
            s0[v248] = v244;
        }
        float v249 = m3;
        float v250 = simd_sum(v249);
        uint v252 = v230 + 3u;
        uint v253 = v252 * 1u;
        uint v254 = v253 + v4;
        if (v229) {
            s0[v254] = v250;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v255 = v4 == 0u;
        bool v256 = v229 && v255;
        uint v258 = v13 + 0u;
        bool v259 = v258 < 1024u;
        bool v260 = v256 && v259;
        if (v260) {
            uint v262 = v230 + 0u;
            uint v263 = v262 * 1u;
            float v264 = s0[v263];
            float v265 = float(half(v264));
            uint v266 = uint(b2[v258]);
            float v267 = float(as_type<half>(ushort(v266)));
            float v268 = v267 + v265;
            uint v269 = uint(as_type<ushort>(half(v268)));
            b2[v258] = ushort(v269);
        }
        uint v271 = v13 + 1u;
        bool v272 = v271 < 1024u;
        bool v273 = v256 && v272;
        if (v273) {
            uint v275 = v230 + 1u;
            uint v276 = v275 * 1u;
            float v277 = s0[v276];
            float v278 = float(half(v277));
            uint v279 = uint(b2[v271]);
            float v280 = float(as_type<half>(ushort(v279)));
            float v281 = v280 + v278;
            uint v282 = uint(as_type<ushort>(half(v281)));
            b2[v271] = ushort(v282);
        }
        uint v284 = v13 + 2u;
        bool v285 = v284 < 1024u;
        bool v286 = v256 && v285;
        if (v286) {
            uint v288 = v230 + 2u;
            uint v289 = v288 * 1u;
            float v290 = s0[v289];
            float v291 = float(half(v290));
            uint v292 = uint(b2[v284]);
            float v293 = float(as_type<half>(ushort(v292)));
            float v294 = v293 + v291;
            uint v295 = uint(as_type<ushort>(half(v294)));
            b2[v284] = ushort(v295);
        }
        uint v297 = v13 + 3u;
        bool v298 = v297 < 1024u;
        bool v299 = v256 && v298;
        if (v299) {
            uint v301 = v230 + 3u;
            uint v302 = v301 * 1u;
            float v303 = s0[v302];
            float v304 = float(half(v303));
            uint v305 = uint(b2[v297]);
            float v306 = float(as_type<half>(ushort(v305)));
            float v307 = v306 + v304;
            uint v308 = uint(as_type<ushort>(half(v307)));
            b2[v297] = ushort(v308);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
