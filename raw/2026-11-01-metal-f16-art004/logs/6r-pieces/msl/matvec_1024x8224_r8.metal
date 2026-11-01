#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x8224_r8(
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
    uint v3 = tgid.x * 4u;
    uint v4 = v3 + sgid;
    uint v6 = v4 * 8u;
    bool v8 = v6 < 8224u;
    if (v8) {
        uint v12 = v6 + 0u;
        bool v13 = v12 < 8224u;
        uint v14 = v13 ? v12 : 8223u;
        uint v15 = v14 * 128u;
        uint v17 = v6 + 1u;
        bool v18 = v17 < 8224u;
        uint v19 = v18 ? v17 : 8223u;
        uint v20 = v19 * 128u;
        uint v22 = v6 + 2u;
        bool v23 = v22 < 8224u;
        uint v24 = v23 ? v22 : 8223u;
        uint v25 = v24 * 128u;
        uint v27 = v6 + 3u;
        bool v28 = v27 < 8224u;
        uint v29 = v28 ? v27 : 8223u;
        uint v30 = v29 * 128u;
        uint v32 = v6 + 4u;
        bool v33 = v32 < 8224u;
        uint v34 = v33 ? v32 : 8223u;
        uint v35 = v34 * 128u;
        uint v37 = v6 + 5u;
        bool v38 = v37 < 8224u;
        uint v39 = v38 ? v37 : 8223u;
        uint v40 = v39 * 128u;
        uint v42 = v6 + 6u;
        bool v43 = v42 < 8224u;
        uint v44 = v43 ? v42 : 8223u;
        uint v45 = v44 * 128u;
        uint v47 = v6 + 7u;
        bool v48 = v47 < 8224u;
        uint v49 = v48 ? v47 : 8223u;
        uint v50 = v49 * 128u;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        float m2 = as_type<float>(0x00000000u);
        float m3 = as_type<float>(0x00000000u);
        float m4 = as_type<float>(0x00000000u);
        float m5 = as_type<float>(0x00000000u);
        float m6 = as_type<float>(0x00000000u);
        float m7 = as_type<float>(0x00000000u);
        for (uint v54 = lane; v54 < 128u; v54 += 32u) {
            uint4 v55 = b1[v54];
            uint v56 = v55.x;
            float v57 = float(as_type<half>(ushort(v56)));
            float v58 = float(as_type<half>(ushort(v56 >> 16)));
            uint v59 = v55.y;
            float v60 = float(as_type<half>(ushort(v59)));
            float v61 = float(as_type<half>(ushort(v59 >> 16)));
            uint v62 = v55.z;
            float v63 = float(as_type<half>(ushort(v62)));
            float v64 = float(as_type<half>(ushort(v62 >> 16)));
            uint v65 = v55.w;
            float v66 = float(as_type<half>(ushort(v65)));
            float v67 = float(as_type<half>(ushort(v65 >> 16)));
            uint v68 = v15 + v54;
            uint4 v69 = b0[v68];
            float v70 = m0;
            uint v71 = v69.x;
            float v72 = float(as_type<half>(ushort(v71)));
            float v73 = fma(v72, v57, v70);
            float v74 = float(as_type<half>(ushort(v71 >> 16)));
            float v75 = fma(v74, v58, v73);
            uint v76 = v69.y;
            float v77 = float(as_type<half>(ushort(v76)));
            float v78 = fma(v77, v60, v75);
            float v79 = float(as_type<half>(ushort(v76 >> 16)));
            float v80 = fma(v79, v61, v78);
            uint v81 = v69.z;
            float v82 = float(as_type<half>(ushort(v81)));
            float v83 = fma(v82, v63, v80);
            float v84 = float(as_type<half>(ushort(v81 >> 16)));
            float v85 = fma(v84, v64, v83);
            uint v86 = v69.w;
            float v87 = float(as_type<half>(ushort(v86)));
            float v88 = fma(v87, v66, v85);
            float v89 = float(as_type<half>(ushort(v86 >> 16)));
            float v90 = fma(v89, v67, v88);
            m0 = v90;
            uint v91 = v20 + v54;
            uint4 v92 = b0[v91];
            float v93 = m1;
            uint v94 = v92.x;
            float v95 = float(as_type<half>(ushort(v94)));
            float v96 = fma(v95, v57, v93);
            float v97 = float(as_type<half>(ushort(v94 >> 16)));
            float v98 = fma(v97, v58, v96);
            uint v99 = v92.y;
            float v100 = float(as_type<half>(ushort(v99)));
            float v101 = fma(v100, v60, v98);
            float v102 = float(as_type<half>(ushort(v99 >> 16)));
            float v103 = fma(v102, v61, v101);
            uint v104 = v92.z;
            float v105 = float(as_type<half>(ushort(v104)));
            float v106 = fma(v105, v63, v103);
            float v107 = float(as_type<half>(ushort(v104 >> 16)));
            float v108 = fma(v107, v64, v106);
            uint v109 = v92.w;
            float v110 = float(as_type<half>(ushort(v109)));
            float v111 = fma(v110, v66, v108);
            float v112 = float(as_type<half>(ushort(v109 >> 16)));
            float v113 = fma(v112, v67, v111);
            m1 = v113;
            uint v114 = v25 + v54;
            uint4 v115 = b0[v114];
            float v116 = m2;
            uint v117 = v115.x;
            float v118 = float(as_type<half>(ushort(v117)));
            float v119 = fma(v118, v57, v116);
            float v120 = float(as_type<half>(ushort(v117 >> 16)));
            float v121 = fma(v120, v58, v119);
            uint v122 = v115.y;
            float v123 = float(as_type<half>(ushort(v122)));
            float v124 = fma(v123, v60, v121);
            float v125 = float(as_type<half>(ushort(v122 >> 16)));
            float v126 = fma(v125, v61, v124);
            uint v127 = v115.z;
            float v128 = float(as_type<half>(ushort(v127)));
            float v129 = fma(v128, v63, v126);
            float v130 = float(as_type<half>(ushort(v127 >> 16)));
            float v131 = fma(v130, v64, v129);
            uint v132 = v115.w;
            float v133 = float(as_type<half>(ushort(v132)));
            float v134 = fma(v133, v66, v131);
            float v135 = float(as_type<half>(ushort(v132 >> 16)));
            float v136 = fma(v135, v67, v134);
            m2 = v136;
            uint v137 = v30 + v54;
            uint4 v138 = b0[v137];
            float v139 = m3;
            uint v140 = v138.x;
            float v141 = float(as_type<half>(ushort(v140)));
            float v142 = fma(v141, v57, v139);
            float v143 = float(as_type<half>(ushort(v140 >> 16)));
            float v144 = fma(v143, v58, v142);
            uint v145 = v138.y;
            float v146 = float(as_type<half>(ushort(v145)));
            float v147 = fma(v146, v60, v144);
            float v148 = float(as_type<half>(ushort(v145 >> 16)));
            float v149 = fma(v148, v61, v147);
            uint v150 = v138.z;
            float v151 = float(as_type<half>(ushort(v150)));
            float v152 = fma(v151, v63, v149);
            float v153 = float(as_type<half>(ushort(v150 >> 16)));
            float v154 = fma(v153, v64, v152);
            uint v155 = v138.w;
            float v156 = float(as_type<half>(ushort(v155)));
            float v157 = fma(v156, v66, v154);
            float v158 = float(as_type<half>(ushort(v155 >> 16)));
            float v159 = fma(v158, v67, v157);
            m3 = v159;
            uint v160 = v35 + v54;
            uint4 v161 = b0[v160];
            float v162 = m4;
            uint v163 = v161.x;
            float v164 = float(as_type<half>(ushort(v163)));
            float v165 = fma(v164, v57, v162);
            float v166 = float(as_type<half>(ushort(v163 >> 16)));
            float v167 = fma(v166, v58, v165);
            uint v168 = v161.y;
            float v169 = float(as_type<half>(ushort(v168)));
            float v170 = fma(v169, v60, v167);
            float v171 = float(as_type<half>(ushort(v168 >> 16)));
            float v172 = fma(v171, v61, v170);
            uint v173 = v161.z;
            float v174 = float(as_type<half>(ushort(v173)));
            float v175 = fma(v174, v63, v172);
            float v176 = float(as_type<half>(ushort(v173 >> 16)));
            float v177 = fma(v176, v64, v175);
            uint v178 = v161.w;
            float v179 = float(as_type<half>(ushort(v178)));
            float v180 = fma(v179, v66, v177);
            float v181 = float(as_type<half>(ushort(v178 >> 16)));
            float v182 = fma(v181, v67, v180);
            m4 = v182;
            uint v183 = v40 + v54;
            uint4 v184 = b0[v183];
            float v185 = m5;
            uint v186 = v184.x;
            float v187 = float(as_type<half>(ushort(v186)));
            float v188 = fma(v187, v57, v185);
            float v189 = float(as_type<half>(ushort(v186 >> 16)));
            float v190 = fma(v189, v58, v188);
            uint v191 = v184.y;
            float v192 = float(as_type<half>(ushort(v191)));
            float v193 = fma(v192, v60, v190);
            float v194 = float(as_type<half>(ushort(v191 >> 16)));
            float v195 = fma(v194, v61, v193);
            uint v196 = v184.z;
            float v197 = float(as_type<half>(ushort(v196)));
            float v198 = fma(v197, v63, v195);
            float v199 = float(as_type<half>(ushort(v196 >> 16)));
            float v200 = fma(v199, v64, v198);
            uint v201 = v184.w;
            float v202 = float(as_type<half>(ushort(v201)));
            float v203 = fma(v202, v66, v200);
            float v204 = float(as_type<half>(ushort(v201 >> 16)));
            float v205 = fma(v204, v67, v203);
            m5 = v205;
            uint v206 = v45 + v54;
            uint4 v207 = b0[v206];
            float v208 = m6;
            uint v209 = v207.x;
            float v210 = float(as_type<half>(ushort(v209)));
            float v211 = fma(v210, v57, v208);
            float v212 = float(as_type<half>(ushort(v209 >> 16)));
            float v213 = fma(v212, v58, v211);
            uint v214 = v207.y;
            float v215 = float(as_type<half>(ushort(v214)));
            float v216 = fma(v215, v60, v213);
            float v217 = float(as_type<half>(ushort(v214 >> 16)));
            float v218 = fma(v217, v61, v216);
            uint v219 = v207.z;
            float v220 = float(as_type<half>(ushort(v219)));
            float v221 = fma(v220, v63, v218);
            float v222 = float(as_type<half>(ushort(v219 >> 16)));
            float v223 = fma(v222, v64, v221);
            uint v224 = v207.w;
            float v225 = float(as_type<half>(ushort(v224)));
            float v226 = fma(v225, v66, v223);
            float v227 = float(as_type<half>(ushort(v224 >> 16)));
            float v228 = fma(v227, v67, v226);
            m6 = v228;
            uint v229 = v50 + v54;
            uint4 v230 = b0[v229];
            float v231 = m7;
            uint v232 = v230.x;
            float v233 = float(as_type<half>(ushort(v232)));
            float v234 = fma(v233, v57, v231);
            float v235 = float(as_type<half>(ushort(v232 >> 16)));
            float v236 = fma(v235, v58, v234);
            uint v237 = v230.y;
            float v238 = float(as_type<half>(ushort(v237)));
            float v239 = fma(v238, v60, v236);
            float v240 = float(as_type<half>(ushort(v237 >> 16)));
            float v241 = fma(v240, v61, v239);
            uint v242 = v230.z;
            float v243 = float(as_type<half>(ushort(v242)));
            float v244 = fma(v243, v63, v241);
            float v245 = float(as_type<half>(ushort(v242 >> 16)));
            float v246 = fma(v245, v64, v244);
            uint v247 = v230.w;
            float v248 = float(as_type<half>(ushort(v247)));
            float v249 = fma(v248, v66, v246);
            float v250 = float(as_type<half>(ushort(v247 >> 16)));
            float v251 = fma(v250, v67, v249);
            m7 = v251;
        }
        bool v253 = lane == 0u;
        float v254 = m0;
        float v255 = simd_sum(v254);
        uint v257 = v6 + 0u;
        bool v258 = v257 < 8224u;
        bool v259 = v253 && v258;
        if (v259) {
            uint v260 = uint(as_type<ushort>(half(v255)));
            b2[v257] = ushort(v260);
        }
        float v261 = m1;
        float v262 = simd_sum(v261);
        uint v264 = v6 + 1u;
        bool v265 = v264 < 8224u;
        bool v266 = v253 && v265;
        if (v266) {
            uint v267 = uint(as_type<ushort>(half(v262)));
            b2[v264] = ushort(v267);
        }
        float v268 = m2;
        float v269 = simd_sum(v268);
        uint v271 = v6 + 2u;
        bool v272 = v271 < 8224u;
        bool v273 = v253 && v272;
        if (v273) {
            uint v274 = uint(as_type<ushort>(half(v269)));
            b2[v271] = ushort(v274);
        }
        float v275 = m3;
        float v276 = simd_sum(v275);
        uint v278 = v6 + 3u;
        bool v279 = v278 < 8224u;
        bool v280 = v253 && v279;
        if (v280) {
            uint v281 = uint(as_type<ushort>(half(v276)));
            b2[v278] = ushort(v281);
        }
        float v282 = m4;
        float v283 = simd_sum(v282);
        uint v285 = v6 + 4u;
        bool v286 = v285 < 8224u;
        bool v287 = v253 && v286;
        if (v287) {
            uint v288 = uint(as_type<ushort>(half(v283)));
            b2[v285] = ushort(v288);
        }
        float v289 = m5;
        float v290 = simd_sum(v289);
        uint v292 = v6 + 5u;
        bool v293 = v292 < 8224u;
        bool v294 = v253 && v293;
        if (v294) {
            uint v295 = uint(as_type<ushort>(half(v290)));
            b2[v292] = ushort(v295);
        }
        float v296 = m6;
        float v297 = simd_sum(v296);
        uint v299 = v6 + 6u;
        bool v300 = v299 < 8224u;
        bool v301 = v253 && v300;
        if (v301) {
            uint v302 = uint(as_type<ushort>(half(v297)));
            b2[v299] = ushort(v302);
        }
        float v303 = m7;
        float v304 = simd_sum(v303);
        uint v306 = v6 + 7u;
        bool v307 = v306 < 8224u;
        bool v308 = v253 && v307;
        if (v308) {
            uint v309 = uint(as_type<ushort>(half(v304)));
            b2[v306] = ushort(v309);
        }
    }
}
