#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x5120_r8_k2(
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
    threadgroup float s0[32];
    uint v3 = sgid / 2u;
    uint v4 = sgid % 2u;
    uint v6 = tgid.x * 2u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 8u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 5120u;
    uint v17 = v16 ? v15 : 5119u;
    uint v18 = v17 * 128u;
    uint v20 = v10 + 1u;
    bool v21 = v20 < 5120u;
    uint v22 = v21 ? v20 : 5119u;
    uint v23 = v22 * 128u;
    uint v25 = v10 + 2u;
    bool v26 = v25 < 5120u;
    uint v27 = v26 ? v25 : 5119u;
    uint v28 = v27 * 128u;
    uint v30 = v10 + 3u;
    bool v31 = v30 < 5120u;
    uint v32 = v31 ? v30 : 5119u;
    uint v33 = v32 * 128u;
    uint v35 = v10 + 4u;
    bool v36 = v35 < 5120u;
    uint v37 = v36 ? v35 : 5119u;
    uint v38 = v37 * 128u;
    uint v40 = v10 + 5u;
    bool v41 = v40 < 5120u;
    uint v42 = v41 ? v40 : 5119u;
    uint v43 = v42 * 128u;
    uint v45 = v10 + 6u;
    bool v46 = v45 < 5120u;
    uint v47 = v46 ? v45 : 5119u;
    uint v48 = v47 * 128u;
    uint v50 = v10 + 7u;
    bool v51 = v50 < 5120u;
    uint v52 = v51 ? v50 : 5119u;
    uint v53 = v52 * 128u;
    uint v56 = v4 * 32u;
    uint v57 = v56 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    float m4 = as_type<float>(0x00000000u);
    float m5 = as_type<float>(0x00000000u);
    float m6 = as_type<float>(0x00000000u);
    float m7 = as_type<float>(0x00000000u);
    for (uint v60 = v57; v60 < 128u; v60 += 64u) {
        uint4 v62 = b1[v60];
        uint v63 = v62.x;
        float v64 = float(as_type<half>(ushort(v63)));
        float v65 = float(as_type<half>(ushort(v63 >> 16)));
        uint v66 = v62.y;
        float v67 = float(as_type<half>(ushort(v66)));
        float v68 = float(as_type<half>(ushort(v66 >> 16)));
        uint v69 = v62.z;
        float v70 = float(as_type<half>(ushort(v69)));
        float v71 = float(as_type<half>(ushort(v69 >> 16)));
        uint v72 = v62.w;
        float v73 = float(as_type<half>(ushort(v72)));
        float v74 = float(as_type<half>(ushort(v72 >> 16)));
        uint v75 = v18 + v60;
        uint4 v76 = b0[v75];
        float v77 = m0;
        uint v78 = v76.x;
        float v79 = float(as_type<half>(ushort(v78)));
        float v80 = fma(v79, v64, v77);
        float v81 = float(as_type<half>(ushort(v78 >> 16)));
        float v82 = fma(v81, v65, v80);
        uint v83 = v76.y;
        float v84 = float(as_type<half>(ushort(v83)));
        float v85 = fma(v84, v67, v82);
        float v86 = float(as_type<half>(ushort(v83 >> 16)));
        float v87 = fma(v86, v68, v85);
        uint v88 = v76.z;
        float v89 = float(as_type<half>(ushort(v88)));
        float v90 = fma(v89, v70, v87);
        float v91 = float(as_type<half>(ushort(v88 >> 16)));
        float v92 = fma(v91, v71, v90);
        uint v93 = v76.w;
        float v94 = float(as_type<half>(ushort(v93)));
        float v95 = fma(v94, v73, v92);
        float v96 = float(as_type<half>(ushort(v93 >> 16)));
        float v97 = fma(v96, v74, v95);
        m0 = v97;
        uint v98 = v23 + v60;
        uint4 v99 = b0[v98];
        float v100 = m1;
        uint v101 = v99.x;
        float v102 = float(as_type<half>(ushort(v101)));
        float v103 = fma(v102, v64, v100);
        float v104 = float(as_type<half>(ushort(v101 >> 16)));
        float v105 = fma(v104, v65, v103);
        uint v106 = v99.y;
        float v107 = float(as_type<half>(ushort(v106)));
        float v108 = fma(v107, v67, v105);
        float v109 = float(as_type<half>(ushort(v106 >> 16)));
        float v110 = fma(v109, v68, v108);
        uint v111 = v99.z;
        float v112 = float(as_type<half>(ushort(v111)));
        float v113 = fma(v112, v70, v110);
        float v114 = float(as_type<half>(ushort(v111 >> 16)));
        float v115 = fma(v114, v71, v113);
        uint v116 = v99.w;
        float v117 = float(as_type<half>(ushort(v116)));
        float v118 = fma(v117, v73, v115);
        float v119 = float(as_type<half>(ushort(v116 >> 16)));
        float v120 = fma(v119, v74, v118);
        m1 = v120;
        uint v121 = v28 + v60;
        uint4 v122 = b0[v121];
        float v123 = m2;
        uint v124 = v122.x;
        float v125 = float(as_type<half>(ushort(v124)));
        float v126 = fma(v125, v64, v123);
        float v127 = float(as_type<half>(ushort(v124 >> 16)));
        float v128 = fma(v127, v65, v126);
        uint v129 = v122.y;
        float v130 = float(as_type<half>(ushort(v129)));
        float v131 = fma(v130, v67, v128);
        float v132 = float(as_type<half>(ushort(v129 >> 16)));
        float v133 = fma(v132, v68, v131);
        uint v134 = v122.z;
        float v135 = float(as_type<half>(ushort(v134)));
        float v136 = fma(v135, v70, v133);
        float v137 = float(as_type<half>(ushort(v134 >> 16)));
        float v138 = fma(v137, v71, v136);
        uint v139 = v122.w;
        float v140 = float(as_type<half>(ushort(v139)));
        float v141 = fma(v140, v73, v138);
        float v142 = float(as_type<half>(ushort(v139 >> 16)));
        float v143 = fma(v142, v74, v141);
        m2 = v143;
        uint v144 = v33 + v60;
        uint4 v145 = b0[v144];
        float v146 = m3;
        uint v147 = v145.x;
        float v148 = float(as_type<half>(ushort(v147)));
        float v149 = fma(v148, v64, v146);
        float v150 = float(as_type<half>(ushort(v147 >> 16)));
        float v151 = fma(v150, v65, v149);
        uint v152 = v145.y;
        float v153 = float(as_type<half>(ushort(v152)));
        float v154 = fma(v153, v67, v151);
        float v155 = float(as_type<half>(ushort(v152 >> 16)));
        float v156 = fma(v155, v68, v154);
        uint v157 = v145.z;
        float v158 = float(as_type<half>(ushort(v157)));
        float v159 = fma(v158, v70, v156);
        float v160 = float(as_type<half>(ushort(v157 >> 16)));
        float v161 = fma(v160, v71, v159);
        uint v162 = v145.w;
        float v163 = float(as_type<half>(ushort(v162)));
        float v164 = fma(v163, v73, v161);
        float v165 = float(as_type<half>(ushort(v162 >> 16)));
        float v166 = fma(v165, v74, v164);
        m3 = v166;
        uint v167 = v38 + v60;
        uint4 v168 = b0[v167];
        float v169 = m4;
        uint v170 = v168.x;
        float v171 = float(as_type<half>(ushort(v170)));
        float v172 = fma(v171, v64, v169);
        float v173 = float(as_type<half>(ushort(v170 >> 16)));
        float v174 = fma(v173, v65, v172);
        uint v175 = v168.y;
        float v176 = float(as_type<half>(ushort(v175)));
        float v177 = fma(v176, v67, v174);
        float v178 = float(as_type<half>(ushort(v175 >> 16)));
        float v179 = fma(v178, v68, v177);
        uint v180 = v168.z;
        float v181 = float(as_type<half>(ushort(v180)));
        float v182 = fma(v181, v70, v179);
        float v183 = float(as_type<half>(ushort(v180 >> 16)));
        float v184 = fma(v183, v71, v182);
        uint v185 = v168.w;
        float v186 = float(as_type<half>(ushort(v185)));
        float v187 = fma(v186, v73, v184);
        float v188 = float(as_type<half>(ushort(v185 >> 16)));
        float v189 = fma(v188, v74, v187);
        m4 = v189;
        uint v190 = v43 + v60;
        uint4 v191 = b0[v190];
        float v192 = m5;
        uint v193 = v191.x;
        float v194 = float(as_type<half>(ushort(v193)));
        float v195 = fma(v194, v64, v192);
        float v196 = float(as_type<half>(ushort(v193 >> 16)));
        float v197 = fma(v196, v65, v195);
        uint v198 = v191.y;
        float v199 = float(as_type<half>(ushort(v198)));
        float v200 = fma(v199, v67, v197);
        float v201 = float(as_type<half>(ushort(v198 >> 16)));
        float v202 = fma(v201, v68, v200);
        uint v203 = v191.z;
        float v204 = float(as_type<half>(ushort(v203)));
        float v205 = fma(v204, v70, v202);
        float v206 = float(as_type<half>(ushort(v203 >> 16)));
        float v207 = fma(v206, v71, v205);
        uint v208 = v191.w;
        float v209 = float(as_type<half>(ushort(v208)));
        float v210 = fma(v209, v73, v207);
        float v211 = float(as_type<half>(ushort(v208 >> 16)));
        float v212 = fma(v211, v74, v210);
        m5 = v212;
        uint v213 = v48 + v60;
        uint4 v214 = b0[v213];
        float v215 = m6;
        uint v216 = v214.x;
        float v217 = float(as_type<half>(ushort(v216)));
        float v218 = fma(v217, v64, v215);
        float v219 = float(as_type<half>(ushort(v216 >> 16)));
        float v220 = fma(v219, v65, v218);
        uint v221 = v214.y;
        float v222 = float(as_type<half>(ushort(v221)));
        float v223 = fma(v222, v67, v220);
        float v224 = float(as_type<half>(ushort(v221 >> 16)));
        float v225 = fma(v224, v68, v223);
        uint v226 = v214.z;
        float v227 = float(as_type<half>(ushort(v226)));
        float v228 = fma(v227, v70, v225);
        float v229 = float(as_type<half>(ushort(v226 >> 16)));
        float v230 = fma(v229, v71, v228);
        uint v231 = v214.w;
        float v232 = float(as_type<half>(ushort(v231)));
        float v233 = fma(v232, v73, v230);
        float v234 = float(as_type<half>(ushort(v231 >> 16)));
        float v235 = fma(v234, v74, v233);
        m6 = v235;
        uint v236 = v53 + v60;
        uint4 v237 = b0[v236];
        float v238 = m7;
        uint v239 = v237.x;
        float v240 = float(as_type<half>(ushort(v239)));
        float v241 = fma(v240, v64, v238);
        float v242 = float(as_type<half>(ushort(v239 >> 16)));
        float v243 = fma(v242, v65, v241);
        uint v244 = v237.y;
        float v245 = float(as_type<half>(ushort(v244)));
        float v246 = fma(v245, v67, v243);
        float v247 = float(as_type<half>(ushort(v244 >> 16)));
        float v248 = fma(v247, v68, v246);
        uint v249 = v237.z;
        float v250 = float(as_type<half>(ushort(v249)));
        float v251 = fma(v250, v70, v248);
        float v252 = float(as_type<half>(ushort(v249 >> 16)));
        float v253 = fma(v252, v71, v251);
        uint v254 = v237.w;
        float v255 = float(as_type<half>(ushort(v254)));
        float v256 = fma(v255, v73, v253);
        float v257 = float(as_type<half>(ushort(v254 >> 16)));
        float v258 = fma(v257, v74, v256);
        m7 = v258;
    }
    bool v260 = lane == 0u;
    uint v261 = v3 * 8u;
    float v262 = m0;
    float v263 = simd_sum(v262);
    uint v265 = v261 + 0u;
    uint v266 = v265 * 2u;
    uint v267 = v266 + v4;
    if (v260) {
        s0[v267] = v263;
    }
    float v268 = m1;
    float v269 = simd_sum(v268);
    uint v271 = v261 + 1u;
    uint v272 = v271 * 2u;
    uint v273 = v272 + v4;
    if (v260) {
        s0[v273] = v269;
    }
    float v274 = m2;
    float v275 = simd_sum(v274);
    uint v277 = v261 + 2u;
    uint v278 = v277 * 2u;
    uint v279 = v278 + v4;
    if (v260) {
        s0[v279] = v275;
    }
    float v280 = m3;
    float v281 = simd_sum(v280);
    uint v283 = v261 + 3u;
    uint v284 = v283 * 2u;
    uint v285 = v284 + v4;
    if (v260) {
        s0[v285] = v281;
    }
    float v286 = m4;
    float v287 = simd_sum(v286);
    uint v289 = v261 + 4u;
    uint v290 = v289 * 2u;
    uint v291 = v290 + v4;
    if (v260) {
        s0[v291] = v287;
    }
    float v292 = m5;
    float v293 = simd_sum(v292);
    uint v295 = v261 + 5u;
    uint v296 = v295 * 2u;
    uint v297 = v296 + v4;
    if (v260) {
        s0[v297] = v293;
    }
    float v298 = m6;
    float v299 = simd_sum(v298);
    uint v301 = v261 + 6u;
    uint v302 = v301 * 2u;
    uint v303 = v302 + v4;
    if (v260) {
        s0[v303] = v299;
    }
    float v304 = m7;
    float v305 = simd_sum(v304);
    uint v307 = v261 + 7u;
    uint v308 = v307 * 2u;
    uint v309 = v308 + v4;
    if (v260) {
        s0[v309] = v305;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v310 = v4 == 0u;
    bool v311 = v260 && v310;
    uint v313 = v10 + 0u;
    bool v314 = v313 < 5120u;
    bool v315 = v311 && v314;
    if (v315) {
        uint v317 = v261 + 0u;
        uint v318 = v317 * 2u;
        float v319 = s0[v318];
        uint v321 = v318 + 1u;
        float v322 = s0[v321];
        float v323 = v319 + v322;
        uint v324 = uint(as_type<ushort>(half(v323)));
        b2[v313] = ushort(v324);
    }
    uint v326 = v10 + 1u;
    bool v327 = v326 < 5120u;
    bool v328 = v311 && v327;
    if (v328) {
        uint v330 = v261 + 1u;
        uint v331 = v330 * 2u;
        float v332 = s0[v331];
        uint v334 = v331 + 1u;
        float v335 = s0[v334];
        float v336 = v332 + v335;
        uint v337 = uint(as_type<ushort>(half(v336)));
        b2[v326] = ushort(v337);
    }
    uint v339 = v10 + 2u;
    bool v340 = v339 < 5120u;
    bool v341 = v311 && v340;
    if (v341) {
        uint v343 = v261 + 2u;
        uint v344 = v343 * 2u;
        float v345 = s0[v344];
        uint v347 = v344 + 1u;
        float v348 = s0[v347];
        float v349 = v345 + v348;
        uint v350 = uint(as_type<ushort>(half(v349)));
        b2[v339] = ushort(v350);
    }
    uint v352 = v10 + 3u;
    bool v353 = v352 < 5120u;
    bool v354 = v311 && v353;
    if (v354) {
        uint v356 = v261 + 3u;
        uint v357 = v356 * 2u;
        float v358 = s0[v357];
        uint v360 = v357 + 1u;
        float v361 = s0[v360];
        float v362 = v358 + v361;
        uint v363 = uint(as_type<ushort>(half(v362)));
        b2[v352] = ushort(v363);
    }
    uint v365 = v10 + 4u;
    bool v366 = v365 < 5120u;
    bool v367 = v311 && v366;
    if (v367) {
        uint v369 = v261 + 4u;
        uint v370 = v369 * 2u;
        float v371 = s0[v370];
        uint v373 = v370 + 1u;
        float v374 = s0[v373];
        float v375 = v371 + v374;
        uint v376 = uint(as_type<ushort>(half(v375)));
        b2[v365] = ushort(v376);
    }
    uint v378 = v10 + 5u;
    bool v379 = v378 < 5120u;
    bool v380 = v311 && v379;
    if (v380) {
        uint v382 = v261 + 5u;
        uint v383 = v382 * 2u;
        float v384 = s0[v383];
        uint v386 = v383 + 1u;
        float v387 = s0[v386];
        float v388 = v384 + v387;
        uint v389 = uint(as_type<ushort>(half(v388)));
        b2[v378] = ushort(v389);
    }
    uint v391 = v10 + 6u;
    bool v392 = v391 < 5120u;
    bool v393 = v311 && v392;
    if (v393) {
        uint v395 = v261 + 6u;
        uint v396 = v395 * 2u;
        float v397 = s0[v396];
        uint v399 = v396 + 1u;
        float v400 = s0[v399];
        float v401 = v397 + v400;
        uint v402 = uint(as_type<ushort>(half(v401)));
        b2[v391] = ushort(v402);
    }
    uint v404 = v10 + 7u;
    bool v405 = v404 < 5120u;
    bool v406 = v311 && v405;
    if (v406) {
        uint v408 = v261 + 7u;
        uint v409 = v408 * 2u;
        float v410 = s0[v409];
        uint v412 = v409 + 1u;
        float v413 = s0[v412];
        float v414 = v410 + v413;
        uint v415 = uint(as_type<ushort>(half(v414)));
        b2[v404] = ushort(v415);
    }
}
