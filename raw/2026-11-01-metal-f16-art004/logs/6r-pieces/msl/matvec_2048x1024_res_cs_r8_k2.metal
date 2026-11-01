#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_cs_r8_k2(
    device uint4 *b0 [[buffer(0)]],
    device uint4 *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    device ushort *b4 [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[32];
    bool v2 = tgid.x < 64u;
    bool v3 = !v2;
    if (v3) {
        uint v4 = tgid.x - 64u;
        uint v7 = v4 * 128u;
        uint v8 = v7 + lid;
        bool v10 = v8 < 6144u;
        if (v10) {
            uint v12 = v8 * 4u;
            uint v13 = uint(b3[v8]);
            float v14 = float(as_type<half>(ushort(v13)));
            uint v16 = v12 + 1u;
            uint v17 = uint(b4[v16]);
            float v18 = float(as_type<half>(ushort(v17)));
            uint v20 = v12 + 2u;
            uint v21 = uint(b4[v20]);
            float v22 = float(as_type<half>(ushort(v21)));
            uint v24 = v12 + 3u;
            uint v25 = uint(b4[v24]);
            float v26 = float(as_type<half>(ushort(v25)));
            uint v28 = v12 + 0u;
            uint v29 = uint(as_type<ushort>(half(v18)));
            b4[v28] = ushort(v29);
            uint v31 = v12 + 1u;
            uint v32 = uint(as_type<ushort>(half(v22)));
            b4[v31] = ushort(v32);
            uint v34 = v12 + 2u;
            uint v35 = uint(as_type<ushort>(half(v26)));
            b4[v34] = ushort(v35);
            uint v37 = v12 + 3u;
            uint v38 = uint(as_type<ushort>(half(v14)));
            b4[v37] = ushort(v38);
        }
        return;
    }
    uint v41 = sgid / 2u;
    uint v42 = sgid % 2u;
    uint v44 = tgid.x * 2u;
    uint v45 = v44 + v41;
    uint v48 = v45 * 8u;
    uint v53 = v48 + 0u;
    bool v54 = v53 < 1024u;
    uint v55 = v54 ? v53 : 1023u;
    uint v56 = v55 * 256u;
    uint v58 = v48 + 1u;
    bool v59 = v58 < 1024u;
    uint v60 = v59 ? v58 : 1023u;
    uint v61 = v60 * 256u;
    uint v63 = v48 + 2u;
    bool v64 = v63 < 1024u;
    uint v65 = v64 ? v63 : 1023u;
    uint v66 = v65 * 256u;
    uint v68 = v48 + 3u;
    bool v69 = v68 < 1024u;
    uint v70 = v69 ? v68 : 1023u;
    uint v71 = v70 * 256u;
    uint v73 = v48 + 4u;
    bool v74 = v73 < 1024u;
    uint v75 = v74 ? v73 : 1023u;
    uint v76 = v75 * 256u;
    uint v78 = v48 + 5u;
    bool v79 = v78 < 1024u;
    uint v80 = v79 ? v78 : 1023u;
    uint v81 = v80 * 256u;
    uint v83 = v48 + 6u;
    bool v84 = v83 < 1024u;
    uint v85 = v84 ? v83 : 1023u;
    uint v86 = v85 * 256u;
    uint v88 = v48 + 7u;
    bool v89 = v88 < 1024u;
    uint v90 = v89 ? v88 : 1023u;
    uint v91 = v90 * 256u;
    uint v94 = v42 * 32u;
    uint v95 = v94 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    float m4 = as_type<float>(0x00000000u);
    float m5 = as_type<float>(0x00000000u);
    float m6 = as_type<float>(0x00000000u);
    float m7 = as_type<float>(0x00000000u);
    for (uint v98 = v95; v98 < 256u; v98 += 64u) {
        uint4 v100 = b1[v98];
        uint v101 = v100.x;
        float v102 = float(as_type<half>(ushort(v101)));
        float v103 = float(as_type<half>(ushort(v101 >> 16)));
        uint v104 = v100.y;
        float v105 = float(as_type<half>(ushort(v104)));
        float v106 = float(as_type<half>(ushort(v104 >> 16)));
        uint v107 = v100.z;
        float v108 = float(as_type<half>(ushort(v107)));
        float v109 = float(as_type<half>(ushort(v107 >> 16)));
        uint v110 = v100.w;
        float v111 = float(as_type<half>(ushort(v110)));
        float v112 = float(as_type<half>(ushort(v110 >> 16)));
        uint v113 = v56 + v98;
        uint4 v114 = b0[v113];
        float v115 = m0;
        uint v116 = v114.x;
        float v117 = float(as_type<half>(ushort(v116)));
        float v118 = fma(v117, v102, v115);
        float v119 = float(as_type<half>(ushort(v116 >> 16)));
        float v120 = fma(v119, v103, v118);
        uint v121 = v114.y;
        float v122 = float(as_type<half>(ushort(v121)));
        float v123 = fma(v122, v105, v120);
        float v124 = float(as_type<half>(ushort(v121 >> 16)));
        float v125 = fma(v124, v106, v123);
        uint v126 = v114.z;
        float v127 = float(as_type<half>(ushort(v126)));
        float v128 = fma(v127, v108, v125);
        float v129 = float(as_type<half>(ushort(v126 >> 16)));
        float v130 = fma(v129, v109, v128);
        uint v131 = v114.w;
        float v132 = float(as_type<half>(ushort(v131)));
        float v133 = fma(v132, v111, v130);
        float v134 = float(as_type<half>(ushort(v131 >> 16)));
        float v135 = fma(v134, v112, v133);
        m0 = v135;
        uint v136 = v61 + v98;
        uint4 v137 = b0[v136];
        float v138 = m1;
        uint v139 = v137.x;
        float v140 = float(as_type<half>(ushort(v139)));
        float v141 = fma(v140, v102, v138);
        float v142 = float(as_type<half>(ushort(v139 >> 16)));
        float v143 = fma(v142, v103, v141);
        uint v144 = v137.y;
        float v145 = float(as_type<half>(ushort(v144)));
        float v146 = fma(v145, v105, v143);
        float v147 = float(as_type<half>(ushort(v144 >> 16)));
        float v148 = fma(v147, v106, v146);
        uint v149 = v137.z;
        float v150 = float(as_type<half>(ushort(v149)));
        float v151 = fma(v150, v108, v148);
        float v152 = float(as_type<half>(ushort(v149 >> 16)));
        float v153 = fma(v152, v109, v151);
        uint v154 = v137.w;
        float v155 = float(as_type<half>(ushort(v154)));
        float v156 = fma(v155, v111, v153);
        float v157 = float(as_type<half>(ushort(v154 >> 16)));
        float v158 = fma(v157, v112, v156);
        m1 = v158;
        uint v159 = v66 + v98;
        uint4 v160 = b0[v159];
        float v161 = m2;
        uint v162 = v160.x;
        float v163 = float(as_type<half>(ushort(v162)));
        float v164 = fma(v163, v102, v161);
        float v165 = float(as_type<half>(ushort(v162 >> 16)));
        float v166 = fma(v165, v103, v164);
        uint v167 = v160.y;
        float v168 = float(as_type<half>(ushort(v167)));
        float v169 = fma(v168, v105, v166);
        float v170 = float(as_type<half>(ushort(v167 >> 16)));
        float v171 = fma(v170, v106, v169);
        uint v172 = v160.z;
        float v173 = float(as_type<half>(ushort(v172)));
        float v174 = fma(v173, v108, v171);
        float v175 = float(as_type<half>(ushort(v172 >> 16)));
        float v176 = fma(v175, v109, v174);
        uint v177 = v160.w;
        float v178 = float(as_type<half>(ushort(v177)));
        float v179 = fma(v178, v111, v176);
        float v180 = float(as_type<half>(ushort(v177 >> 16)));
        float v181 = fma(v180, v112, v179);
        m2 = v181;
        uint v182 = v71 + v98;
        uint4 v183 = b0[v182];
        float v184 = m3;
        uint v185 = v183.x;
        float v186 = float(as_type<half>(ushort(v185)));
        float v187 = fma(v186, v102, v184);
        float v188 = float(as_type<half>(ushort(v185 >> 16)));
        float v189 = fma(v188, v103, v187);
        uint v190 = v183.y;
        float v191 = float(as_type<half>(ushort(v190)));
        float v192 = fma(v191, v105, v189);
        float v193 = float(as_type<half>(ushort(v190 >> 16)));
        float v194 = fma(v193, v106, v192);
        uint v195 = v183.z;
        float v196 = float(as_type<half>(ushort(v195)));
        float v197 = fma(v196, v108, v194);
        float v198 = float(as_type<half>(ushort(v195 >> 16)));
        float v199 = fma(v198, v109, v197);
        uint v200 = v183.w;
        float v201 = float(as_type<half>(ushort(v200)));
        float v202 = fma(v201, v111, v199);
        float v203 = float(as_type<half>(ushort(v200 >> 16)));
        float v204 = fma(v203, v112, v202);
        m3 = v204;
        uint v205 = v76 + v98;
        uint4 v206 = b0[v205];
        float v207 = m4;
        uint v208 = v206.x;
        float v209 = float(as_type<half>(ushort(v208)));
        float v210 = fma(v209, v102, v207);
        float v211 = float(as_type<half>(ushort(v208 >> 16)));
        float v212 = fma(v211, v103, v210);
        uint v213 = v206.y;
        float v214 = float(as_type<half>(ushort(v213)));
        float v215 = fma(v214, v105, v212);
        float v216 = float(as_type<half>(ushort(v213 >> 16)));
        float v217 = fma(v216, v106, v215);
        uint v218 = v206.z;
        float v219 = float(as_type<half>(ushort(v218)));
        float v220 = fma(v219, v108, v217);
        float v221 = float(as_type<half>(ushort(v218 >> 16)));
        float v222 = fma(v221, v109, v220);
        uint v223 = v206.w;
        float v224 = float(as_type<half>(ushort(v223)));
        float v225 = fma(v224, v111, v222);
        float v226 = float(as_type<half>(ushort(v223 >> 16)));
        float v227 = fma(v226, v112, v225);
        m4 = v227;
        uint v228 = v81 + v98;
        uint4 v229 = b0[v228];
        float v230 = m5;
        uint v231 = v229.x;
        float v232 = float(as_type<half>(ushort(v231)));
        float v233 = fma(v232, v102, v230);
        float v234 = float(as_type<half>(ushort(v231 >> 16)));
        float v235 = fma(v234, v103, v233);
        uint v236 = v229.y;
        float v237 = float(as_type<half>(ushort(v236)));
        float v238 = fma(v237, v105, v235);
        float v239 = float(as_type<half>(ushort(v236 >> 16)));
        float v240 = fma(v239, v106, v238);
        uint v241 = v229.z;
        float v242 = float(as_type<half>(ushort(v241)));
        float v243 = fma(v242, v108, v240);
        float v244 = float(as_type<half>(ushort(v241 >> 16)));
        float v245 = fma(v244, v109, v243);
        uint v246 = v229.w;
        float v247 = float(as_type<half>(ushort(v246)));
        float v248 = fma(v247, v111, v245);
        float v249 = float(as_type<half>(ushort(v246 >> 16)));
        float v250 = fma(v249, v112, v248);
        m5 = v250;
        uint v251 = v86 + v98;
        uint4 v252 = b0[v251];
        float v253 = m6;
        uint v254 = v252.x;
        float v255 = float(as_type<half>(ushort(v254)));
        float v256 = fma(v255, v102, v253);
        float v257 = float(as_type<half>(ushort(v254 >> 16)));
        float v258 = fma(v257, v103, v256);
        uint v259 = v252.y;
        float v260 = float(as_type<half>(ushort(v259)));
        float v261 = fma(v260, v105, v258);
        float v262 = float(as_type<half>(ushort(v259 >> 16)));
        float v263 = fma(v262, v106, v261);
        uint v264 = v252.z;
        float v265 = float(as_type<half>(ushort(v264)));
        float v266 = fma(v265, v108, v263);
        float v267 = float(as_type<half>(ushort(v264 >> 16)));
        float v268 = fma(v267, v109, v266);
        uint v269 = v252.w;
        float v270 = float(as_type<half>(ushort(v269)));
        float v271 = fma(v270, v111, v268);
        float v272 = float(as_type<half>(ushort(v269 >> 16)));
        float v273 = fma(v272, v112, v271);
        m6 = v273;
        uint v274 = v91 + v98;
        uint4 v275 = b0[v274];
        float v276 = m7;
        uint v277 = v275.x;
        float v278 = float(as_type<half>(ushort(v277)));
        float v279 = fma(v278, v102, v276);
        float v280 = float(as_type<half>(ushort(v277 >> 16)));
        float v281 = fma(v280, v103, v279);
        uint v282 = v275.y;
        float v283 = float(as_type<half>(ushort(v282)));
        float v284 = fma(v283, v105, v281);
        float v285 = float(as_type<half>(ushort(v282 >> 16)));
        float v286 = fma(v285, v106, v284);
        uint v287 = v275.z;
        float v288 = float(as_type<half>(ushort(v287)));
        float v289 = fma(v288, v108, v286);
        float v290 = float(as_type<half>(ushort(v287 >> 16)));
        float v291 = fma(v290, v109, v289);
        uint v292 = v275.w;
        float v293 = float(as_type<half>(ushort(v292)));
        float v294 = fma(v293, v111, v291);
        float v295 = float(as_type<half>(ushort(v292 >> 16)));
        float v296 = fma(v295, v112, v294);
        m7 = v296;
    }
    bool v298 = lane == 0u;
    uint v299 = v41 * 8u;
    float v300 = m0;
    float v301 = simd_sum(v300);
    uint v303 = v299 + 0u;
    uint v304 = v303 * 2u;
    uint v305 = v304 + v42;
    if (v298) {
        s0[v305] = v301;
    }
    float v306 = m1;
    float v307 = simd_sum(v306);
    uint v309 = v299 + 1u;
    uint v310 = v309 * 2u;
    uint v311 = v310 + v42;
    if (v298) {
        s0[v311] = v307;
    }
    float v312 = m2;
    float v313 = simd_sum(v312);
    uint v315 = v299 + 2u;
    uint v316 = v315 * 2u;
    uint v317 = v316 + v42;
    if (v298) {
        s0[v317] = v313;
    }
    float v318 = m3;
    float v319 = simd_sum(v318);
    uint v321 = v299 + 3u;
    uint v322 = v321 * 2u;
    uint v323 = v322 + v42;
    if (v298) {
        s0[v323] = v319;
    }
    float v324 = m4;
    float v325 = simd_sum(v324);
    uint v327 = v299 + 4u;
    uint v328 = v327 * 2u;
    uint v329 = v328 + v42;
    if (v298) {
        s0[v329] = v325;
    }
    float v330 = m5;
    float v331 = simd_sum(v330);
    uint v333 = v299 + 5u;
    uint v334 = v333 * 2u;
    uint v335 = v334 + v42;
    if (v298) {
        s0[v335] = v331;
    }
    float v336 = m6;
    float v337 = simd_sum(v336);
    uint v339 = v299 + 6u;
    uint v340 = v339 * 2u;
    uint v341 = v340 + v42;
    if (v298) {
        s0[v341] = v337;
    }
    float v342 = m7;
    float v343 = simd_sum(v342);
    uint v345 = v299 + 7u;
    uint v346 = v345 * 2u;
    uint v347 = v346 + v42;
    if (v298) {
        s0[v347] = v343;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v348 = v42 == 0u;
    bool v349 = v298 && v348;
    uint v351 = v48 + 0u;
    bool v352 = v351 < 1024u;
    bool v353 = v349 && v352;
    if (v353) {
        uint v355 = v299 + 0u;
        uint v356 = v355 * 2u;
        float v357 = s0[v356];
        uint v359 = v356 + 1u;
        float v360 = s0[v359];
        float v361 = v357 + v360;
        float v362 = float(half(v361));
        uint v363 = uint(b2[v351]);
        float v364 = float(as_type<half>(ushort(v363)));
        float v365 = v364 + v362;
        uint v366 = uint(as_type<ushort>(half(v365)));
        b2[v351] = ushort(v366);
    }
    uint v368 = v48 + 1u;
    bool v369 = v368 < 1024u;
    bool v370 = v349 && v369;
    if (v370) {
        uint v372 = v299 + 1u;
        uint v373 = v372 * 2u;
        float v374 = s0[v373];
        uint v376 = v373 + 1u;
        float v377 = s0[v376];
        float v378 = v374 + v377;
        float v379 = float(half(v378));
        uint v380 = uint(b2[v368]);
        float v381 = float(as_type<half>(ushort(v380)));
        float v382 = v381 + v379;
        uint v383 = uint(as_type<ushort>(half(v382)));
        b2[v368] = ushort(v383);
    }
    uint v385 = v48 + 2u;
    bool v386 = v385 < 1024u;
    bool v387 = v349 && v386;
    if (v387) {
        uint v389 = v299 + 2u;
        uint v390 = v389 * 2u;
        float v391 = s0[v390];
        uint v393 = v390 + 1u;
        float v394 = s0[v393];
        float v395 = v391 + v394;
        float v396 = float(half(v395));
        uint v397 = uint(b2[v385]);
        float v398 = float(as_type<half>(ushort(v397)));
        float v399 = v398 + v396;
        uint v400 = uint(as_type<ushort>(half(v399)));
        b2[v385] = ushort(v400);
    }
    uint v402 = v48 + 3u;
    bool v403 = v402 < 1024u;
    bool v404 = v349 && v403;
    if (v404) {
        uint v406 = v299 + 3u;
        uint v407 = v406 * 2u;
        float v408 = s0[v407];
        uint v410 = v407 + 1u;
        float v411 = s0[v410];
        float v412 = v408 + v411;
        float v413 = float(half(v412));
        uint v414 = uint(b2[v402]);
        float v415 = float(as_type<half>(ushort(v414)));
        float v416 = v415 + v413;
        uint v417 = uint(as_type<ushort>(half(v416)));
        b2[v402] = ushort(v417);
    }
    uint v419 = v48 + 4u;
    bool v420 = v419 < 1024u;
    bool v421 = v349 && v420;
    if (v421) {
        uint v423 = v299 + 4u;
        uint v424 = v423 * 2u;
        float v425 = s0[v424];
        uint v427 = v424 + 1u;
        float v428 = s0[v427];
        float v429 = v425 + v428;
        float v430 = float(half(v429));
        uint v431 = uint(b2[v419]);
        float v432 = float(as_type<half>(ushort(v431)));
        float v433 = v432 + v430;
        uint v434 = uint(as_type<ushort>(half(v433)));
        b2[v419] = ushort(v434);
    }
    uint v436 = v48 + 5u;
    bool v437 = v436 < 1024u;
    bool v438 = v349 && v437;
    if (v438) {
        uint v440 = v299 + 5u;
        uint v441 = v440 * 2u;
        float v442 = s0[v441];
        uint v444 = v441 + 1u;
        float v445 = s0[v444];
        float v446 = v442 + v445;
        float v447 = float(half(v446));
        uint v448 = uint(b2[v436]);
        float v449 = float(as_type<half>(ushort(v448)));
        float v450 = v449 + v447;
        uint v451 = uint(as_type<ushort>(half(v450)));
        b2[v436] = ushort(v451);
    }
    uint v453 = v48 + 6u;
    bool v454 = v453 < 1024u;
    bool v455 = v349 && v454;
    if (v455) {
        uint v457 = v299 + 6u;
        uint v458 = v457 * 2u;
        float v459 = s0[v458];
        uint v461 = v458 + 1u;
        float v462 = s0[v461];
        float v463 = v459 + v462;
        float v464 = float(half(v463));
        uint v465 = uint(b2[v453]);
        float v466 = float(as_type<half>(ushort(v465)));
        float v467 = v466 + v464;
        uint v468 = uint(as_type<ushort>(half(v467)));
        b2[v453] = ushort(v468);
    }
    uint v470 = v48 + 7u;
    bool v471 = v470 < 1024u;
    bool v472 = v349 && v471;
    if (v472) {
        uint v474 = v299 + 7u;
        uint v475 = v474 * 2u;
        float v476 = s0[v475];
        uint v478 = v475 + 1u;
        float v479 = s0[v478];
        float v480 = v476 + v479;
        float v481 = float(half(v480));
        uint v482 = uint(b2[v470]);
        float v483 = float(as_type<half>(ushort(v482)));
        float v484 = v483 + v481;
        uint v485 = uint(as_type<ushort>(half(v484)));
        b2[v470] = ushort(v485);
    }
}
