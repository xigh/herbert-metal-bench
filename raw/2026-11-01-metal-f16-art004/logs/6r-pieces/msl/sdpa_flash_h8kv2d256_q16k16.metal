#include <metal_stdlib>
using namespace metal;

kernel void sdpa_flash_h8kv2d256_q16k16(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device uint *b2 [[buffer(2)]],
    device uint *b3 [[buffer(3)]],
    device ushort *b4 [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[4096];
    threadgroup uint s1[2064];
    threadgroup float s2[256];
    threadgroup float s3[16];
    threadgroup float s4[16];
    threadgroup float s5[16];
    uint v1 = b0[3u];
    uint v3 = b0[1u];
    uint v6 = tgid.x % 8u;
    uint v7 = tgid.x / 8u;
    uint v9 = v7 * 16u;
    uint v14 = v6 * 512u;
    uint v16 = v6 / 4u;
    uint v21 = v16 * 262144u;
    uint v28 = v1 - v9;
    bool v29 = v28 < 16u;
    uint v30 = v29 ? v28 : 16u;
    uint v31 = v3 + v9;
    uint v32 = v31 + v30;
    for (uint v34 = lid; v34 < 4096u; v34 += 256u) {
        uint v35 = v34 / 256u;
        uint v36 = v34 % 256u;
        uint v37 = v9 + v35;
        bool v38 = v37 < v1;
        float m0 = as_type<float>(0x00000000u);
        if (v38) {
            uint v39 = v37 * 5120u;
            uint v40 = v39 + v14;
            uint v41 = v40 + v36;
            uint v42 = uint(b1[v41]);
            float v43 = float(as_type<half>(ushort(v42)));
            m0 = v43;
        }
        float v44 = m0;
        s0[v34] = v44;
    }
    bool v45 = lid < 16u;
    if (v45) {
        s3[lid] = as_type<float>(0xff7fffffu);
        s4[lid] = as_type<float>(0x00000000u);
    }
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    float m4 = as_type<float>(0x00000000u);
    float m5 = as_type<float>(0x00000000u);
    float m6 = as_type<float>(0x00000000u);
    float m7 = as_type<float>(0x00000000u);
    float m8 = as_type<float>(0x00000000u);
    float m9 = as_type<float>(0x00000000u);
    float m10 = as_type<float>(0x00000000u);
    float m11 = as_type<float>(0x00000000u);
    float m12 = as_type<float>(0x00000000u);
    float m13 = as_type<float>(0x00000000u);
    float m14 = as_type<float>(0x00000000u);
    float m15 = as_type<float>(0x00000000u);
    float m16 = as_type<float>(0x00000000u);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint v49 = 0u; v49 < v32; v49 += 16u) {
        for (uint v50 = lid; v50 < 2048u; v50 += 256u) {
            uint v51 = v50 / 128u;
            uint v52 = v50 % 128u;
            uint v53 = v49 + v51;
            bool v54 = 2047u < v53;
            uint v55 = v54 ? 2047u : v53;
            uint v56 = v55 * 128u;
            uint v57 = v21 + v56;
            uint v58 = v57 + v52;
            uint v59 = b2[v58];
            uint v60 = v51 * 129u;
            uint v61 = v60 + v52;
            s1[v61] = v59;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        uint v63 = lid + 0u;
        uint v64 = v63 / 16u;
        uint v65 = v63 % 16u;
        uint v66 = v64 * 256u;
        uint v67 = v65 * 129u;
        float m17 = as_type<float>(0x00000000u);
        for (uint v68 = 0u; v68 < 128u; v68 += 1u) {
            uint v69 = v67 + v68;
            uint v70 = s1[v69];
            uint v72 = v68 * 2u;
            uint v73 = v66 + v72;
            float v74 = s0[v73];
            uint v75 = v73 + 1u;
            float v76 = s0[v75];
            float v77 = float(as_type<half>(ushort(v70)));
            float v78 = float(as_type<half>(ushort(v70 >> 16)));
            float v79 = m17;
            float v80 = fma(v74, v77, v79);
            float v81 = fma(v76, v78, v80);
            m17 = v81;
        }
        float v82 = m17;
        float v83 = v82 * as_type<float>(0x3d800000u);
        uint v84 = v49 + v65;
        uint v85 = v31 + v64;
        bool v86 = v85 < v84;
        uint v87 = v9 + v64;
        bool v88 = v87 < v1;
        bool v89 = !v86;
        bool v90 = v89 && v88;
        float v91 = v90 ? v83 : as_type<float>(0xff7fffffu);
        s2[v63] = v91;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (v45) {
            uint v92 = lid * 16u;
            float v93 = s3[lid];
            float m18 = v93;
            for (uint v94 = 0u; v94 < 16u; v94 += 1u) {
                uint v95 = v92 + v94;
                float v96 = s2[v95];
                float v97 = m18;
                float v98 = fmax(v97, v96);
                m18 = v98;
            }
            float v99 = m18;
            float v100 = v93 - v99;
            float v101 = exp(v100);
            float m19 = as_type<float>(0x00000000u);
            for (uint v102 = 0u; v102 < 16u; v102 += 1u) {
                uint v103 = v92 + v102;
                float v104 = s2[v103];
                float v105 = v104 - v99;
                float v106 = exp(v105);
                s2[v103] = v106;
                float v107 = m19;
                float v108 = v107 + v106;
                m19 = v108;
            }
            float v109 = s4[lid];
            float v110 = m19;
            float v111 = fma(v109, v101, v110);
            s4[lid] = v111;
            s3[lid] = v99;
            s5[lid] = v101;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        for (uint v112 = lid; v112 < 2048u; v112 += 256u) {
            uint v113 = v112 / 128u;
            uint v114 = v112 % 128u;
            uint v115 = v49 + v113;
            bool v116 = 2047u < v115;
            uint v117 = v116 ? 2047u : v115;
            uint v118 = v117 * 128u;
            uint v119 = v21 + v118;
            uint v120 = v119 + v114;
            uint v121 = b3[v120];
            uint v122 = v113 * 129u;
            uint v123 = v122 + v114;
            s1[v123] = v121;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        uint v124 = lid >> 1u;
        uint v125 = lid & 1u;
        bool v126 = v125 == 1u;
        float v128 = s5[0u];
        float v129 = m1;
        float v130 = v129 * v128;
        float m20 = v130;
        for (uint v132 = 0u; v132 < 16u; v132 += 1u) {
            uint v133 = 0u + v132;
            float v134 = s2[v133];
            uint v135 = v132 * 129u;
            uint v136 = v135 + v124;
            uint v137 = s1[v136];
            float v138 = float(as_type<half>(ushort(v137)));
            float v139 = float(as_type<half>(ushort(v137 >> 16)));
            float v140 = v126 ? v139 : v138;
            float v141 = m20;
            float v142 = fma(v134, v140, v141);
            m20 = v142;
        }
        float v143 = m20;
        m1 = v143;
        float v145 = s5[1u];
        float v146 = m2;
        float v147 = v146 * v145;
        float m21 = v147;
        for (uint v149 = 0u; v149 < 16u; v149 += 1u) {
            uint v150 = 16u + v149;
            float v151 = s2[v150];
            uint v152 = v149 * 129u;
            uint v153 = v152 + v124;
            uint v154 = s1[v153];
            float v155 = float(as_type<half>(ushort(v154)));
            float v156 = float(as_type<half>(ushort(v154 >> 16)));
            float v157 = v126 ? v156 : v155;
            float v158 = m21;
            float v159 = fma(v151, v157, v158);
            m21 = v159;
        }
        float v160 = m21;
        m2 = v160;
        float v162 = s5[2u];
        float v163 = m3;
        float v164 = v163 * v162;
        float m22 = v164;
        for (uint v166 = 0u; v166 < 16u; v166 += 1u) {
            uint v167 = 32u + v166;
            float v168 = s2[v167];
            uint v169 = v166 * 129u;
            uint v170 = v169 + v124;
            uint v171 = s1[v170];
            float v172 = float(as_type<half>(ushort(v171)));
            float v173 = float(as_type<half>(ushort(v171 >> 16)));
            float v174 = v126 ? v173 : v172;
            float v175 = m22;
            float v176 = fma(v168, v174, v175);
            m22 = v176;
        }
        float v177 = m22;
        m3 = v177;
        float v179 = s5[3u];
        float v180 = m4;
        float v181 = v180 * v179;
        float m23 = v181;
        for (uint v183 = 0u; v183 < 16u; v183 += 1u) {
            uint v184 = 48u + v183;
            float v185 = s2[v184];
            uint v186 = v183 * 129u;
            uint v187 = v186 + v124;
            uint v188 = s1[v187];
            float v189 = float(as_type<half>(ushort(v188)));
            float v190 = float(as_type<half>(ushort(v188 >> 16)));
            float v191 = v126 ? v190 : v189;
            float v192 = m23;
            float v193 = fma(v185, v191, v192);
            m23 = v193;
        }
        float v194 = m23;
        m4 = v194;
        float v196 = s5[4u];
        float v197 = m5;
        float v198 = v197 * v196;
        float m24 = v198;
        for (uint v200 = 0u; v200 < 16u; v200 += 1u) {
            uint v201 = 64u + v200;
            float v202 = s2[v201];
            uint v203 = v200 * 129u;
            uint v204 = v203 + v124;
            uint v205 = s1[v204];
            float v206 = float(as_type<half>(ushort(v205)));
            float v207 = float(as_type<half>(ushort(v205 >> 16)));
            float v208 = v126 ? v207 : v206;
            float v209 = m24;
            float v210 = fma(v202, v208, v209);
            m24 = v210;
        }
        float v211 = m24;
        m5 = v211;
        float v213 = s5[5u];
        float v214 = m6;
        float v215 = v214 * v213;
        float m25 = v215;
        for (uint v217 = 0u; v217 < 16u; v217 += 1u) {
            uint v218 = 80u + v217;
            float v219 = s2[v218];
            uint v220 = v217 * 129u;
            uint v221 = v220 + v124;
            uint v222 = s1[v221];
            float v223 = float(as_type<half>(ushort(v222)));
            float v224 = float(as_type<half>(ushort(v222 >> 16)));
            float v225 = v126 ? v224 : v223;
            float v226 = m25;
            float v227 = fma(v219, v225, v226);
            m25 = v227;
        }
        float v228 = m25;
        m6 = v228;
        float v230 = s5[6u];
        float v231 = m7;
        float v232 = v231 * v230;
        float m26 = v232;
        for (uint v234 = 0u; v234 < 16u; v234 += 1u) {
            uint v235 = 96u + v234;
            float v236 = s2[v235];
            uint v237 = v234 * 129u;
            uint v238 = v237 + v124;
            uint v239 = s1[v238];
            float v240 = float(as_type<half>(ushort(v239)));
            float v241 = float(as_type<half>(ushort(v239 >> 16)));
            float v242 = v126 ? v241 : v240;
            float v243 = m26;
            float v244 = fma(v236, v242, v243);
            m26 = v244;
        }
        float v245 = m26;
        m7 = v245;
        float v247 = s5[7u];
        float v248 = m8;
        float v249 = v248 * v247;
        float m27 = v249;
        for (uint v251 = 0u; v251 < 16u; v251 += 1u) {
            uint v252 = 112u + v251;
            float v253 = s2[v252];
            uint v254 = v251 * 129u;
            uint v255 = v254 + v124;
            uint v256 = s1[v255];
            float v257 = float(as_type<half>(ushort(v256)));
            float v258 = float(as_type<half>(ushort(v256 >> 16)));
            float v259 = v126 ? v258 : v257;
            float v260 = m27;
            float v261 = fma(v253, v259, v260);
            m27 = v261;
        }
        float v262 = m27;
        m8 = v262;
        float v264 = s5[8u];
        float v265 = m9;
        float v266 = v265 * v264;
        float m28 = v266;
        for (uint v268 = 0u; v268 < 16u; v268 += 1u) {
            uint v269 = 128u + v268;
            float v270 = s2[v269];
            uint v271 = v268 * 129u;
            uint v272 = v271 + v124;
            uint v273 = s1[v272];
            float v274 = float(as_type<half>(ushort(v273)));
            float v275 = float(as_type<half>(ushort(v273 >> 16)));
            float v276 = v126 ? v275 : v274;
            float v277 = m28;
            float v278 = fma(v270, v276, v277);
            m28 = v278;
        }
        float v279 = m28;
        m9 = v279;
        float v281 = s5[9u];
        float v282 = m10;
        float v283 = v282 * v281;
        float m29 = v283;
        for (uint v285 = 0u; v285 < 16u; v285 += 1u) {
            uint v286 = 144u + v285;
            float v287 = s2[v286];
            uint v288 = v285 * 129u;
            uint v289 = v288 + v124;
            uint v290 = s1[v289];
            float v291 = float(as_type<half>(ushort(v290)));
            float v292 = float(as_type<half>(ushort(v290 >> 16)));
            float v293 = v126 ? v292 : v291;
            float v294 = m29;
            float v295 = fma(v287, v293, v294);
            m29 = v295;
        }
        float v296 = m29;
        m10 = v296;
        float v298 = s5[10u];
        float v299 = m11;
        float v300 = v299 * v298;
        float m30 = v300;
        for (uint v302 = 0u; v302 < 16u; v302 += 1u) {
            uint v303 = 160u + v302;
            float v304 = s2[v303];
            uint v305 = v302 * 129u;
            uint v306 = v305 + v124;
            uint v307 = s1[v306];
            float v308 = float(as_type<half>(ushort(v307)));
            float v309 = float(as_type<half>(ushort(v307 >> 16)));
            float v310 = v126 ? v309 : v308;
            float v311 = m30;
            float v312 = fma(v304, v310, v311);
            m30 = v312;
        }
        float v313 = m30;
        m11 = v313;
        float v315 = s5[11u];
        float v316 = m12;
        float v317 = v316 * v315;
        float m31 = v317;
        for (uint v319 = 0u; v319 < 16u; v319 += 1u) {
            uint v320 = 176u + v319;
            float v321 = s2[v320];
            uint v322 = v319 * 129u;
            uint v323 = v322 + v124;
            uint v324 = s1[v323];
            float v325 = float(as_type<half>(ushort(v324)));
            float v326 = float(as_type<half>(ushort(v324 >> 16)));
            float v327 = v126 ? v326 : v325;
            float v328 = m31;
            float v329 = fma(v321, v327, v328);
            m31 = v329;
        }
        float v330 = m31;
        m12 = v330;
        float v332 = s5[12u];
        float v333 = m13;
        float v334 = v333 * v332;
        float m32 = v334;
        for (uint v336 = 0u; v336 < 16u; v336 += 1u) {
            uint v337 = 192u + v336;
            float v338 = s2[v337];
            uint v339 = v336 * 129u;
            uint v340 = v339 + v124;
            uint v341 = s1[v340];
            float v342 = float(as_type<half>(ushort(v341)));
            float v343 = float(as_type<half>(ushort(v341 >> 16)));
            float v344 = v126 ? v343 : v342;
            float v345 = m32;
            float v346 = fma(v338, v344, v345);
            m32 = v346;
        }
        float v347 = m32;
        m13 = v347;
        float v349 = s5[13u];
        float v350 = m14;
        float v351 = v350 * v349;
        float m33 = v351;
        for (uint v353 = 0u; v353 < 16u; v353 += 1u) {
            uint v354 = 208u + v353;
            float v355 = s2[v354];
            uint v356 = v353 * 129u;
            uint v357 = v356 + v124;
            uint v358 = s1[v357];
            float v359 = float(as_type<half>(ushort(v358)));
            float v360 = float(as_type<half>(ushort(v358 >> 16)));
            float v361 = v126 ? v360 : v359;
            float v362 = m33;
            float v363 = fma(v355, v361, v362);
            m33 = v363;
        }
        float v364 = m33;
        m14 = v364;
        float v366 = s5[14u];
        float v367 = m15;
        float v368 = v367 * v366;
        float m34 = v368;
        for (uint v370 = 0u; v370 < 16u; v370 += 1u) {
            uint v371 = 224u + v370;
            float v372 = s2[v371];
            uint v373 = v370 * 129u;
            uint v374 = v373 + v124;
            uint v375 = s1[v374];
            float v376 = float(as_type<half>(ushort(v375)));
            float v377 = float(as_type<half>(ushort(v375 >> 16)));
            float v378 = v126 ? v377 : v376;
            float v379 = m34;
            float v380 = fma(v372, v378, v379);
            m34 = v380;
        }
        float v381 = m34;
        m15 = v381;
        float v383 = s5[15u];
        float v384 = m16;
        float v385 = v384 * v383;
        float m35 = v385;
        for (uint v387 = 0u; v387 < 16u; v387 += 1u) {
            uint v388 = 240u + v387;
            float v389 = s2[v388];
            uint v390 = v387 * 129u;
            uint v391 = v390 + v124;
            uint v392 = s1[v391];
            float v393 = float(as_type<half>(ushort(v392)));
            float v394 = float(as_type<half>(ushort(v392 >> 16)));
            float v395 = v126 ? v394 : v393;
            float v396 = m35;
            float v397 = fma(v389, v395, v396);
            m35 = v397;
        }
        float v398 = m35;
        m16 = v398;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    uint v401 = v9 + 0u;
    bool v402 = v401 < v1;
    float v403 = m1;
    if (v402) {
        float v404 = s4[0u];
        float v405 = v403 / v404;
        float v406 = float(half(v405));
        uint v407 = v401 * 5120u;
        uint v408 = v407 + v14;
        uint v409 = v408 + 256u;
        uint v410 = v409 + lid;
        uint v411 = uint(b1[v410]);
        float v412 = float(as_type<half>(ushort(v411)));
        float v413 = -v412;
        float v414 = exp(v413);
        float v416 = as_type<float>(0x3f800000u) + v414;
        float v417 = as_type<float>(0x3f800000u) / v416;
        float v418 = v406 * v417;
        uint v419 = uint(as_type<ushort>(half(v418)));
        uint v420 = v401 * 2048u;
        uint v421 = v6 * 256u;
        uint v422 = v420 + v421;
        uint v423 = v422 + lid;
        b4[v423] = ushort(v419);
    }
    uint v425 = v9 + 1u;
    bool v426 = v425 < v1;
    float v427 = m2;
    if (v426) {
        float v428 = s4[1u];
        float v429 = v427 / v428;
        float v430 = float(half(v429));
        uint v431 = v425 * 5120u;
        uint v432 = v431 + v14;
        uint v433 = v432 + 256u;
        uint v434 = v433 + lid;
        uint v435 = uint(b1[v434]);
        float v436 = float(as_type<half>(ushort(v435)));
        float v437 = -v436;
        float v438 = exp(v437);
        float v440 = as_type<float>(0x3f800000u) + v438;
        float v441 = as_type<float>(0x3f800000u) / v440;
        float v442 = v430 * v441;
        uint v443 = uint(as_type<ushort>(half(v442)));
        uint v444 = v425 * 2048u;
        uint v445 = v6 * 256u;
        uint v446 = v444 + v445;
        uint v447 = v446 + lid;
        b4[v447] = ushort(v443);
    }
    uint v449 = v9 + 2u;
    bool v450 = v449 < v1;
    float v451 = m3;
    if (v450) {
        float v452 = s4[2u];
        float v453 = v451 / v452;
        float v454 = float(half(v453));
        uint v455 = v449 * 5120u;
        uint v456 = v455 + v14;
        uint v457 = v456 + 256u;
        uint v458 = v457 + lid;
        uint v459 = uint(b1[v458]);
        float v460 = float(as_type<half>(ushort(v459)));
        float v461 = -v460;
        float v462 = exp(v461);
        float v464 = as_type<float>(0x3f800000u) + v462;
        float v465 = as_type<float>(0x3f800000u) / v464;
        float v466 = v454 * v465;
        uint v467 = uint(as_type<ushort>(half(v466)));
        uint v468 = v449 * 2048u;
        uint v469 = v6 * 256u;
        uint v470 = v468 + v469;
        uint v471 = v470 + lid;
        b4[v471] = ushort(v467);
    }
    uint v473 = v9 + 3u;
    bool v474 = v473 < v1;
    float v475 = m4;
    if (v474) {
        float v476 = s4[3u];
        float v477 = v475 / v476;
        float v478 = float(half(v477));
        uint v479 = v473 * 5120u;
        uint v480 = v479 + v14;
        uint v481 = v480 + 256u;
        uint v482 = v481 + lid;
        uint v483 = uint(b1[v482]);
        float v484 = float(as_type<half>(ushort(v483)));
        float v485 = -v484;
        float v486 = exp(v485);
        float v488 = as_type<float>(0x3f800000u) + v486;
        float v489 = as_type<float>(0x3f800000u) / v488;
        float v490 = v478 * v489;
        uint v491 = uint(as_type<ushort>(half(v490)));
        uint v492 = v473 * 2048u;
        uint v493 = v6 * 256u;
        uint v494 = v492 + v493;
        uint v495 = v494 + lid;
        b4[v495] = ushort(v491);
    }
    uint v497 = v9 + 4u;
    bool v498 = v497 < v1;
    float v499 = m5;
    if (v498) {
        float v500 = s4[4u];
        float v501 = v499 / v500;
        float v502 = float(half(v501));
        uint v503 = v497 * 5120u;
        uint v504 = v503 + v14;
        uint v505 = v504 + 256u;
        uint v506 = v505 + lid;
        uint v507 = uint(b1[v506]);
        float v508 = float(as_type<half>(ushort(v507)));
        float v509 = -v508;
        float v510 = exp(v509);
        float v512 = as_type<float>(0x3f800000u) + v510;
        float v513 = as_type<float>(0x3f800000u) / v512;
        float v514 = v502 * v513;
        uint v515 = uint(as_type<ushort>(half(v514)));
        uint v516 = v497 * 2048u;
        uint v517 = v6 * 256u;
        uint v518 = v516 + v517;
        uint v519 = v518 + lid;
        b4[v519] = ushort(v515);
    }
    uint v521 = v9 + 5u;
    bool v522 = v521 < v1;
    float v523 = m6;
    if (v522) {
        float v524 = s4[5u];
        float v525 = v523 / v524;
        float v526 = float(half(v525));
        uint v527 = v521 * 5120u;
        uint v528 = v527 + v14;
        uint v529 = v528 + 256u;
        uint v530 = v529 + lid;
        uint v531 = uint(b1[v530]);
        float v532 = float(as_type<half>(ushort(v531)));
        float v533 = -v532;
        float v534 = exp(v533);
        float v536 = as_type<float>(0x3f800000u) + v534;
        float v537 = as_type<float>(0x3f800000u) / v536;
        float v538 = v526 * v537;
        uint v539 = uint(as_type<ushort>(half(v538)));
        uint v540 = v521 * 2048u;
        uint v541 = v6 * 256u;
        uint v542 = v540 + v541;
        uint v543 = v542 + lid;
        b4[v543] = ushort(v539);
    }
    uint v545 = v9 + 6u;
    bool v546 = v545 < v1;
    float v547 = m7;
    if (v546) {
        float v548 = s4[6u];
        float v549 = v547 / v548;
        float v550 = float(half(v549));
        uint v551 = v545 * 5120u;
        uint v552 = v551 + v14;
        uint v553 = v552 + 256u;
        uint v554 = v553 + lid;
        uint v555 = uint(b1[v554]);
        float v556 = float(as_type<half>(ushort(v555)));
        float v557 = -v556;
        float v558 = exp(v557);
        float v560 = as_type<float>(0x3f800000u) + v558;
        float v561 = as_type<float>(0x3f800000u) / v560;
        float v562 = v550 * v561;
        uint v563 = uint(as_type<ushort>(half(v562)));
        uint v564 = v545 * 2048u;
        uint v565 = v6 * 256u;
        uint v566 = v564 + v565;
        uint v567 = v566 + lid;
        b4[v567] = ushort(v563);
    }
    uint v569 = v9 + 7u;
    bool v570 = v569 < v1;
    float v571 = m8;
    if (v570) {
        float v572 = s4[7u];
        float v573 = v571 / v572;
        float v574 = float(half(v573));
        uint v575 = v569 * 5120u;
        uint v576 = v575 + v14;
        uint v577 = v576 + 256u;
        uint v578 = v577 + lid;
        uint v579 = uint(b1[v578]);
        float v580 = float(as_type<half>(ushort(v579)));
        float v581 = -v580;
        float v582 = exp(v581);
        float v584 = as_type<float>(0x3f800000u) + v582;
        float v585 = as_type<float>(0x3f800000u) / v584;
        float v586 = v574 * v585;
        uint v587 = uint(as_type<ushort>(half(v586)));
        uint v588 = v569 * 2048u;
        uint v589 = v6 * 256u;
        uint v590 = v588 + v589;
        uint v591 = v590 + lid;
        b4[v591] = ushort(v587);
    }
    uint v593 = v9 + 8u;
    bool v594 = v593 < v1;
    float v595 = m9;
    if (v594) {
        float v596 = s4[8u];
        float v597 = v595 / v596;
        float v598 = float(half(v597));
        uint v599 = v593 * 5120u;
        uint v600 = v599 + v14;
        uint v601 = v600 + 256u;
        uint v602 = v601 + lid;
        uint v603 = uint(b1[v602]);
        float v604 = float(as_type<half>(ushort(v603)));
        float v605 = -v604;
        float v606 = exp(v605);
        float v608 = as_type<float>(0x3f800000u) + v606;
        float v609 = as_type<float>(0x3f800000u) / v608;
        float v610 = v598 * v609;
        uint v611 = uint(as_type<ushort>(half(v610)));
        uint v612 = v593 * 2048u;
        uint v613 = v6 * 256u;
        uint v614 = v612 + v613;
        uint v615 = v614 + lid;
        b4[v615] = ushort(v611);
    }
    uint v617 = v9 + 9u;
    bool v618 = v617 < v1;
    float v619 = m10;
    if (v618) {
        float v620 = s4[9u];
        float v621 = v619 / v620;
        float v622 = float(half(v621));
        uint v623 = v617 * 5120u;
        uint v624 = v623 + v14;
        uint v625 = v624 + 256u;
        uint v626 = v625 + lid;
        uint v627 = uint(b1[v626]);
        float v628 = float(as_type<half>(ushort(v627)));
        float v629 = -v628;
        float v630 = exp(v629);
        float v632 = as_type<float>(0x3f800000u) + v630;
        float v633 = as_type<float>(0x3f800000u) / v632;
        float v634 = v622 * v633;
        uint v635 = uint(as_type<ushort>(half(v634)));
        uint v636 = v617 * 2048u;
        uint v637 = v6 * 256u;
        uint v638 = v636 + v637;
        uint v639 = v638 + lid;
        b4[v639] = ushort(v635);
    }
    uint v641 = v9 + 10u;
    bool v642 = v641 < v1;
    float v643 = m11;
    if (v642) {
        float v644 = s4[10u];
        float v645 = v643 / v644;
        float v646 = float(half(v645));
        uint v647 = v641 * 5120u;
        uint v648 = v647 + v14;
        uint v649 = v648 + 256u;
        uint v650 = v649 + lid;
        uint v651 = uint(b1[v650]);
        float v652 = float(as_type<half>(ushort(v651)));
        float v653 = -v652;
        float v654 = exp(v653);
        float v656 = as_type<float>(0x3f800000u) + v654;
        float v657 = as_type<float>(0x3f800000u) / v656;
        float v658 = v646 * v657;
        uint v659 = uint(as_type<ushort>(half(v658)));
        uint v660 = v641 * 2048u;
        uint v661 = v6 * 256u;
        uint v662 = v660 + v661;
        uint v663 = v662 + lid;
        b4[v663] = ushort(v659);
    }
    uint v665 = v9 + 11u;
    bool v666 = v665 < v1;
    float v667 = m12;
    if (v666) {
        float v668 = s4[11u];
        float v669 = v667 / v668;
        float v670 = float(half(v669));
        uint v671 = v665 * 5120u;
        uint v672 = v671 + v14;
        uint v673 = v672 + 256u;
        uint v674 = v673 + lid;
        uint v675 = uint(b1[v674]);
        float v676 = float(as_type<half>(ushort(v675)));
        float v677 = -v676;
        float v678 = exp(v677);
        float v680 = as_type<float>(0x3f800000u) + v678;
        float v681 = as_type<float>(0x3f800000u) / v680;
        float v682 = v670 * v681;
        uint v683 = uint(as_type<ushort>(half(v682)));
        uint v684 = v665 * 2048u;
        uint v685 = v6 * 256u;
        uint v686 = v684 + v685;
        uint v687 = v686 + lid;
        b4[v687] = ushort(v683);
    }
    uint v689 = v9 + 12u;
    bool v690 = v689 < v1;
    float v691 = m13;
    if (v690) {
        float v692 = s4[12u];
        float v693 = v691 / v692;
        float v694 = float(half(v693));
        uint v695 = v689 * 5120u;
        uint v696 = v695 + v14;
        uint v697 = v696 + 256u;
        uint v698 = v697 + lid;
        uint v699 = uint(b1[v698]);
        float v700 = float(as_type<half>(ushort(v699)));
        float v701 = -v700;
        float v702 = exp(v701);
        float v704 = as_type<float>(0x3f800000u) + v702;
        float v705 = as_type<float>(0x3f800000u) / v704;
        float v706 = v694 * v705;
        uint v707 = uint(as_type<ushort>(half(v706)));
        uint v708 = v689 * 2048u;
        uint v709 = v6 * 256u;
        uint v710 = v708 + v709;
        uint v711 = v710 + lid;
        b4[v711] = ushort(v707);
    }
    uint v713 = v9 + 13u;
    bool v714 = v713 < v1;
    float v715 = m14;
    if (v714) {
        float v716 = s4[13u];
        float v717 = v715 / v716;
        float v718 = float(half(v717));
        uint v719 = v713 * 5120u;
        uint v720 = v719 + v14;
        uint v721 = v720 + 256u;
        uint v722 = v721 + lid;
        uint v723 = uint(b1[v722]);
        float v724 = float(as_type<half>(ushort(v723)));
        float v725 = -v724;
        float v726 = exp(v725);
        float v728 = as_type<float>(0x3f800000u) + v726;
        float v729 = as_type<float>(0x3f800000u) / v728;
        float v730 = v718 * v729;
        uint v731 = uint(as_type<ushort>(half(v730)));
        uint v732 = v713 * 2048u;
        uint v733 = v6 * 256u;
        uint v734 = v732 + v733;
        uint v735 = v734 + lid;
        b4[v735] = ushort(v731);
    }
    uint v737 = v9 + 14u;
    bool v738 = v737 < v1;
    float v739 = m15;
    if (v738) {
        float v740 = s4[14u];
        float v741 = v739 / v740;
        float v742 = float(half(v741));
        uint v743 = v737 * 5120u;
        uint v744 = v743 + v14;
        uint v745 = v744 + 256u;
        uint v746 = v745 + lid;
        uint v747 = uint(b1[v746]);
        float v748 = float(as_type<half>(ushort(v747)));
        float v749 = -v748;
        float v750 = exp(v749);
        float v752 = as_type<float>(0x3f800000u) + v750;
        float v753 = as_type<float>(0x3f800000u) / v752;
        float v754 = v742 * v753;
        uint v755 = uint(as_type<ushort>(half(v754)));
        uint v756 = v737 * 2048u;
        uint v757 = v6 * 256u;
        uint v758 = v756 + v757;
        uint v759 = v758 + lid;
        b4[v759] = ushort(v755);
    }
    uint v761 = v9 + 15u;
    bool v762 = v761 < v1;
    float v763 = m16;
    if (v762) {
        float v764 = s4[15u];
        float v765 = v763 / v764;
        float v766 = float(half(v765));
        uint v767 = v761 * 5120u;
        uint v768 = v767 + v14;
        uint v769 = v768 + 256u;
        uint v770 = v769 + lid;
        uint v771 = uint(b1[v770]);
        float v772 = float(as_type<half>(ushort(v771)));
        float v773 = -v772;
        float v774 = exp(v773);
        float v776 = as_type<float>(0x3f800000u) + v774;
        float v777 = as_type<float>(0x3f800000u) / v776;
        float v778 = v766 * v777;
        uint v779 = uint(as_type<ushort>(half(v778)));
        uint v780 = v761 * 2048u;
        uint v781 = v6 * 256u;
        uint v782 = v780 + v781;
        uint v783 = v782 + lid;
        b4[v783] = ushort(v779);
    }
}
