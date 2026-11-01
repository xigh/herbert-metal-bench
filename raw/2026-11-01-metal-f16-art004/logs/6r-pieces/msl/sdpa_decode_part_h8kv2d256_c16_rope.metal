#include <metal_stdlib>
using namespace metal;

kernel void sdpa_decode_part_h8kv2d256_c16_rope(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device uint4 *b2 [[buffer(2)]],
    device uint4 *b3 [[buffer(3)]],
    device float *b4 [[buffer(4)]],
    device ushort *b5 [[buffer(5)]],
    device ushort *b6 [[buffer(6)]],
    device ushort *b7 [[buffer(7)]],
    device ushort *b8 [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[8];
    threadgroup float s1[8];
    threadgroup float s2[2048];
    threadgroup float s3[9];
    threadgroup float s4[256];
    threadgroup float s5[256];
    device ushort *b2_a1 = (device ushort *)b2;
    device ushort *b3_a1 = (device ushort *)b3;
    uint v2 = tgid.x / 16u;
    uint v3 = tgid.x % 16u;
    uint v8 = b0[1u];
    uint v10 = v8 + 1u;
    uint v12 = v10 + 15u;
    uint v13 = v12 / 16u;
    uint v14 = v3 * v13;
    uint v15 = v14 + v13;
    bool v16 = v10 < v15;
    uint v17 = v16 ? v10 : v15;
    uint v19 = v2 / 4u;
    uint v21 = v2 * 512u;
    bool v28 = lid < 64u;
    bool v29 = v8 < v14;
    bool v30 = !v29;
    bool v31 = v8 < v17;
    bool v32 = v30 && v31;
    uint v34 = v19 * 256u;
    uint v35 = v19 * 524288u;
    uint v36 = v8 * 256u;
    uint v37 = v35 + v36;
    uint v38 = v37 + lid;
    uint v39 = v21 + lid;
    uint v40 = uint(b1[v39]);
    float v41 = float(as_type<half>(ushort(v40)));
    float v42 = v41 * v41;
    float v43 = simd_sum(v42);
    bool v47 = lane == 0u;
    if (v47) {
        s3[sgid] = v43;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v49 = lid == 0u;
    if (v49) {
        float v51 = s3[0u];
        float m0 = v51;
        for (uint v54 = 1u; v54 < 8u; v54 += 1u) {
            float v55 = m0;
            float v56 = s3[v54];
            float v57 = v55 + v56;
            m0 = v57;
        }
        float v58 = m0;
        s3[8u] = v58;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v61 = s3[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v62 = v61 * as_type<float>(0x3b800000u);
    float v63 = v62 + as_type<float>(0x358637bdu);
    float v64 = sqrt(v63);
    float v65 = as_type<float>(0x3f800000u) / v64;
    uint v66 = uint(b5[lid]);
    float v67 = float(as_type<half>(ushort(v66)));
    float v68 = v67 + as_type<float>(0x3f800000u);
    float v69 = v41 * v65;
    float v70 = v69 * v68;
    float v71 = float(half(v70));
    s4[lid] = v71;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float m1 = v71;
    if (v28) {
        bool v72 = lid < 32u;
        uint v73 = lid - 32u;
        uint v74 = v72 ? lid : v73;
        uint v75 = v8 * 32u;
        uint v76 = v75 + v74;
        uint v77 = uint(b7[v76]);
        uint v78 = uint(b8[v76]);
        float v79 = float(as_type<half>(ushort(v77)));
        float v80 = float(as_type<half>(ushort(v78)));
        float v81 = s4[v74];
        uint v82 = v74 + 32u;
        float v83 = s4[v82];
        float v84 = v81 * v79;
        float v85 = v83 * v80;
        float v86 = v84 - v85;
        float v87 = v81 * v80;
        float v88 = v83 * v79;
        float v89 = v87 + v88;
        float v90 = v72 ? v86 : v89;
        m1 = v90;
    }
    float v91 = m1;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v92 = uint(as_type<ushort>(half(v91)));
    float v93 = float(as_type<half>(ushort(v92)));
    s5[lid] = v93;
    uint v95 = 4096u + v34;
    uint v96 = v95 + lid;
    uint v97 = uint(b1[v96]);
    float v98 = float(as_type<half>(ushort(v97)));
    float v99 = v98 * v98;
    float v100 = simd_sum(v99);
    bool v104 = lane == 0u;
    if (v104) {
        s3[sgid] = v100;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v106 = lid == 0u;
    if (v106) {
        float v108 = s3[0u];
        float m2 = v108;
        for (uint v111 = 1u; v111 < 8u; v111 += 1u) {
            float v112 = m2;
            float v113 = s3[v111];
            float v114 = v112 + v113;
            m2 = v114;
        }
        float v115 = m2;
        s3[8u] = v115;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v118 = s3[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v119 = v118 * as_type<float>(0x3b800000u);
    float v120 = v119 + as_type<float>(0x358637bdu);
    float v121 = sqrt(v120);
    float v122 = as_type<float>(0x3f800000u) / v121;
    uint v123 = uint(b6[lid]);
    float v124 = float(as_type<half>(ushort(v123)));
    float v125 = v124 + as_type<float>(0x3f800000u);
    float v126 = v98 * v122;
    float v127 = v126 * v125;
    float v128 = float(half(v127));
    s4[lid] = v128;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float m3 = v128;
    if (v28) {
        bool v129 = lid < 32u;
        uint v130 = lid - 32u;
        uint v131 = v129 ? lid : v130;
        uint v132 = v8 * 32u;
        uint v133 = v132 + v131;
        uint v134 = uint(b7[v133]);
        uint v135 = uint(b8[v133]);
        float v136 = float(as_type<half>(ushort(v134)));
        float v137 = float(as_type<half>(ushort(v135)));
        float v138 = s4[v131];
        uint v139 = v131 + 32u;
        float v140 = s4[v139];
        float v141 = v138 * v136;
        float v142 = v140 * v137;
        float v143 = v141 - v142;
        float v144 = v138 * v137;
        float v145 = v140 * v136;
        float v146 = v144 + v145;
        float v147 = v129 ? v143 : v146;
        m3 = v147;
    }
    float v148 = m3;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v149 = uint(as_type<ushort>(half(v148)));
    uint v151 = 4608u + v34;
    uint v152 = v151 + lid;
    uint v153 = uint(b1[v152]);
    if (v32) {
        b2_a1[v38] = ushort(v149);
        b3_a1[v38] = ushort(v153);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v155 = lane * 8u;
    uint v157 = v155 + 0u;
    float v158 = s5[v157];
    uint v160 = v155 + 1u;
    float v161 = s5[v160];
    uint v163 = v155 + 2u;
    float v164 = s5[v163];
    uint v166 = v155 + 3u;
    float v167 = s5[v166];
    uint v169 = v155 + 4u;
    float v170 = s5[v169];
    uint v172 = v155 + 5u;
    float v173 = s5[v172];
    uint v175 = v155 + 6u;
    float v176 = s5[v175];
    uint v178 = v155 + 7u;
    float v179 = s5[v178];
    uint v182 = v19 * 2048u;
    float m4 = as_type<float>(0xff7fffffu);
    float m5 = as_type<float>(0x00000000u);
    float m6 = as_type<float>(0x00000000u);
    float m7 = as_type<float>(0x00000000u);
    float m8 = as_type<float>(0x00000000u);
    float m9 = as_type<float>(0x00000000u);
    float m10 = as_type<float>(0x00000000u);
    float m11 = as_type<float>(0x00000000u);
    float m12 = as_type<float>(0x00000000u);
    float m13 = as_type<float>(0x00000000u);
    uint v186 = v14 + sgid;
    for (uint v188 = v186; v188 < v17; v188 += 8u) {
        uint v189 = v182 + v188;
        uint v190 = v189 * 32u;
        uint v191 = v190 + lane;
        uint v194 = v191 + 0u;
        uint4 v195 = b2[v194];
        uint4 v196 = b3[v194];
        uint v197 = v195.x;
        float v198 = float(as_type<half>(ushort(v197)));
        float v199 = fma(v158, v198, as_type<float>(0x00000000u));
        float v200 = float(as_type<half>(ushort(v197 >> 16)));
        float v201 = fma(v161, v200, v199);
        uint v202 = v196.x;
        float v203 = float(as_type<half>(ushort(v202)));
        float v204 = float(as_type<half>(ushort(v202 >> 16)));
        uint v205 = v195.y;
        float v206 = float(as_type<half>(ushort(v205)));
        float v207 = fma(v164, v206, v201);
        float v208 = float(as_type<half>(ushort(v205 >> 16)));
        float v209 = fma(v167, v208, v207);
        uint v210 = v196.y;
        float v211 = float(as_type<half>(ushort(v210)));
        float v212 = float(as_type<half>(ushort(v210 >> 16)));
        uint v213 = v195.z;
        float v214 = float(as_type<half>(ushort(v213)));
        float v215 = fma(v170, v214, v209);
        float v216 = float(as_type<half>(ushort(v213 >> 16)));
        float v217 = fma(v173, v216, v215);
        uint v218 = v196.z;
        float v219 = float(as_type<half>(ushort(v218)));
        float v220 = float(as_type<half>(ushort(v218 >> 16)));
        uint v221 = v195.w;
        float v222 = float(as_type<half>(ushort(v221)));
        float v223 = fma(v176, v222, v217);
        float v224 = float(as_type<half>(ushort(v221 >> 16)));
        float v225 = fma(v179, v224, v223);
        uint v226 = v196.w;
        float v227 = float(as_type<half>(ushort(v226)));
        float v228 = float(as_type<half>(ushort(v226 >> 16)));
        float v229 = simd_sum(v225);
        float v230 = v229 * as_type<float>(0x3d800000u);
        float v231 = m4;
        float v232 = fmax(v231, v230);
        float v233 = v231 - v232;
        float v234 = exp(v233);
        float v235 = v230 - v232;
        float v236 = exp(v235);
        float v237 = m5;
        float v238 = v237 * v234;
        float v239 = v238 + v236;
        m5 = v239;
        m4 = v232;
        float v240 = m6;
        float v241 = v240 * v234;
        float v242 = fma(v236, v203, v241);
        m6 = v242;
        float v243 = m7;
        float v244 = v243 * v234;
        float v245 = fma(v236, v204, v244);
        m7 = v245;
        float v246 = m8;
        float v247 = v246 * v234;
        float v248 = fma(v236, v211, v247);
        m8 = v248;
        float v249 = m9;
        float v250 = v249 * v234;
        float v251 = fma(v236, v212, v250);
        m9 = v251;
        float v252 = m10;
        float v253 = v252 * v234;
        float v254 = fma(v236, v219, v253);
        m10 = v254;
        float v255 = m11;
        float v256 = v255 * v234;
        float v257 = fma(v236, v220, v256);
        m11 = v257;
        float v258 = m12;
        float v259 = v258 * v234;
        float v260 = fma(v236, v227, v259);
        m12 = v260;
        float v261 = m13;
        float v262 = v261 * v234;
        float v263 = fma(v236, v228, v262);
        m13 = v263;
    }
    bool v265 = lane == 0u;
    float v266 = m4;
    float v267 = m5;
    if (v265) {
        s0[sgid] = v266;
        s1[sgid] = v267;
    }
    uint v269 = sgid * 256u;
    uint v271 = v155 + 0u;
    uint v272 = v269 + v271;
    float v273 = m6;
    s2[v272] = v273;
    uint v275 = v155 + 1u;
    uint v276 = v269 + v275;
    float v277 = m7;
    s2[v276] = v277;
    uint v279 = v155 + 2u;
    uint v280 = v269 + v279;
    float v281 = m8;
    s2[v280] = v281;
    uint v283 = v155 + 3u;
    uint v284 = v269 + v283;
    float v285 = m9;
    s2[v284] = v285;
    uint v287 = v155 + 4u;
    uint v288 = v269 + v287;
    float v289 = m10;
    s2[v288] = v289;
    uint v291 = v155 + 5u;
    uint v292 = v269 + v291;
    float v293 = m11;
    s2[v292] = v293;
    uint v295 = v155 + 6u;
    uint v296 = v269 + v295;
    float v297 = m12;
    s2[v296] = v297;
    uint v299 = v155 + 7u;
    uint v300 = v269 + v299;
    float v301 = m13;
    s2[v300] = v301;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v302 = s0[0u];
    float v304 = s0[1u];
    float v305 = fmax(v302, v304);
    float v307 = s0[2u];
    float v308 = fmax(v305, v307);
    float v310 = s0[3u];
    float v311 = fmax(v308, v310);
    float v313 = s0[4u];
    float v314 = fmax(v311, v313);
    float v316 = s0[5u];
    float v317 = fmax(v314, v316);
    float v319 = s0[6u];
    float v320 = fmax(v317, v319);
    float v322 = s0[7u];
    float v323 = fmax(v320, v322);
    uint v324 = v2 * 16u;
    uint v325 = v324 + v3;
    uint v327 = v325 * 258u;
    for (uint v329 = lid; v329 < 256u; v329 += 256u) {
        float v332 = s0[0u];
        float v333 = v332 - v323;
        float v334 = exp(v333);
        uint v336 = 0u + v329;
        float v337 = s2[v336];
        float v338 = fma(v337, v334, as_type<float>(0x00000000u));
        float v340 = s0[1u];
        float v341 = v340 - v323;
        float v342 = exp(v341);
        uint v344 = 256u + v329;
        float v345 = s2[v344];
        float v346 = fma(v345, v342, v338);
        float v348 = s0[2u];
        float v349 = v348 - v323;
        float v350 = exp(v349);
        uint v352 = 512u + v329;
        float v353 = s2[v352];
        float v354 = fma(v353, v350, v346);
        float v356 = s0[3u];
        float v357 = v356 - v323;
        float v358 = exp(v357);
        uint v360 = 768u + v329;
        float v361 = s2[v360];
        float v362 = fma(v361, v358, v354);
        float v364 = s0[4u];
        float v365 = v364 - v323;
        float v366 = exp(v365);
        uint v368 = 1024u + v329;
        float v369 = s2[v368];
        float v370 = fma(v369, v366, v362);
        float v372 = s0[5u];
        float v373 = v372 - v323;
        float v374 = exp(v373);
        uint v376 = 1280u + v329;
        float v377 = s2[v376];
        float v378 = fma(v377, v374, v370);
        float v380 = s0[6u];
        float v381 = v380 - v323;
        float v382 = exp(v381);
        uint v384 = 1536u + v329;
        float v385 = s2[v384];
        float v386 = fma(v385, v382, v378);
        float v388 = s0[7u];
        float v389 = v388 - v323;
        float v390 = exp(v389);
        uint v392 = 1792u + v329;
        float v393 = s2[v392];
        float v394 = fma(v393, v390, v386);
        uint v395 = v327 + v329;
        b4[v395] = v394;
    }
    bool v396 = lid == 0u;
    if (v396) {
        float v399 = s0[0u];
        float v400 = v399 - v323;
        float v401 = exp(v400);
        float v402 = s1[0u];
        float v403 = fma(v402, v401, as_type<float>(0x00000000u));
        float v405 = s0[1u];
        float v406 = v405 - v323;
        float v407 = exp(v406);
        float v408 = s1[1u];
        float v409 = fma(v408, v407, v403);
        float v411 = s0[2u];
        float v412 = v411 - v323;
        float v413 = exp(v412);
        float v414 = s1[2u];
        float v415 = fma(v414, v413, v409);
        float v417 = s0[3u];
        float v418 = v417 - v323;
        float v419 = exp(v418);
        float v420 = s1[3u];
        float v421 = fma(v420, v419, v415);
        float v423 = s0[4u];
        float v424 = v423 - v323;
        float v425 = exp(v424);
        float v426 = s1[4u];
        float v427 = fma(v426, v425, v421);
        float v429 = s0[5u];
        float v430 = v429 - v323;
        float v431 = exp(v430);
        float v432 = s1[5u];
        float v433 = fma(v432, v431, v427);
        float v435 = s0[6u];
        float v436 = v435 - v323;
        float v437 = exp(v436);
        float v438 = s1[6u];
        float v439 = fma(v438, v437, v433);
        float v441 = s0[7u];
        float v442 = v441 - v323;
        float v443 = exp(v442);
        float v444 = s1[7u];
        float v445 = fma(v444, v443, v439);
        uint v446 = v327 + 256u;
        b4[v446] = v323;
        uint v447 = v446 + 1u;
        b4[v447] = v445;
    }
}
