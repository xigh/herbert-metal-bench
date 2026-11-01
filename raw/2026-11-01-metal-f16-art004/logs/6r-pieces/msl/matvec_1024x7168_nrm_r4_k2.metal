#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x7168_nrm_r4_k2(
    device uint4 *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[16];
    threadgroup ushort s1[1024];
    threadgroup uint4 *s2 = (threadgroup uint4 *)s1;
    threadgroup float s3[9];
    uint v5 = lid + 0u;
    uint v6 = uint(b1[v5]);
    uint v8 = lid + 256u;
    uint v9 = uint(b1[v8]);
    uint v11 = lid + 512u;
    uint v12 = uint(b1[v11]);
    uint v14 = lid + 768u;
    uint v15 = uint(b1[v14]);
    float v17 = float(as_type<half>(ushort(v6)));
    float v18 = fma(v17, v17, as_type<float>(0x00000000u));
    float v19 = float(as_type<half>(ushort(v9)));
    float v20 = fma(v19, v19, v18);
    float v21 = float(as_type<half>(ushort(v12)));
    float v22 = fma(v21, v21, v20);
    float v23 = float(as_type<half>(ushort(v15)));
    float v24 = fma(v23, v23, v22);
    uint v26 = lid + 128u;
    uint v27 = uint(b1[v26]);
    uint v29 = lid + 384u;
    uint v30 = uint(b1[v29]);
    uint v32 = lid + 640u;
    uint v33 = uint(b1[v32]);
    uint v35 = lid + 896u;
    uint v36 = uint(b1[v35]);
    float v38 = float(as_type<half>(ushort(v27)));
    float v39 = fma(v38, v38, as_type<float>(0x00000000u));
    float v40 = float(as_type<half>(ushort(v30)));
    float v41 = fma(v40, v40, v39);
    float v42 = float(as_type<half>(ushort(v33)));
    float v43 = fma(v42, v42, v41);
    float v44 = float(as_type<half>(ushort(v36)));
    float v45 = fma(v44, v44, v43);
    bool v47 = lane == 0u;
    float v48 = simd_sum(v24);
    uint v50 = sgid + 0u;
    if (v47) {
        s3[v50] = v48;
    }
    float v51 = simd_sum(v45);
    uint v53 = sgid + 4u;
    if (v47) {
        s3[v53] = v51;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v54 = lid == 0u;
    if (v54) {
        float v56 = s3[0u];
        float v58 = s3[1u];
        float v59 = v56 + v58;
        float v61 = s3[2u];
        float v62 = v59 + v61;
        float v64 = s3[3u];
        float v65 = v62 + v64;
        float v67 = s3[4u];
        float v68 = v65 + v67;
        float v70 = s3[5u];
        float v71 = v68 + v70;
        float v73 = s3[6u];
        float v74 = v71 + v73;
        float v76 = s3[7u];
        float v77 = v74 + v76;
        s3[8u] = v77;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v80 = s3[8u];
    float v83 = v80 * as_type<float>(0x3a800000u);
    float v84 = v83 + as_type<float>(0x358637bdu);
    float v85 = sqrt(v84);
    float v87 = as_type<float>(0x3f800000u) / v85;
    uint v88 = uint(b3[v5]);
    uint v89 = uint(b3[v8]);
    uint v90 = uint(b3[v11]);
    uint v91 = uint(b3[v14]);
    uint v92 = uint(b3[v26]);
    uint v93 = uint(b3[v29]);
    uint v94 = uint(b3[v32]);
    uint v95 = uint(b3[v35]);
    float v96 = float(as_type<half>(ushort(v88)));
    float v98 = v96 + as_type<float>(0x3f800000u);
    float v99 = v17 * v87;
    float v100 = v99 * v98;
    uint v101 = uint(as_type<ushort>(half(v100)));
    s1[v5] = ushort(v101);
    float v102 = float(as_type<half>(ushort(v89)));
    float v104 = v102 + as_type<float>(0x3f800000u);
    float v105 = v19 * v87;
    float v106 = v105 * v104;
    uint v107 = uint(as_type<ushort>(half(v106)));
    s1[v8] = ushort(v107);
    float v108 = float(as_type<half>(ushort(v90)));
    float v110 = v108 + as_type<float>(0x3f800000u);
    float v111 = v21 * v87;
    float v112 = v111 * v110;
    uint v113 = uint(as_type<ushort>(half(v112)));
    s1[v11] = ushort(v113);
    float v114 = float(as_type<half>(ushort(v91)));
    float v116 = v114 + as_type<float>(0x3f800000u);
    float v117 = v23 * v87;
    float v118 = v117 * v116;
    uint v119 = uint(as_type<ushort>(half(v118)));
    s1[v14] = ushort(v119);
    float v120 = float(as_type<half>(ushort(v92)));
    float v122 = v120 + as_type<float>(0x3f800000u);
    float v123 = v38 * v87;
    float v124 = v123 * v122;
    uint v125 = uint(as_type<ushort>(half(v124)));
    s1[v26] = ushort(v125);
    float v126 = float(as_type<half>(ushort(v93)));
    float v128 = v126 + as_type<float>(0x3f800000u);
    float v129 = v40 * v87;
    float v130 = v129 * v128;
    uint v131 = uint(as_type<ushort>(half(v130)));
    s1[v29] = ushort(v131);
    float v132 = float(as_type<half>(ushort(v94)));
    float v134 = v132 + as_type<float>(0x3f800000u);
    float v135 = v42 * v87;
    float v136 = v135 * v134;
    uint v137 = uint(as_type<ushort>(half(v136)));
    s1[v32] = ushort(v137);
    float v138 = float(as_type<half>(ushort(v95)));
    float v140 = v138 + as_type<float>(0x3f800000u);
    float v141 = v44 * v87;
    float v142 = v141 * v140;
    uint v143 = uint(as_type<ushort>(half(v142)));
    s1[v35] = ushort(v143);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v146 = sgid / 2u;
    uint v147 = sgid % 2u;
    uint v149 = tgid.x * 2u;
    uint v150 = v149 + v146;
    uint v153 = v150 * 4u;
    uint v158 = v153 + 0u;
    bool v159 = v158 < 7168u;
    uint v160 = v159 ? v158 : 7167u;
    uint v161 = v160 * 128u;
    uint v163 = v153 + 1u;
    bool v164 = v163 < 7168u;
    uint v165 = v164 ? v163 : 7167u;
    uint v166 = v165 * 128u;
    uint v168 = v153 + 2u;
    bool v169 = v168 < 7168u;
    uint v170 = v169 ? v168 : 7167u;
    uint v171 = v170 * 128u;
    uint v173 = v153 + 3u;
    bool v174 = v173 < 7168u;
    uint v175 = v174 ? v173 : 7167u;
    uint v176 = v175 * 128u;
    uint v179 = v147 * 32u;
    uint v180 = v179 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    for (uint v183 = v180; v183 < 128u; v183 += 64u) {
        uint4 v185 = s2[v183];
        uint v186 = v185.x;
        float v187 = float(as_type<half>(ushort(v186)));
        float v188 = float(as_type<half>(ushort(v186 >> 16)));
        uint v189 = v185.y;
        float v190 = float(as_type<half>(ushort(v189)));
        float v191 = float(as_type<half>(ushort(v189 >> 16)));
        uint v192 = v185.z;
        float v193 = float(as_type<half>(ushort(v192)));
        float v194 = float(as_type<half>(ushort(v192 >> 16)));
        uint v195 = v185.w;
        float v196 = float(as_type<half>(ushort(v195)));
        float v197 = float(as_type<half>(ushort(v195 >> 16)));
        uint v198 = v161 + v183;
        uint4 v199 = b0[v198];
        float v200 = m0;
        uint v201 = v199.x;
        float v202 = float(as_type<half>(ushort(v201)));
        float v203 = fma(v202, v187, v200);
        float v204 = float(as_type<half>(ushort(v201 >> 16)));
        float v205 = fma(v204, v188, v203);
        uint v206 = v199.y;
        float v207 = float(as_type<half>(ushort(v206)));
        float v208 = fma(v207, v190, v205);
        float v209 = float(as_type<half>(ushort(v206 >> 16)));
        float v210 = fma(v209, v191, v208);
        uint v211 = v199.z;
        float v212 = float(as_type<half>(ushort(v211)));
        float v213 = fma(v212, v193, v210);
        float v214 = float(as_type<half>(ushort(v211 >> 16)));
        float v215 = fma(v214, v194, v213);
        uint v216 = v199.w;
        float v217 = float(as_type<half>(ushort(v216)));
        float v218 = fma(v217, v196, v215);
        float v219 = float(as_type<half>(ushort(v216 >> 16)));
        float v220 = fma(v219, v197, v218);
        m0 = v220;
        uint v221 = v166 + v183;
        uint4 v222 = b0[v221];
        float v223 = m1;
        uint v224 = v222.x;
        float v225 = float(as_type<half>(ushort(v224)));
        float v226 = fma(v225, v187, v223);
        float v227 = float(as_type<half>(ushort(v224 >> 16)));
        float v228 = fma(v227, v188, v226);
        uint v229 = v222.y;
        float v230 = float(as_type<half>(ushort(v229)));
        float v231 = fma(v230, v190, v228);
        float v232 = float(as_type<half>(ushort(v229 >> 16)));
        float v233 = fma(v232, v191, v231);
        uint v234 = v222.z;
        float v235 = float(as_type<half>(ushort(v234)));
        float v236 = fma(v235, v193, v233);
        float v237 = float(as_type<half>(ushort(v234 >> 16)));
        float v238 = fma(v237, v194, v236);
        uint v239 = v222.w;
        float v240 = float(as_type<half>(ushort(v239)));
        float v241 = fma(v240, v196, v238);
        float v242 = float(as_type<half>(ushort(v239 >> 16)));
        float v243 = fma(v242, v197, v241);
        m1 = v243;
        uint v244 = v171 + v183;
        uint4 v245 = b0[v244];
        float v246 = m2;
        uint v247 = v245.x;
        float v248 = float(as_type<half>(ushort(v247)));
        float v249 = fma(v248, v187, v246);
        float v250 = float(as_type<half>(ushort(v247 >> 16)));
        float v251 = fma(v250, v188, v249);
        uint v252 = v245.y;
        float v253 = float(as_type<half>(ushort(v252)));
        float v254 = fma(v253, v190, v251);
        float v255 = float(as_type<half>(ushort(v252 >> 16)));
        float v256 = fma(v255, v191, v254);
        uint v257 = v245.z;
        float v258 = float(as_type<half>(ushort(v257)));
        float v259 = fma(v258, v193, v256);
        float v260 = float(as_type<half>(ushort(v257 >> 16)));
        float v261 = fma(v260, v194, v259);
        uint v262 = v245.w;
        float v263 = float(as_type<half>(ushort(v262)));
        float v264 = fma(v263, v196, v261);
        float v265 = float(as_type<half>(ushort(v262 >> 16)));
        float v266 = fma(v265, v197, v264);
        m2 = v266;
        uint v267 = v176 + v183;
        uint4 v268 = b0[v267];
        float v269 = m3;
        uint v270 = v268.x;
        float v271 = float(as_type<half>(ushort(v270)));
        float v272 = fma(v271, v187, v269);
        float v273 = float(as_type<half>(ushort(v270 >> 16)));
        float v274 = fma(v273, v188, v272);
        uint v275 = v268.y;
        float v276 = float(as_type<half>(ushort(v275)));
        float v277 = fma(v276, v190, v274);
        float v278 = float(as_type<half>(ushort(v275 >> 16)));
        float v279 = fma(v278, v191, v277);
        uint v280 = v268.z;
        float v281 = float(as_type<half>(ushort(v280)));
        float v282 = fma(v281, v193, v279);
        float v283 = float(as_type<half>(ushort(v280 >> 16)));
        float v284 = fma(v283, v194, v282);
        uint v285 = v268.w;
        float v286 = float(as_type<half>(ushort(v285)));
        float v287 = fma(v286, v196, v284);
        float v288 = float(as_type<half>(ushort(v285 >> 16)));
        float v289 = fma(v288, v197, v287);
        m3 = v289;
    }
    bool v291 = lane == 0u;
    uint v292 = v146 * 4u;
    float v293 = m0;
    float v294 = simd_sum(v293);
    uint v296 = v292 + 0u;
    uint v297 = v296 * 2u;
    uint v298 = v297 + v147;
    if (v291) {
        s0[v298] = v294;
    }
    float v299 = m1;
    float v300 = simd_sum(v299);
    uint v302 = v292 + 1u;
    uint v303 = v302 * 2u;
    uint v304 = v303 + v147;
    if (v291) {
        s0[v304] = v300;
    }
    float v305 = m2;
    float v306 = simd_sum(v305);
    uint v308 = v292 + 2u;
    uint v309 = v308 * 2u;
    uint v310 = v309 + v147;
    if (v291) {
        s0[v310] = v306;
    }
    float v311 = m3;
    float v312 = simd_sum(v311);
    uint v314 = v292 + 3u;
    uint v315 = v314 * 2u;
    uint v316 = v315 + v147;
    if (v291) {
        s0[v316] = v312;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v317 = v147 == 0u;
    bool v318 = v291 && v317;
    uint v320 = v153 + 0u;
    bool v321 = v320 < 7168u;
    bool v322 = v318 && v321;
    if (v322) {
        uint v324 = v292 + 0u;
        uint v325 = v324 * 2u;
        float v326 = s0[v325];
        uint v328 = v325 + 1u;
        float v329 = s0[v328];
        float v330 = v326 + v329;
        uint v331 = uint(as_type<ushort>(half(v330)));
        b2[v320] = ushort(v331);
    }
    uint v333 = v153 + 1u;
    bool v334 = v333 < 7168u;
    bool v335 = v318 && v334;
    if (v335) {
        uint v337 = v292 + 1u;
        uint v338 = v337 * 2u;
        float v339 = s0[v338];
        uint v341 = v338 + 1u;
        float v342 = s0[v341];
        float v343 = v339 + v342;
        uint v344 = uint(as_type<ushort>(half(v343)));
        b2[v333] = ushort(v344);
    }
    uint v346 = v153 + 2u;
    bool v347 = v346 < 7168u;
    bool v348 = v318 && v347;
    if (v348) {
        uint v350 = v292 + 2u;
        uint v351 = v350 * 2u;
        float v352 = s0[v351];
        uint v354 = v351 + 1u;
        float v355 = s0[v354];
        float v356 = v352 + v355;
        uint v357 = uint(as_type<ushort>(half(v356)));
        b2[v346] = ushort(v357);
    }
    uint v359 = v153 + 3u;
    bool v360 = v359 < 7168u;
    bool v361 = v318 && v360;
    if (v361) {
        uint v363 = v292 + 3u;
        uint v364 = v363 * 2u;
        float v365 = s0[v364];
        uint v367 = v364 + 1u;
        float v368 = s0[v367];
        float v369 = v365 + v368;
        uint v370 = uint(as_type<ushort>(half(v369)));
        b2[v359] = ushort(v370);
    }
}
