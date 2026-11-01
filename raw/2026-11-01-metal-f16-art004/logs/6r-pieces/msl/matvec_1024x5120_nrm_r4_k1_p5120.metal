#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x5120_nrm_r4_k1_p5120(
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
    uint v146 = sgid / 1u;
    uint v147 = sgid % 1u;
    for (uint v150 = tgid.x; v150 < 320u; v150 += 5120u) {
        uint v152 = v150 * 4u;
        uint v153 = v152 + v146;
        uint v156 = v153 * 4u;
        uint v161 = v156 + 0u;
        bool v162 = v161 < 5120u;
        uint v163 = v162 ? v161 : 5119u;
        uint v164 = v163 * 128u;
        uint v166 = v156 + 1u;
        bool v167 = v166 < 5120u;
        uint v168 = v167 ? v166 : 5119u;
        uint v169 = v168 * 128u;
        uint v171 = v156 + 2u;
        bool v172 = v171 < 5120u;
        uint v173 = v172 ? v171 : 5119u;
        uint v174 = v173 * 128u;
        uint v176 = v156 + 3u;
        bool v177 = v176 < 5120u;
        uint v178 = v177 ? v176 : 5119u;
        uint v179 = v178 * 128u;
        uint v182 = v147 * 32u;
        uint v183 = v182 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        float m2 = as_type<float>(0x00000000u);
        float m3 = as_type<float>(0x00000000u);
        for (uint v186 = v183; v186 < 128u; v186 += 32u) {
            uint4 v188 = s2[v186];
            uint v189 = v188.x;
            float v190 = float(as_type<half>(ushort(v189)));
            float v191 = float(as_type<half>(ushort(v189 >> 16)));
            uint v192 = v188.y;
            float v193 = float(as_type<half>(ushort(v192)));
            float v194 = float(as_type<half>(ushort(v192 >> 16)));
            uint v195 = v188.z;
            float v196 = float(as_type<half>(ushort(v195)));
            float v197 = float(as_type<half>(ushort(v195 >> 16)));
            uint v198 = v188.w;
            float v199 = float(as_type<half>(ushort(v198)));
            float v200 = float(as_type<half>(ushort(v198 >> 16)));
            uint v201 = v164 + v186;
            uint4 v202 = b0[v201];
            float v203 = m0;
            uint v204 = v202.x;
            float v205 = float(as_type<half>(ushort(v204)));
            float v206 = fma(v205, v190, v203);
            float v207 = float(as_type<half>(ushort(v204 >> 16)));
            float v208 = fma(v207, v191, v206);
            uint v209 = v202.y;
            float v210 = float(as_type<half>(ushort(v209)));
            float v211 = fma(v210, v193, v208);
            float v212 = float(as_type<half>(ushort(v209 >> 16)));
            float v213 = fma(v212, v194, v211);
            uint v214 = v202.z;
            float v215 = float(as_type<half>(ushort(v214)));
            float v216 = fma(v215, v196, v213);
            float v217 = float(as_type<half>(ushort(v214 >> 16)));
            float v218 = fma(v217, v197, v216);
            uint v219 = v202.w;
            float v220 = float(as_type<half>(ushort(v219)));
            float v221 = fma(v220, v199, v218);
            float v222 = float(as_type<half>(ushort(v219 >> 16)));
            float v223 = fma(v222, v200, v221);
            m0 = v223;
            uint v224 = v169 + v186;
            uint4 v225 = b0[v224];
            float v226 = m1;
            uint v227 = v225.x;
            float v228 = float(as_type<half>(ushort(v227)));
            float v229 = fma(v228, v190, v226);
            float v230 = float(as_type<half>(ushort(v227 >> 16)));
            float v231 = fma(v230, v191, v229);
            uint v232 = v225.y;
            float v233 = float(as_type<half>(ushort(v232)));
            float v234 = fma(v233, v193, v231);
            float v235 = float(as_type<half>(ushort(v232 >> 16)));
            float v236 = fma(v235, v194, v234);
            uint v237 = v225.z;
            float v238 = float(as_type<half>(ushort(v237)));
            float v239 = fma(v238, v196, v236);
            float v240 = float(as_type<half>(ushort(v237 >> 16)));
            float v241 = fma(v240, v197, v239);
            uint v242 = v225.w;
            float v243 = float(as_type<half>(ushort(v242)));
            float v244 = fma(v243, v199, v241);
            float v245 = float(as_type<half>(ushort(v242 >> 16)));
            float v246 = fma(v245, v200, v244);
            m1 = v246;
            uint v247 = v174 + v186;
            uint4 v248 = b0[v247];
            float v249 = m2;
            uint v250 = v248.x;
            float v251 = float(as_type<half>(ushort(v250)));
            float v252 = fma(v251, v190, v249);
            float v253 = float(as_type<half>(ushort(v250 >> 16)));
            float v254 = fma(v253, v191, v252);
            uint v255 = v248.y;
            float v256 = float(as_type<half>(ushort(v255)));
            float v257 = fma(v256, v193, v254);
            float v258 = float(as_type<half>(ushort(v255 >> 16)));
            float v259 = fma(v258, v194, v257);
            uint v260 = v248.z;
            float v261 = float(as_type<half>(ushort(v260)));
            float v262 = fma(v261, v196, v259);
            float v263 = float(as_type<half>(ushort(v260 >> 16)));
            float v264 = fma(v263, v197, v262);
            uint v265 = v248.w;
            float v266 = float(as_type<half>(ushort(v265)));
            float v267 = fma(v266, v199, v264);
            float v268 = float(as_type<half>(ushort(v265 >> 16)));
            float v269 = fma(v268, v200, v267);
            m2 = v269;
            uint v270 = v179 + v186;
            uint4 v271 = b0[v270];
            float v272 = m3;
            uint v273 = v271.x;
            float v274 = float(as_type<half>(ushort(v273)));
            float v275 = fma(v274, v190, v272);
            float v276 = float(as_type<half>(ushort(v273 >> 16)));
            float v277 = fma(v276, v191, v275);
            uint v278 = v271.y;
            float v279 = float(as_type<half>(ushort(v278)));
            float v280 = fma(v279, v193, v277);
            float v281 = float(as_type<half>(ushort(v278 >> 16)));
            float v282 = fma(v281, v194, v280);
            uint v283 = v271.z;
            float v284 = float(as_type<half>(ushort(v283)));
            float v285 = fma(v284, v196, v282);
            float v286 = float(as_type<half>(ushort(v283 >> 16)));
            float v287 = fma(v286, v197, v285);
            uint v288 = v271.w;
            float v289 = float(as_type<half>(ushort(v288)));
            float v290 = fma(v289, v199, v287);
            float v291 = float(as_type<half>(ushort(v288 >> 16)));
            float v292 = fma(v291, v200, v290);
            m3 = v292;
        }
        bool v294 = lane == 0u;
        uint v295 = v146 * 4u;
        float v296 = m0;
        float v297 = simd_sum(v296);
        uint v299 = v295 + 0u;
        uint v300 = v299 * 1u;
        uint v301 = v300 + v147;
        if (v294) {
            s0[v301] = v297;
        }
        float v302 = m1;
        float v303 = simd_sum(v302);
        uint v305 = v295 + 1u;
        uint v306 = v305 * 1u;
        uint v307 = v306 + v147;
        if (v294) {
            s0[v307] = v303;
        }
        float v308 = m2;
        float v309 = simd_sum(v308);
        uint v311 = v295 + 2u;
        uint v312 = v311 * 1u;
        uint v313 = v312 + v147;
        if (v294) {
            s0[v313] = v309;
        }
        float v314 = m3;
        float v315 = simd_sum(v314);
        uint v317 = v295 + 3u;
        uint v318 = v317 * 1u;
        uint v319 = v318 + v147;
        if (v294) {
            s0[v319] = v315;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v320 = v147 == 0u;
        bool v321 = v294 && v320;
        uint v323 = v156 + 0u;
        bool v324 = v323 < 5120u;
        bool v325 = v321 && v324;
        if (v325) {
            uint v327 = v295 + 0u;
            uint v328 = v327 * 1u;
            float v329 = s0[v328];
            uint v330 = uint(as_type<ushort>(half(v329)));
            b2[v323] = ushort(v330);
        }
        uint v332 = v156 + 1u;
        bool v333 = v332 < 5120u;
        bool v334 = v321 && v333;
        if (v334) {
            uint v336 = v295 + 1u;
            uint v337 = v336 * 1u;
            float v338 = s0[v337];
            uint v339 = uint(as_type<ushort>(half(v338)));
            b2[v332] = ushort(v339);
        }
        uint v341 = v156 + 2u;
        bool v342 = v341 < 5120u;
        bool v343 = v321 && v342;
        if (v343) {
            uint v345 = v295 + 2u;
            uint v346 = v345 * 1u;
            float v347 = s0[v346];
            uint v348 = uint(as_type<ushort>(half(v347)));
            b2[v341] = ushort(v348);
        }
        uint v350 = v156 + 3u;
        bool v351 = v350 < 5120u;
        bool v352 = v321 && v351;
        if (v352) {
            uint v354 = v295 + 3u;
            uint v355 = v354 * 1u;
            float v356 = s0[v355];
            uint v357 = uint(as_type<ushort>(half(v356)));
            b2[v350] = ushort(v357);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
