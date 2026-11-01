#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x8224_nrm_r2_k1(
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
    threadgroup float s0[8];
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
    uint v149 = tgid.x * 4u;
    uint v150 = v149 + v146;
    uint v153 = v150 * 2u;
    uint v158 = v153 + 0u;
    bool v159 = v158 < 8224u;
    uint v160 = v159 ? v158 : 8223u;
    uint v161 = v160 * 128u;
    uint v163 = v153 + 1u;
    bool v164 = v163 < 8224u;
    uint v165 = v164 ? v163 : 8223u;
    uint v166 = v165 * 128u;
    uint v169 = v147 * 32u;
    uint v170 = v169 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    for (uint v173 = v170; v173 < 128u; v173 += 32u) {
        uint4 v175 = s2[v173];
        uint v176 = v175.x;
        float v177 = float(as_type<half>(ushort(v176)));
        float v178 = float(as_type<half>(ushort(v176 >> 16)));
        uint v179 = v175.y;
        float v180 = float(as_type<half>(ushort(v179)));
        float v181 = float(as_type<half>(ushort(v179 >> 16)));
        uint v182 = v175.z;
        float v183 = float(as_type<half>(ushort(v182)));
        float v184 = float(as_type<half>(ushort(v182 >> 16)));
        uint v185 = v175.w;
        float v186 = float(as_type<half>(ushort(v185)));
        float v187 = float(as_type<half>(ushort(v185 >> 16)));
        uint v188 = v161 + v173;
        uint4 v189 = b0[v188];
        float v190 = m0;
        uint v191 = v189.x;
        float v192 = float(as_type<half>(ushort(v191)));
        float v193 = fma(v192, v177, v190);
        float v194 = float(as_type<half>(ushort(v191 >> 16)));
        float v195 = fma(v194, v178, v193);
        uint v196 = v189.y;
        float v197 = float(as_type<half>(ushort(v196)));
        float v198 = fma(v197, v180, v195);
        float v199 = float(as_type<half>(ushort(v196 >> 16)));
        float v200 = fma(v199, v181, v198);
        uint v201 = v189.z;
        float v202 = float(as_type<half>(ushort(v201)));
        float v203 = fma(v202, v183, v200);
        float v204 = float(as_type<half>(ushort(v201 >> 16)));
        float v205 = fma(v204, v184, v203);
        uint v206 = v189.w;
        float v207 = float(as_type<half>(ushort(v206)));
        float v208 = fma(v207, v186, v205);
        float v209 = float(as_type<half>(ushort(v206 >> 16)));
        float v210 = fma(v209, v187, v208);
        m0 = v210;
        uint v211 = v166 + v173;
        uint4 v212 = b0[v211];
        float v213 = m1;
        uint v214 = v212.x;
        float v215 = float(as_type<half>(ushort(v214)));
        float v216 = fma(v215, v177, v213);
        float v217 = float(as_type<half>(ushort(v214 >> 16)));
        float v218 = fma(v217, v178, v216);
        uint v219 = v212.y;
        float v220 = float(as_type<half>(ushort(v219)));
        float v221 = fma(v220, v180, v218);
        float v222 = float(as_type<half>(ushort(v219 >> 16)));
        float v223 = fma(v222, v181, v221);
        uint v224 = v212.z;
        float v225 = float(as_type<half>(ushort(v224)));
        float v226 = fma(v225, v183, v223);
        float v227 = float(as_type<half>(ushort(v224 >> 16)));
        float v228 = fma(v227, v184, v226);
        uint v229 = v212.w;
        float v230 = float(as_type<half>(ushort(v229)));
        float v231 = fma(v230, v186, v228);
        float v232 = float(as_type<half>(ushort(v229 >> 16)));
        float v233 = fma(v232, v187, v231);
        m1 = v233;
    }
    bool v235 = lane == 0u;
    uint v236 = v146 * 2u;
    float v237 = m0;
    float v238 = simd_sum(v237);
    uint v240 = v236 + 0u;
    uint v241 = v240 * 1u;
    uint v242 = v241 + v147;
    if (v235) {
        s0[v242] = v238;
    }
    float v243 = m1;
    float v244 = simd_sum(v243);
    uint v246 = v236 + 1u;
    uint v247 = v246 * 1u;
    uint v248 = v247 + v147;
    if (v235) {
        s0[v248] = v244;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v249 = v147 == 0u;
    bool v250 = v235 && v249;
    uint v252 = v153 + 0u;
    bool v253 = v252 < 8224u;
    bool v254 = v250 && v253;
    if (v254) {
        uint v256 = v236 + 0u;
        uint v257 = v256 * 1u;
        float v258 = s0[v257];
        uint v259 = uint(as_type<ushort>(half(v258)));
        b2[v252] = ushort(v259);
    }
    uint v261 = v153 + 1u;
    bool v262 = v261 < 8224u;
    bool v263 = v250 && v262;
    if (v263) {
        uint v265 = v236 + 1u;
        uint v266 = v265 * 1u;
        float v267 = s0[v266];
        uint v268 = uint(as_type<ushort>(half(v267)));
        b2[v261] = ushort(v268);
    }
}
