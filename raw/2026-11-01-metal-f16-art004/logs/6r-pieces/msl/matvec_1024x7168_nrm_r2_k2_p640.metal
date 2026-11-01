#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x7168_nrm_r2_k2_p640(
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
    uint v146 = sgid / 2u;
    uint v147 = sgid % 2u;
    for (uint v150 = tgid.x; v150 < 1792u; v150 += 640u) {
        uint v152 = v150 * 2u;
        uint v153 = v152 + v146;
        uint v156 = v153 * 2u;
        uint v161 = v156 + 0u;
        bool v162 = v161 < 7168u;
        uint v163 = v162 ? v161 : 7167u;
        uint v164 = v163 * 128u;
        uint v166 = v156 + 1u;
        bool v167 = v166 < 7168u;
        uint v168 = v167 ? v166 : 7167u;
        uint v169 = v168 * 128u;
        uint v172 = v147 * 32u;
        uint v173 = v172 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        for (uint v176 = v173; v176 < 128u; v176 += 64u) {
            uint4 v178 = s2[v176];
            uint v179 = v178.x;
            float v180 = float(as_type<half>(ushort(v179)));
            float v181 = float(as_type<half>(ushort(v179 >> 16)));
            uint v182 = v178.y;
            float v183 = float(as_type<half>(ushort(v182)));
            float v184 = float(as_type<half>(ushort(v182 >> 16)));
            uint v185 = v178.z;
            float v186 = float(as_type<half>(ushort(v185)));
            float v187 = float(as_type<half>(ushort(v185 >> 16)));
            uint v188 = v178.w;
            float v189 = float(as_type<half>(ushort(v188)));
            float v190 = float(as_type<half>(ushort(v188 >> 16)));
            uint v191 = v164 + v176;
            uint4 v192 = b0[v191];
            float v193 = m0;
            uint v194 = v192.x;
            float v195 = float(as_type<half>(ushort(v194)));
            float v196 = fma(v195, v180, v193);
            float v197 = float(as_type<half>(ushort(v194 >> 16)));
            float v198 = fma(v197, v181, v196);
            uint v199 = v192.y;
            float v200 = float(as_type<half>(ushort(v199)));
            float v201 = fma(v200, v183, v198);
            float v202 = float(as_type<half>(ushort(v199 >> 16)));
            float v203 = fma(v202, v184, v201);
            uint v204 = v192.z;
            float v205 = float(as_type<half>(ushort(v204)));
            float v206 = fma(v205, v186, v203);
            float v207 = float(as_type<half>(ushort(v204 >> 16)));
            float v208 = fma(v207, v187, v206);
            uint v209 = v192.w;
            float v210 = float(as_type<half>(ushort(v209)));
            float v211 = fma(v210, v189, v208);
            float v212 = float(as_type<half>(ushort(v209 >> 16)));
            float v213 = fma(v212, v190, v211);
            m0 = v213;
            uint v214 = v169 + v176;
            uint4 v215 = b0[v214];
            float v216 = m1;
            uint v217 = v215.x;
            float v218 = float(as_type<half>(ushort(v217)));
            float v219 = fma(v218, v180, v216);
            float v220 = float(as_type<half>(ushort(v217 >> 16)));
            float v221 = fma(v220, v181, v219);
            uint v222 = v215.y;
            float v223 = float(as_type<half>(ushort(v222)));
            float v224 = fma(v223, v183, v221);
            float v225 = float(as_type<half>(ushort(v222 >> 16)));
            float v226 = fma(v225, v184, v224);
            uint v227 = v215.z;
            float v228 = float(as_type<half>(ushort(v227)));
            float v229 = fma(v228, v186, v226);
            float v230 = float(as_type<half>(ushort(v227 >> 16)));
            float v231 = fma(v230, v187, v229);
            uint v232 = v215.w;
            float v233 = float(as_type<half>(ushort(v232)));
            float v234 = fma(v233, v189, v231);
            float v235 = float(as_type<half>(ushort(v232 >> 16)));
            float v236 = fma(v235, v190, v234);
            m1 = v236;
        }
        bool v238 = lane == 0u;
        uint v239 = v146 * 2u;
        float v240 = m0;
        float v241 = simd_sum(v240);
        uint v243 = v239 + 0u;
        uint v244 = v243 * 2u;
        uint v245 = v244 + v147;
        if (v238) {
            s0[v245] = v241;
        }
        float v246 = m1;
        float v247 = simd_sum(v246);
        uint v249 = v239 + 1u;
        uint v250 = v249 * 2u;
        uint v251 = v250 + v147;
        if (v238) {
            s0[v251] = v247;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v252 = v147 == 0u;
        bool v253 = v238 && v252;
        uint v255 = v156 + 0u;
        bool v256 = v255 < 7168u;
        bool v257 = v253 && v256;
        if (v257) {
            uint v259 = v239 + 0u;
            uint v260 = v259 * 2u;
            float v261 = s0[v260];
            uint v263 = v260 + 1u;
            float v264 = s0[v263];
            float v265 = v261 + v264;
            uint v266 = uint(as_type<ushort>(half(v265)));
            b2[v255] = ushort(v266);
        }
        uint v268 = v156 + 1u;
        bool v269 = v268 < 7168u;
        bool v270 = v253 && v269;
        if (v270) {
            uint v272 = v239 + 1u;
            uint v273 = v272 * 2u;
            float v274 = s0[v273];
            uint v276 = v273 + 1u;
            float v277 = s0[v276];
            float v278 = v274 + v277;
            uint v279 = uint(as_type<ushort>(half(v278)));
            b2[v268] = ushort(v279);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
