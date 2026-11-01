#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_cs_r4_k2(
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
    threadgroup float s0[16];
    bool v2 = tgid.x < 128u;
    bool v3 = !v2;
    if (v3) {
        uint v4 = tgid.x - 128u;
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
    uint v48 = v45 * 4u;
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
    uint v74 = v42 * 32u;
    uint v75 = v74 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    for (uint v78 = v75; v78 < 256u; v78 += 64u) {
        uint4 v80 = b1[v78];
        uint v81 = v80.x;
        float v82 = float(as_type<half>(ushort(v81)));
        float v83 = float(as_type<half>(ushort(v81 >> 16)));
        uint v84 = v80.y;
        float v85 = float(as_type<half>(ushort(v84)));
        float v86 = float(as_type<half>(ushort(v84 >> 16)));
        uint v87 = v80.z;
        float v88 = float(as_type<half>(ushort(v87)));
        float v89 = float(as_type<half>(ushort(v87 >> 16)));
        uint v90 = v80.w;
        float v91 = float(as_type<half>(ushort(v90)));
        float v92 = float(as_type<half>(ushort(v90 >> 16)));
        uint v93 = v56 + v78;
        uint4 v94 = b0[v93];
        float v95 = m0;
        uint v96 = v94.x;
        float v97 = float(as_type<half>(ushort(v96)));
        float v98 = fma(v97, v82, v95);
        float v99 = float(as_type<half>(ushort(v96 >> 16)));
        float v100 = fma(v99, v83, v98);
        uint v101 = v94.y;
        float v102 = float(as_type<half>(ushort(v101)));
        float v103 = fma(v102, v85, v100);
        float v104 = float(as_type<half>(ushort(v101 >> 16)));
        float v105 = fma(v104, v86, v103);
        uint v106 = v94.z;
        float v107 = float(as_type<half>(ushort(v106)));
        float v108 = fma(v107, v88, v105);
        float v109 = float(as_type<half>(ushort(v106 >> 16)));
        float v110 = fma(v109, v89, v108);
        uint v111 = v94.w;
        float v112 = float(as_type<half>(ushort(v111)));
        float v113 = fma(v112, v91, v110);
        float v114 = float(as_type<half>(ushort(v111 >> 16)));
        float v115 = fma(v114, v92, v113);
        m0 = v115;
        uint v116 = v61 + v78;
        uint4 v117 = b0[v116];
        float v118 = m1;
        uint v119 = v117.x;
        float v120 = float(as_type<half>(ushort(v119)));
        float v121 = fma(v120, v82, v118);
        float v122 = float(as_type<half>(ushort(v119 >> 16)));
        float v123 = fma(v122, v83, v121);
        uint v124 = v117.y;
        float v125 = float(as_type<half>(ushort(v124)));
        float v126 = fma(v125, v85, v123);
        float v127 = float(as_type<half>(ushort(v124 >> 16)));
        float v128 = fma(v127, v86, v126);
        uint v129 = v117.z;
        float v130 = float(as_type<half>(ushort(v129)));
        float v131 = fma(v130, v88, v128);
        float v132 = float(as_type<half>(ushort(v129 >> 16)));
        float v133 = fma(v132, v89, v131);
        uint v134 = v117.w;
        float v135 = float(as_type<half>(ushort(v134)));
        float v136 = fma(v135, v91, v133);
        float v137 = float(as_type<half>(ushort(v134 >> 16)));
        float v138 = fma(v137, v92, v136);
        m1 = v138;
        uint v139 = v66 + v78;
        uint4 v140 = b0[v139];
        float v141 = m2;
        uint v142 = v140.x;
        float v143 = float(as_type<half>(ushort(v142)));
        float v144 = fma(v143, v82, v141);
        float v145 = float(as_type<half>(ushort(v142 >> 16)));
        float v146 = fma(v145, v83, v144);
        uint v147 = v140.y;
        float v148 = float(as_type<half>(ushort(v147)));
        float v149 = fma(v148, v85, v146);
        float v150 = float(as_type<half>(ushort(v147 >> 16)));
        float v151 = fma(v150, v86, v149);
        uint v152 = v140.z;
        float v153 = float(as_type<half>(ushort(v152)));
        float v154 = fma(v153, v88, v151);
        float v155 = float(as_type<half>(ushort(v152 >> 16)));
        float v156 = fma(v155, v89, v154);
        uint v157 = v140.w;
        float v158 = float(as_type<half>(ushort(v157)));
        float v159 = fma(v158, v91, v156);
        float v160 = float(as_type<half>(ushort(v157 >> 16)));
        float v161 = fma(v160, v92, v159);
        m2 = v161;
        uint v162 = v71 + v78;
        uint4 v163 = b0[v162];
        float v164 = m3;
        uint v165 = v163.x;
        float v166 = float(as_type<half>(ushort(v165)));
        float v167 = fma(v166, v82, v164);
        float v168 = float(as_type<half>(ushort(v165 >> 16)));
        float v169 = fma(v168, v83, v167);
        uint v170 = v163.y;
        float v171 = float(as_type<half>(ushort(v170)));
        float v172 = fma(v171, v85, v169);
        float v173 = float(as_type<half>(ushort(v170 >> 16)));
        float v174 = fma(v173, v86, v172);
        uint v175 = v163.z;
        float v176 = float(as_type<half>(ushort(v175)));
        float v177 = fma(v176, v88, v174);
        float v178 = float(as_type<half>(ushort(v175 >> 16)));
        float v179 = fma(v178, v89, v177);
        uint v180 = v163.w;
        float v181 = float(as_type<half>(ushort(v180)));
        float v182 = fma(v181, v91, v179);
        float v183 = float(as_type<half>(ushort(v180 >> 16)));
        float v184 = fma(v183, v92, v182);
        m3 = v184;
    }
    bool v186 = lane == 0u;
    uint v187 = v41 * 4u;
    float v188 = m0;
    float v189 = simd_sum(v188);
    uint v191 = v187 + 0u;
    uint v192 = v191 * 2u;
    uint v193 = v192 + v42;
    if (v186) {
        s0[v193] = v189;
    }
    float v194 = m1;
    float v195 = simd_sum(v194);
    uint v197 = v187 + 1u;
    uint v198 = v197 * 2u;
    uint v199 = v198 + v42;
    if (v186) {
        s0[v199] = v195;
    }
    float v200 = m2;
    float v201 = simd_sum(v200);
    uint v203 = v187 + 2u;
    uint v204 = v203 * 2u;
    uint v205 = v204 + v42;
    if (v186) {
        s0[v205] = v201;
    }
    float v206 = m3;
    float v207 = simd_sum(v206);
    uint v209 = v187 + 3u;
    uint v210 = v209 * 2u;
    uint v211 = v210 + v42;
    if (v186) {
        s0[v211] = v207;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v212 = v42 == 0u;
    bool v213 = v186 && v212;
    uint v215 = v48 + 0u;
    bool v216 = v215 < 1024u;
    bool v217 = v213 && v216;
    if (v217) {
        uint v219 = v187 + 0u;
        uint v220 = v219 * 2u;
        float v221 = s0[v220];
        uint v223 = v220 + 1u;
        float v224 = s0[v223];
        float v225 = v221 + v224;
        float v226 = float(half(v225));
        uint v227 = uint(b2[v215]);
        float v228 = float(as_type<half>(ushort(v227)));
        float v229 = v228 + v226;
        uint v230 = uint(as_type<ushort>(half(v229)));
        b2[v215] = ushort(v230);
    }
    uint v232 = v48 + 1u;
    bool v233 = v232 < 1024u;
    bool v234 = v213 && v233;
    if (v234) {
        uint v236 = v187 + 1u;
        uint v237 = v236 * 2u;
        float v238 = s0[v237];
        uint v240 = v237 + 1u;
        float v241 = s0[v240];
        float v242 = v238 + v241;
        float v243 = float(half(v242));
        uint v244 = uint(b2[v232]);
        float v245 = float(as_type<half>(ushort(v244)));
        float v246 = v245 + v243;
        uint v247 = uint(as_type<ushort>(half(v246)));
        b2[v232] = ushort(v247);
    }
    uint v249 = v48 + 2u;
    bool v250 = v249 < 1024u;
    bool v251 = v213 && v250;
    if (v251) {
        uint v253 = v187 + 2u;
        uint v254 = v253 * 2u;
        float v255 = s0[v254];
        uint v257 = v254 + 1u;
        float v258 = s0[v257];
        float v259 = v255 + v258;
        float v260 = float(half(v259));
        uint v261 = uint(b2[v249]);
        float v262 = float(as_type<half>(ushort(v261)));
        float v263 = v262 + v260;
        uint v264 = uint(as_type<ushort>(half(v263)));
        b2[v249] = ushort(v264);
    }
    uint v266 = v48 + 3u;
    bool v267 = v266 < 1024u;
    bool v268 = v213 && v267;
    if (v268) {
        uint v270 = v187 + 3u;
        uint v271 = v270 * 2u;
        float v272 = s0[v271];
        uint v274 = v271 + 1u;
        float v275 = s0[v274];
        float v276 = v272 + v275;
        float v277 = float(half(v276));
        uint v278 = uint(b2[v266]);
        float v279 = float(as_type<half>(ushort(v278)));
        float v280 = v279 + v277;
        uint v281 = uint(as_type<ushort>(half(v280)));
        b2[v266] = ushort(v281);
    }
}
