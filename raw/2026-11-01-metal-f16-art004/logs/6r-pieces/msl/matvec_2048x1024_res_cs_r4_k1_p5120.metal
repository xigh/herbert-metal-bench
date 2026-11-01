#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_cs_r4_k1_p5120(
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
    uint v41 = sgid / 1u;
    uint v42 = sgid % 1u;
    for (uint v45 = tgid.x; v45 < 64u; v45 += 5120u) {
        uint v47 = v45 * 4u;
        uint v48 = v47 + v41;
        uint v51 = v48 * 4u;
        uint v56 = v51 + 0u;
        bool v57 = v56 < 1024u;
        uint v58 = v57 ? v56 : 1023u;
        uint v59 = v58 * 256u;
        uint v61 = v51 + 1u;
        bool v62 = v61 < 1024u;
        uint v63 = v62 ? v61 : 1023u;
        uint v64 = v63 * 256u;
        uint v66 = v51 + 2u;
        bool v67 = v66 < 1024u;
        uint v68 = v67 ? v66 : 1023u;
        uint v69 = v68 * 256u;
        uint v71 = v51 + 3u;
        bool v72 = v71 < 1024u;
        uint v73 = v72 ? v71 : 1023u;
        uint v74 = v73 * 256u;
        uint v77 = v42 * 32u;
        uint v78 = v77 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        float m2 = as_type<float>(0x00000000u);
        float m3 = as_type<float>(0x00000000u);
        for (uint v81 = v78; v81 < 256u; v81 += 32u) {
            uint4 v83 = b1[v81];
            uint v84 = v83.x;
            float v85 = float(as_type<half>(ushort(v84)));
            float v86 = float(as_type<half>(ushort(v84 >> 16)));
            uint v87 = v83.y;
            float v88 = float(as_type<half>(ushort(v87)));
            float v89 = float(as_type<half>(ushort(v87 >> 16)));
            uint v90 = v83.z;
            float v91 = float(as_type<half>(ushort(v90)));
            float v92 = float(as_type<half>(ushort(v90 >> 16)));
            uint v93 = v83.w;
            float v94 = float(as_type<half>(ushort(v93)));
            float v95 = float(as_type<half>(ushort(v93 >> 16)));
            uint v96 = v59 + v81;
            uint4 v97 = b0[v96];
            float v98 = m0;
            uint v99 = v97.x;
            float v100 = float(as_type<half>(ushort(v99)));
            float v101 = fma(v100, v85, v98);
            float v102 = float(as_type<half>(ushort(v99 >> 16)));
            float v103 = fma(v102, v86, v101);
            uint v104 = v97.y;
            float v105 = float(as_type<half>(ushort(v104)));
            float v106 = fma(v105, v88, v103);
            float v107 = float(as_type<half>(ushort(v104 >> 16)));
            float v108 = fma(v107, v89, v106);
            uint v109 = v97.z;
            float v110 = float(as_type<half>(ushort(v109)));
            float v111 = fma(v110, v91, v108);
            float v112 = float(as_type<half>(ushort(v109 >> 16)));
            float v113 = fma(v112, v92, v111);
            uint v114 = v97.w;
            float v115 = float(as_type<half>(ushort(v114)));
            float v116 = fma(v115, v94, v113);
            float v117 = float(as_type<half>(ushort(v114 >> 16)));
            float v118 = fma(v117, v95, v116);
            m0 = v118;
            uint v119 = v64 + v81;
            uint4 v120 = b0[v119];
            float v121 = m1;
            uint v122 = v120.x;
            float v123 = float(as_type<half>(ushort(v122)));
            float v124 = fma(v123, v85, v121);
            float v125 = float(as_type<half>(ushort(v122 >> 16)));
            float v126 = fma(v125, v86, v124);
            uint v127 = v120.y;
            float v128 = float(as_type<half>(ushort(v127)));
            float v129 = fma(v128, v88, v126);
            float v130 = float(as_type<half>(ushort(v127 >> 16)));
            float v131 = fma(v130, v89, v129);
            uint v132 = v120.z;
            float v133 = float(as_type<half>(ushort(v132)));
            float v134 = fma(v133, v91, v131);
            float v135 = float(as_type<half>(ushort(v132 >> 16)));
            float v136 = fma(v135, v92, v134);
            uint v137 = v120.w;
            float v138 = float(as_type<half>(ushort(v137)));
            float v139 = fma(v138, v94, v136);
            float v140 = float(as_type<half>(ushort(v137 >> 16)));
            float v141 = fma(v140, v95, v139);
            m1 = v141;
            uint v142 = v69 + v81;
            uint4 v143 = b0[v142];
            float v144 = m2;
            uint v145 = v143.x;
            float v146 = float(as_type<half>(ushort(v145)));
            float v147 = fma(v146, v85, v144);
            float v148 = float(as_type<half>(ushort(v145 >> 16)));
            float v149 = fma(v148, v86, v147);
            uint v150 = v143.y;
            float v151 = float(as_type<half>(ushort(v150)));
            float v152 = fma(v151, v88, v149);
            float v153 = float(as_type<half>(ushort(v150 >> 16)));
            float v154 = fma(v153, v89, v152);
            uint v155 = v143.z;
            float v156 = float(as_type<half>(ushort(v155)));
            float v157 = fma(v156, v91, v154);
            float v158 = float(as_type<half>(ushort(v155 >> 16)));
            float v159 = fma(v158, v92, v157);
            uint v160 = v143.w;
            float v161 = float(as_type<half>(ushort(v160)));
            float v162 = fma(v161, v94, v159);
            float v163 = float(as_type<half>(ushort(v160 >> 16)));
            float v164 = fma(v163, v95, v162);
            m2 = v164;
            uint v165 = v74 + v81;
            uint4 v166 = b0[v165];
            float v167 = m3;
            uint v168 = v166.x;
            float v169 = float(as_type<half>(ushort(v168)));
            float v170 = fma(v169, v85, v167);
            float v171 = float(as_type<half>(ushort(v168 >> 16)));
            float v172 = fma(v171, v86, v170);
            uint v173 = v166.y;
            float v174 = float(as_type<half>(ushort(v173)));
            float v175 = fma(v174, v88, v172);
            float v176 = float(as_type<half>(ushort(v173 >> 16)));
            float v177 = fma(v176, v89, v175);
            uint v178 = v166.z;
            float v179 = float(as_type<half>(ushort(v178)));
            float v180 = fma(v179, v91, v177);
            float v181 = float(as_type<half>(ushort(v178 >> 16)));
            float v182 = fma(v181, v92, v180);
            uint v183 = v166.w;
            float v184 = float(as_type<half>(ushort(v183)));
            float v185 = fma(v184, v94, v182);
            float v186 = float(as_type<half>(ushort(v183 >> 16)));
            float v187 = fma(v186, v95, v185);
            m3 = v187;
        }
        bool v189 = lane == 0u;
        uint v190 = v41 * 4u;
        float v191 = m0;
        float v192 = simd_sum(v191);
        uint v194 = v190 + 0u;
        uint v195 = v194 * 1u;
        uint v196 = v195 + v42;
        if (v189) {
            s0[v196] = v192;
        }
        float v197 = m1;
        float v198 = simd_sum(v197);
        uint v200 = v190 + 1u;
        uint v201 = v200 * 1u;
        uint v202 = v201 + v42;
        if (v189) {
            s0[v202] = v198;
        }
        float v203 = m2;
        float v204 = simd_sum(v203);
        uint v206 = v190 + 2u;
        uint v207 = v206 * 1u;
        uint v208 = v207 + v42;
        if (v189) {
            s0[v208] = v204;
        }
        float v209 = m3;
        float v210 = simd_sum(v209);
        uint v212 = v190 + 3u;
        uint v213 = v212 * 1u;
        uint v214 = v213 + v42;
        if (v189) {
            s0[v214] = v210;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v215 = v42 == 0u;
        bool v216 = v189 && v215;
        uint v218 = v51 + 0u;
        bool v219 = v218 < 1024u;
        bool v220 = v216 && v219;
        if (v220) {
            uint v222 = v190 + 0u;
            uint v223 = v222 * 1u;
            float v224 = s0[v223];
            float v225 = float(half(v224));
            uint v226 = uint(b2[v218]);
            float v227 = float(as_type<half>(ushort(v226)));
            float v228 = v227 + v225;
            uint v229 = uint(as_type<ushort>(half(v228)));
            b2[v218] = ushort(v229);
        }
        uint v231 = v51 + 1u;
        bool v232 = v231 < 1024u;
        bool v233 = v216 && v232;
        if (v233) {
            uint v235 = v190 + 1u;
            uint v236 = v235 * 1u;
            float v237 = s0[v236];
            float v238 = float(half(v237));
            uint v239 = uint(b2[v231]);
            float v240 = float(as_type<half>(ushort(v239)));
            float v241 = v240 + v238;
            uint v242 = uint(as_type<ushort>(half(v241)));
            b2[v231] = ushort(v242);
        }
        uint v244 = v51 + 2u;
        bool v245 = v244 < 1024u;
        bool v246 = v216 && v245;
        if (v246) {
            uint v248 = v190 + 2u;
            uint v249 = v248 * 1u;
            float v250 = s0[v249];
            float v251 = float(half(v250));
            uint v252 = uint(b2[v244]);
            float v253 = float(as_type<half>(ushort(v252)));
            float v254 = v253 + v251;
            uint v255 = uint(as_type<ushort>(half(v254)));
            b2[v244] = ushort(v255);
        }
        uint v257 = v51 + 3u;
        bool v258 = v257 < 1024u;
        bool v259 = v216 && v258;
        if (v259) {
            uint v261 = v190 + 3u;
            uint v262 = v261 * 1u;
            float v263 = s0[v262];
            float v264 = float(half(v263));
            uint v265 = uint(b2[v257]);
            float v266 = float(as_type<half>(ushort(v265)));
            float v267 = v266 + v264;
            uint v268 = uint(as_type<ushort>(half(v267)));
            b2[v257] = ushort(v268);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
