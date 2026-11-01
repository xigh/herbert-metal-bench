#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_swi_res_r4_k1(
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
    threadgroup float s0[16];
    uint v3 = sgid / 1u;
    uint v4 = sgid % 1u;
    uint v6 = tgid.x * 4u;
    uint v7 = v6 + v3;
    uint v10 = v7 * 4u;
    uint v15 = v10 + 0u;
    bool v16 = v15 < 1024u;
    uint v17 = v16 ? v15 : 1023u;
    uint v18 = v17 * 448u;
    uint v20 = v10 + 1u;
    bool v21 = v20 < 1024u;
    uint v22 = v21 ? v20 : 1023u;
    uint v23 = v22 * 448u;
    uint v25 = v10 + 2u;
    bool v26 = v25 < 1024u;
    uint v27 = v26 ? v25 : 1023u;
    uint v28 = v27 * 448u;
    uint v30 = v10 + 3u;
    bool v31 = v30 < 1024u;
    uint v32 = v31 ? v30 : 1023u;
    uint v33 = v32 * 448u;
    uint v36 = v4 * 32u;
    uint v37 = v36 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    float m2 = as_type<float>(0x00000000u);
    float m3 = as_type<float>(0x00000000u);
    for (uint v40 = v37; v40 < 448u; v40 += 32u) {
        uint4 v42 = b1[v40];
        uint v43 = v40 + 448u;
        uint4 v44 = b1[v43];
        uint v45 = v42.x;
        float v46 = float(as_type<half>(ushort(v45)));
        float v47 = float(as_type<half>(ushort(v45 >> 16)));
        uint v48 = v44.x;
        float v49 = float(as_type<half>(ushort(v48)));
        float v50 = float(as_type<half>(ushort(v48 >> 16)));
        float v51 = -v46;
        float v52 = exp(v51);
        float v54 = as_type<float>(0x3f800000u) + v52;
        float v55 = as_type<float>(0x3f800000u) / v54;
        float v56 = v46 * v55;
        float v57 = v56 * v49;
        float v58 = float(half(v57));
        float v59 = -v47;
        float v60 = exp(v59);
        float v62 = as_type<float>(0x3f800000u) + v60;
        float v63 = as_type<float>(0x3f800000u) / v62;
        float v64 = v47 * v63;
        float v65 = v64 * v50;
        float v66 = float(half(v65));
        uint v67 = v42.y;
        float v68 = float(as_type<half>(ushort(v67)));
        float v69 = float(as_type<half>(ushort(v67 >> 16)));
        uint v70 = v44.y;
        float v71 = float(as_type<half>(ushort(v70)));
        float v72 = float(as_type<half>(ushort(v70 >> 16)));
        float v73 = -v68;
        float v74 = exp(v73);
        float v76 = as_type<float>(0x3f800000u) + v74;
        float v77 = as_type<float>(0x3f800000u) / v76;
        float v78 = v68 * v77;
        float v79 = v78 * v71;
        float v80 = float(half(v79));
        float v81 = -v69;
        float v82 = exp(v81);
        float v84 = as_type<float>(0x3f800000u) + v82;
        float v85 = as_type<float>(0x3f800000u) / v84;
        float v86 = v69 * v85;
        float v87 = v86 * v72;
        float v88 = float(half(v87));
        uint v89 = v42.z;
        float v90 = float(as_type<half>(ushort(v89)));
        float v91 = float(as_type<half>(ushort(v89 >> 16)));
        uint v92 = v44.z;
        float v93 = float(as_type<half>(ushort(v92)));
        float v94 = float(as_type<half>(ushort(v92 >> 16)));
        float v95 = -v90;
        float v96 = exp(v95);
        float v98 = as_type<float>(0x3f800000u) + v96;
        float v99 = as_type<float>(0x3f800000u) / v98;
        float v100 = v90 * v99;
        float v101 = v100 * v93;
        float v102 = float(half(v101));
        float v103 = -v91;
        float v104 = exp(v103);
        float v106 = as_type<float>(0x3f800000u) + v104;
        float v107 = as_type<float>(0x3f800000u) / v106;
        float v108 = v91 * v107;
        float v109 = v108 * v94;
        float v110 = float(half(v109));
        uint v111 = v42.w;
        float v112 = float(as_type<half>(ushort(v111)));
        float v113 = float(as_type<half>(ushort(v111 >> 16)));
        uint v114 = v44.w;
        float v115 = float(as_type<half>(ushort(v114)));
        float v116 = float(as_type<half>(ushort(v114 >> 16)));
        float v117 = -v112;
        float v118 = exp(v117);
        float v120 = as_type<float>(0x3f800000u) + v118;
        float v121 = as_type<float>(0x3f800000u) / v120;
        float v122 = v112 * v121;
        float v123 = v122 * v115;
        float v124 = float(half(v123));
        float v125 = -v113;
        float v126 = exp(v125);
        float v128 = as_type<float>(0x3f800000u) + v126;
        float v129 = as_type<float>(0x3f800000u) / v128;
        float v130 = v113 * v129;
        float v131 = v130 * v116;
        float v132 = float(half(v131));
        uint v133 = v18 + v40;
        uint4 v134 = b0[v133];
        float v135 = m0;
        uint v136 = v134.x;
        float v137 = float(as_type<half>(ushort(v136)));
        float v138 = fma(v137, v58, v135);
        float v139 = float(as_type<half>(ushort(v136 >> 16)));
        float v140 = fma(v139, v66, v138);
        uint v141 = v134.y;
        float v142 = float(as_type<half>(ushort(v141)));
        float v143 = fma(v142, v80, v140);
        float v144 = float(as_type<half>(ushort(v141 >> 16)));
        float v145 = fma(v144, v88, v143);
        uint v146 = v134.z;
        float v147 = float(as_type<half>(ushort(v146)));
        float v148 = fma(v147, v102, v145);
        float v149 = float(as_type<half>(ushort(v146 >> 16)));
        float v150 = fma(v149, v110, v148);
        uint v151 = v134.w;
        float v152 = float(as_type<half>(ushort(v151)));
        float v153 = fma(v152, v124, v150);
        float v154 = float(as_type<half>(ushort(v151 >> 16)));
        float v155 = fma(v154, v132, v153);
        m0 = v155;
        uint v156 = v23 + v40;
        uint4 v157 = b0[v156];
        float v158 = m1;
        uint v159 = v157.x;
        float v160 = float(as_type<half>(ushort(v159)));
        float v161 = fma(v160, v58, v158);
        float v162 = float(as_type<half>(ushort(v159 >> 16)));
        float v163 = fma(v162, v66, v161);
        uint v164 = v157.y;
        float v165 = float(as_type<half>(ushort(v164)));
        float v166 = fma(v165, v80, v163);
        float v167 = float(as_type<half>(ushort(v164 >> 16)));
        float v168 = fma(v167, v88, v166);
        uint v169 = v157.z;
        float v170 = float(as_type<half>(ushort(v169)));
        float v171 = fma(v170, v102, v168);
        float v172 = float(as_type<half>(ushort(v169 >> 16)));
        float v173 = fma(v172, v110, v171);
        uint v174 = v157.w;
        float v175 = float(as_type<half>(ushort(v174)));
        float v176 = fma(v175, v124, v173);
        float v177 = float(as_type<half>(ushort(v174 >> 16)));
        float v178 = fma(v177, v132, v176);
        m1 = v178;
        uint v179 = v28 + v40;
        uint4 v180 = b0[v179];
        float v181 = m2;
        uint v182 = v180.x;
        float v183 = float(as_type<half>(ushort(v182)));
        float v184 = fma(v183, v58, v181);
        float v185 = float(as_type<half>(ushort(v182 >> 16)));
        float v186 = fma(v185, v66, v184);
        uint v187 = v180.y;
        float v188 = float(as_type<half>(ushort(v187)));
        float v189 = fma(v188, v80, v186);
        float v190 = float(as_type<half>(ushort(v187 >> 16)));
        float v191 = fma(v190, v88, v189);
        uint v192 = v180.z;
        float v193 = float(as_type<half>(ushort(v192)));
        float v194 = fma(v193, v102, v191);
        float v195 = float(as_type<half>(ushort(v192 >> 16)));
        float v196 = fma(v195, v110, v194);
        uint v197 = v180.w;
        float v198 = float(as_type<half>(ushort(v197)));
        float v199 = fma(v198, v124, v196);
        float v200 = float(as_type<half>(ushort(v197 >> 16)));
        float v201 = fma(v200, v132, v199);
        m2 = v201;
        uint v202 = v33 + v40;
        uint4 v203 = b0[v202];
        float v204 = m3;
        uint v205 = v203.x;
        float v206 = float(as_type<half>(ushort(v205)));
        float v207 = fma(v206, v58, v204);
        float v208 = float(as_type<half>(ushort(v205 >> 16)));
        float v209 = fma(v208, v66, v207);
        uint v210 = v203.y;
        float v211 = float(as_type<half>(ushort(v210)));
        float v212 = fma(v211, v80, v209);
        float v213 = float(as_type<half>(ushort(v210 >> 16)));
        float v214 = fma(v213, v88, v212);
        uint v215 = v203.z;
        float v216 = float(as_type<half>(ushort(v215)));
        float v217 = fma(v216, v102, v214);
        float v218 = float(as_type<half>(ushort(v215 >> 16)));
        float v219 = fma(v218, v110, v217);
        uint v220 = v203.w;
        float v221 = float(as_type<half>(ushort(v220)));
        float v222 = fma(v221, v124, v219);
        float v223 = float(as_type<half>(ushort(v220 >> 16)));
        float v224 = fma(v223, v132, v222);
        m3 = v224;
    }
    bool v226 = lane == 0u;
    uint v227 = v3 * 4u;
    float v228 = m0;
    float v229 = simd_sum(v228);
    uint v231 = v227 + 0u;
    uint v232 = v231 * 1u;
    uint v233 = v232 + v4;
    if (v226) {
        s0[v233] = v229;
    }
    float v234 = m1;
    float v235 = simd_sum(v234);
    uint v237 = v227 + 1u;
    uint v238 = v237 * 1u;
    uint v239 = v238 + v4;
    if (v226) {
        s0[v239] = v235;
    }
    float v240 = m2;
    float v241 = simd_sum(v240);
    uint v243 = v227 + 2u;
    uint v244 = v243 * 1u;
    uint v245 = v244 + v4;
    if (v226) {
        s0[v245] = v241;
    }
    float v246 = m3;
    float v247 = simd_sum(v246);
    uint v249 = v227 + 3u;
    uint v250 = v249 * 1u;
    uint v251 = v250 + v4;
    if (v226) {
        s0[v251] = v247;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v252 = v4 == 0u;
    bool v253 = v226 && v252;
    uint v255 = v10 + 0u;
    bool v256 = v255 < 1024u;
    bool v257 = v253 && v256;
    if (v257) {
        uint v259 = v227 + 0u;
        uint v260 = v259 * 1u;
        float v261 = s0[v260];
        float v262 = float(half(v261));
        uint v263 = uint(b2[v255]);
        float v264 = float(as_type<half>(ushort(v263)));
        float v265 = v264 + v262;
        uint v266 = uint(as_type<ushort>(half(v265)));
        b2[v255] = ushort(v266);
    }
    uint v268 = v10 + 1u;
    bool v269 = v268 < 1024u;
    bool v270 = v253 && v269;
    if (v270) {
        uint v272 = v227 + 1u;
        uint v273 = v272 * 1u;
        float v274 = s0[v273];
        float v275 = float(half(v274));
        uint v276 = uint(b2[v268]);
        float v277 = float(as_type<half>(ushort(v276)));
        float v278 = v277 + v275;
        uint v279 = uint(as_type<ushort>(half(v278)));
        b2[v268] = ushort(v279);
    }
    uint v281 = v10 + 2u;
    bool v282 = v281 < 1024u;
    bool v283 = v253 && v282;
    if (v283) {
        uint v285 = v227 + 2u;
        uint v286 = v285 * 1u;
        float v287 = s0[v286];
        float v288 = float(half(v287));
        uint v289 = uint(b2[v281]);
        float v290 = float(as_type<half>(ushort(v289)));
        float v291 = v290 + v288;
        uint v292 = uint(as_type<ushort>(half(v291)));
        b2[v281] = ushort(v292);
    }
    uint v294 = v10 + 3u;
    bool v295 = v294 < 1024u;
    bool v296 = v253 && v295;
    if (v296) {
        uint v298 = v227 + 3u;
        uint v299 = v298 * 1u;
        float v300 = s0[v299];
        float v301 = float(half(v300));
        uint v302 = uint(b2[v294]);
        float v303 = float(as_type<half>(ushort(v302)));
        float v304 = v303 + v301;
        uint v305 = uint(as_type<ushort>(half(v304)));
        b2[v294] = ushort(v305);
    }
}
