#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_res_r4_k2_p1280(
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
    uint v3 = sgid / 2u;
    uint v4 = sgid % 2u;
    for (uint v7 = tgid.x; v7 < 128u; v7 += 1280u) {
        uint v9 = v7 * 2u;
        uint v10 = v9 + v3;
        uint v13 = v10 * 4u;
        uint v18 = v13 + 0u;
        bool v19 = v18 < 1024u;
        uint v20 = v19 ? v18 : 1023u;
        uint v21 = v20 * 448u;
        uint v23 = v13 + 1u;
        bool v24 = v23 < 1024u;
        uint v25 = v24 ? v23 : 1023u;
        uint v26 = v25 * 448u;
        uint v28 = v13 + 2u;
        bool v29 = v28 < 1024u;
        uint v30 = v29 ? v28 : 1023u;
        uint v31 = v30 * 448u;
        uint v33 = v13 + 3u;
        bool v34 = v33 < 1024u;
        uint v35 = v34 ? v33 : 1023u;
        uint v36 = v35 * 448u;
        uint v39 = v4 * 32u;
        uint v40 = v39 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        float m2 = as_type<float>(0x00000000u);
        float m3 = as_type<float>(0x00000000u);
        for (uint v43 = v40; v43 < 448u; v43 += 64u) {
            uint4 v45 = b1[v43];
            uint v46 = v45.x;
            float v47 = float(as_type<half>(ushort(v46)));
            float v48 = float(as_type<half>(ushort(v46 >> 16)));
            uint v49 = v45.y;
            float v50 = float(as_type<half>(ushort(v49)));
            float v51 = float(as_type<half>(ushort(v49 >> 16)));
            uint v52 = v45.z;
            float v53 = float(as_type<half>(ushort(v52)));
            float v54 = float(as_type<half>(ushort(v52 >> 16)));
            uint v55 = v45.w;
            float v56 = float(as_type<half>(ushort(v55)));
            float v57 = float(as_type<half>(ushort(v55 >> 16)));
            uint v58 = v21 + v43;
            uint4 v59 = b0[v58];
            float v60 = m0;
            uint v61 = v59.x;
            float v62 = float(as_type<half>(ushort(v61)));
            float v63 = fma(v62, v47, v60);
            float v64 = float(as_type<half>(ushort(v61 >> 16)));
            float v65 = fma(v64, v48, v63);
            uint v66 = v59.y;
            float v67 = float(as_type<half>(ushort(v66)));
            float v68 = fma(v67, v50, v65);
            float v69 = float(as_type<half>(ushort(v66 >> 16)));
            float v70 = fma(v69, v51, v68);
            uint v71 = v59.z;
            float v72 = float(as_type<half>(ushort(v71)));
            float v73 = fma(v72, v53, v70);
            float v74 = float(as_type<half>(ushort(v71 >> 16)));
            float v75 = fma(v74, v54, v73);
            uint v76 = v59.w;
            float v77 = float(as_type<half>(ushort(v76)));
            float v78 = fma(v77, v56, v75);
            float v79 = float(as_type<half>(ushort(v76 >> 16)));
            float v80 = fma(v79, v57, v78);
            m0 = v80;
            uint v81 = v26 + v43;
            uint4 v82 = b0[v81];
            float v83 = m1;
            uint v84 = v82.x;
            float v85 = float(as_type<half>(ushort(v84)));
            float v86 = fma(v85, v47, v83);
            float v87 = float(as_type<half>(ushort(v84 >> 16)));
            float v88 = fma(v87, v48, v86);
            uint v89 = v82.y;
            float v90 = float(as_type<half>(ushort(v89)));
            float v91 = fma(v90, v50, v88);
            float v92 = float(as_type<half>(ushort(v89 >> 16)));
            float v93 = fma(v92, v51, v91);
            uint v94 = v82.z;
            float v95 = float(as_type<half>(ushort(v94)));
            float v96 = fma(v95, v53, v93);
            float v97 = float(as_type<half>(ushort(v94 >> 16)));
            float v98 = fma(v97, v54, v96);
            uint v99 = v82.w;
            float v100 = float(as_type<half>(ushort(v99)));
            float v101 = fma(v100, v56, v98);
            float v102 = float(as_type<half>(ushort(v99 >> 16)));
            float v103 = fma(v102, v57, v101);
            m1 = v103;
            uint v104 = v31 + v43;
            uint4 v105 = b0[v104];
            float v106 = m2;
            uint v107 = v105.x;
            float v108 = float(as_type<half>(ushort(v107)));
            float v109 = fma(v108, v47, v106);
            float v110 = float(as_type<half>(ushort(v107 >> 16)));
            float v111 = fma(v110, v48, v109);
            uint v112 = v105.y;
            float v113 = float(as_type<half>(ushort(v112)));
            float v114 = fma(v113, v50, v111);
            float v115 = float(as_type<half>(ushort(v112 >> 16)));
            float v116 = fma(v115, v51, v114);
            uint v117 = v105.z;
            float v118 = float(as_type<half>(ushort(v117)));
            float v119 = fma(v118, v53, v116);
            float v120 = float(as_type<half>(ushort(v117 >> 16)));
            float v121 = fma(v120, v54, v119);
            uint v122 = v105.w;
            float v123 = float(as_type<half>(ushort(v122)));
            float v124 = fma(v123, v56, v121);
            float v125 = float(as_type<half>(ushort(v122 >> 16)));
            float v126 = fma(v125, v57, v124);
            m2 = v126;
            uint v127 = v36 + v43;
            uint4 v128 = b0[v127];
            float v129 = m3;
            uint v130 = v128.x;
            float v131 = float(as_type<half>(ushort(v130)));
            float v132 = fma(v131, v47, v129);
            float v133 = float(as_type<half>(ushort(v130 >> 16)));
            float v134 = fma(v133, v48, v132);
            uint v135 = v128.y;
            float v136 = float(as_type<half>(ushort(v135)));
            float v137 = fma(v136, v50, v134);
            float v138 = float(as_type<half>(ushort(v135 >> 16)));
            float v139 = fma(v138, v51, v137);
            uint v140 = v128.z;
            float v141 = float(as_type<half>(ushort(v140)));
            float v142 = fma(v141, v53, v139);
            float v143 = float(as_type<half>(ushort(v140 >> 16)));
            float v144 = fma(v143, v54, v142);
            uint v145 = v128.w;
            float v146 = float(as_type<half>(ushort(v145)));
            float v147 = fma(v146, v56, v144);
            float v148 = float(as_type<half>(ushort(v145 >> 16)));
            float v149 = fma(v148, v57, v147);
            m3 = v149;
        }
        bool v151 = lane == 0u;
        uint v152 = v3 * 4u;
        float v153 = m0;
        float v154 = simd_sum(v153);
        uint v156 = v152 + 0u;
        uint v157 = v156 * 2u;
        uint v158 = v157 + v4;
        if (v151) {
            s0[v158] = v154;
        }
        float v159 = m1;
        float v160 = simd_sum(v159);
        uint v162 = v152 + 1u;
        uint v163 = v162 * 2u;
        uint v164 = v163 + v4;
        if (v151) {
            s0[v164] = v160;
        }
        float v165 = m2;
        float v166 = simd_sum(v165);
        uint v168 = v152 + 2u;
        uint v169 = v168 * 2u;
        uint v170 = v169 + v4;
        if (v151) {
            s0[v170] = v166;
        }
        float v171 = m3;
        float v172 = simd_sum(v171);
        uint v174 = v152 + 3u;
        uint v175 = v174 * 2u;
        uint v176 = v175 + v4;
        if (v151) {
            s0[v176] = v172;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v177 = v4 == 0u;
        bool v178 = v151 && v177;
        uint v180 = v13 + 0u;
        bool v181 = v180 < 1024u;
        bool v182 = v178 && v181;
        if (v182) {
            uint v184 = v152 + 0u;
            uint v185 = v184 * 2u;
            float v186 = s0[v185];
            uint v188 = v185 + 1u;
            float v189 = s0[v188];
            float v190 = v186 + v189;
            float v191 = float(half(v190));
            uint v192 = uint(b2[v180]);
            float v193 = float(as_type<half>(ushort(v192)));
            float v194 = v193 + v191;
            uint v195 = uint(as_type<ushort>(half(v194)));
            b2[v180] = ushort(v195);
        }
        uint v197 = v13 + 1u;
        bool v198 = v197 < 1024u;
        bool v199 = v178 && v198;
        if (v199) {
            uint v201 = v152 + 1u;
            uint v202 = v201 * 2u;
            float v203 = s0[v202];
            uint v205 = v202 + 1u;
            float v206 = s0[v205];
            float v207 = v203 + v206;
            float v208 = float(half(v207));
            uint v209 = uint(b2[v197]);
            float v210 = float(as_type<half>(ushort(v209)));
            float v211 = v210 + v208;
            uint v212 = uint(as_type<ushort>(half(v211)));
            b2[v197] = ushort(v212);
        }
        uint v214 = v13 + 2u;
        bool v215 = v214 < 1024u;
        bool v216 = v178 && v215;
        if (v216) {
            uint v218 = v152 + 2u;
            uint v219 = v218 * 2u;
            float v220 = s0[v219];
            uint v222 = v219 + 1u;
            float v223 = s0[v222];
            float v224 = v220 + v223;
            float v225 = float(half(v224));
            uint v226 = uint(b2[v214]);
            float v227 = float(as_type<half>(ushort(v226)));
            float v228 = v227 + v225;
            uint v229 = uint(as_type<ushort>(half(v228)));
            b2[v214] = ushort(v229);
        }
        uint v231 = v13 + 3u;
        bool v232 = v231 < 1024u;
        bool v233 = v178 && v232;
        if (v233) {
            uint v235 = v152 + 3u;
            uint v236 = v235 * 2u;
            float v237 = s0[v236];
            uint v239 = v236 + 1u;
            float v240 = s0[v239];
            float v241 = v237 + v240;
            float v242 = float(half(v241));
            uint v243 = uint(b2[v231]);
            float v244 = float(as_type<half>(ushort(v243)));
            float v245 = v244 + v242;
            uint v246 = uint(as_type<ushort>(half(v245)));
            b2[v231] = ushort(v246);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
