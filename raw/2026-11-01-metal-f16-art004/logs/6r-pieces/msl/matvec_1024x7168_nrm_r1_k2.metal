#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x7168_nrm_r1_k2(
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
    threadgroup float s0[4];
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
    uint v153 = v150 * 1u;
    uint v158 = v153 + 0u;
    bool v159 = v158 < 7168u;
    uint v160 = v159 ? v158 : 7167u;
    uint v161 = v160 * 128u;
    uint v164 = v147 * 32u;
    uint v165 = v164 + lane;
    float m0 = as_type<float>(0x00000000u);
    for (uint v168 = v165; v168 < 128u; v168 += 64u) {
        uint4 v170 = s2[v168];
        uint v171 = v170.x;
        float v172 = float(as_type<half>(ushort(v171)));
        float v173 = float(as_type<half>(ushort(v171 >> 16)));
        uint v174 = v170.y;
        float v175 = float(as_type<half>(ushort(v174)));
        float v176 = float(as_type<half>(ushort(v174 >> 16)));
        uint v177 = v170.z;
        float v178 = float(as_type<half>(ushort(v177)));
        float v179 = float(as_type<half>(ushort(v177 >> 16)));
        uint v180 = v170.w;
        float v181 = float(as_type<half>(ushort(v180)));
        float v182 = float(as_type<half>(ushort(v180 >> 16)));
        uint v183 = v161 + v168;
        uint4 v184 = b0[v183];
        float v185 = m0;
        uint v186 = v184.x;
        float v187 = float(as_type<half>(ushort(v186)));
        float v188 = fma(v187, v172, v185);
        float v189 = float(as_type<half>(ushort(v186 >> 16)));
        float v190 = fma(v189, v173, v188);
        uint v191 = v184.y;
        float v192 = float(as_type<half>(ushort(v191)));
        float v193 = fma(v192, v175, v190);
        float v194 = float(as_type<half>(ushort(v191 >> 16)));
        float v195 = fma(v194, v176, v193);
        uint v196 = v184.z;
        float v197 = float(as_type<half>(ushort(v196)));
        float v198 = fma(v197, v178, v195);
        float v199 = float(as_type<half>(ushort(v196 >> 16)));
        float v200 = fma(v199, v179, v198);
        uint v201 = v184.w;
        float v202 = float(as_type<half>(ushort(v201)));
        float v203 = fma(v202, v181, v200);
        float v204 = float(as_type<half>(ushort(v201 >> 16)));
        float v205 = fma(v204, v182, v203);
        m0 = v205;
    }
    bool v207 = lane == 0u;
    uint v208 = v146 * 1u;
    float v209 = m0;
    float v210 = simd_sum(v209);
    uint v212 = v208 + 0u;
    uint v213 = v212 * 2u;
    uint v214 = v213 + v147;
    if (v207) {
        s0[v214] = v210;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v215 = v147 == 0u;
    bool v216 = v207 && v215;
    uint v218 = v153 + 0u;
    bool v219 = v218 < 7168u;
    bool v220 = v216 && v219;
    if (v220) {
        uint v222 = v208 + 0u;
        uint v223 = v222 * 2u;
        float v224 = s0[v223];
        uint v226 = v223 + 1u;
        float v227 = s0[v226];
        float v228 = v224 + v227;
        uint v229 = uint(as_type<ushort>(half(v228)));
        b2[v218] = ushort(v229);
    }
}
