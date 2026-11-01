#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x7168_nrm_r1_k1_p1280(
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
    uint v146 = sgid / 1u;
    uint v147 = sgid % 1u;
    for (uint v150 = tgid.x; v150 < 1792u; v150 += 1280u) {
        uint v152 = v150 * 4u;
        uint v153 = v152 + v146;
        uint v156 = v153 * 1u;
        uint v161 = v156 + 0u;
        bool v162 = v161 < 7168u;
        uint v163 = v162 ? v161 : 7167u;
        uint v164 = v163 * 128u;
        uint v167 = v147 * 32u;
        uint v168 = v167 + lane;
        float m0 = as_type<float>(0x00000000u);
        for (uint v171 = v168; v171 < 128u; v171 += 32u) {
            uint4 v173 = s2[v171];
            uint v174 = v173.x;
            float v175 = float(as_type<half>(ushort(v174)));
            float v176 = float(as_type<half>(ushort(v174 >> 16)));
            uint v177 = v173.y;
            float v178 = float(as_type<half>(ushort(v177)));
            float v179 = float(as_type<half>(ushort(v177 >> 16)));
            uint v180 = v173.z;
            float v181 = float(as_type<half>(ushort(v180)));
            float v182 = float(as_type<half>(ushort(v180 >> 16)));
            uint v183 = v173.w;
            float v184 = float(as_type<half>(ushort(v183)));
            float v185 = float(as_type<half>(ushort(v183 >> 16)));
            uint v186 = v164 + v171;
            uint4 v187 = b0[v186];
            float v188 = m0;
            uint v189 = v187.x;
            float v190 = float(as_type<half>(ushort(v189)));
            float v191 = fma(v190, v175, v188);
            float v192 = float(as_type<half>(ushort(v189 >> 16)));
            float v193 = fma(v192, v176, v191);
            uint v194 = v187.y;
            float v195 = float(as_type<half>(ushort(v194)));
            float v196 = fma(v195, v178, v193);
            float v197 = float(as_type<half>(ushort(v194 >> 16)));
            float v198 = fma(v197, v179, v196);
            uint v199 = v187.z;
            float v200 = float(as_type<half>(ushort(v199)));
            float v201 = fma(v200, v181, v198);
            float v202 = float(as_type<half>(ushort(v199 >> 16)));
            float v203 = fma(v202, v182, v201);
            uint v204 = v187.w;
            float v205 = float(as_type<half>(ushort(v204)));
            float v206 = fma(v205, v184, v203);
            float v207 = float(as_type<half>(ushort(v204 >> 16)));
            float v208 = fma(v207, v185, v206);
            m0 = v208;
        }
        bool v210 = lane == 0u;
        uint v211 = v146 * 1u;
        float v212 = m0;
        float v213 = simd_sum(v212);
        uint v215 = v211 + 0u;
        uint v216 = v215 * 1u;
        uint v217 = v216 + v147;
        if (v210) {
            s0[v217] = v213;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v218 = v147 == 0u;
        bool v219 = v210 && v218;
        uint v221 = v156 + 0u;
        bool v222 = v221 < 7168u;
        bool v223 = v219 && v222;
        if (v223) {
            uint v225 = v211 + 0u;
            uint v226 = v225 * 1u;
            float v227 = s0[v226];
            uint v228 = uint(as_type<ushort>(half(v227)));
            b2[v221] = ushort(v228);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
