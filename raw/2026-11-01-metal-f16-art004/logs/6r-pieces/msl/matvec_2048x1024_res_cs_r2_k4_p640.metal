#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_cs_r2_k4_p640(
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
    threadgroup float s0[8];
    bool v2 = tgid.x < 512u;
    bool v3 = !v2;
    if (v3) {
        uint v4 = tgid.x - 512u;
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
    uint v41 = sgid / 4u;
    uint v42 = sgid % 4u;
    for (uint v45 = tgid.x; v45 < 512u; v45 += 640u) {
        uint v47 = v45 * 1u;
        uint v48 = v47 + v41;
        uint v51 = v48 * 2u;
        uint v56 = v51 + 0u;
        bool v57 = v56 < 1024u;
        uint v58 = v57 ? v56 : 1023u;
        uint v59 = v58 * 256u;
        uint v61 = v51 + 1u;
        bool v62 = v61 < 1024u;
        uint v63 = v62 ? v61 : 1023u;
        uint v64 = v63 * 256u;
        uint v67 = v42 * 32u;
        uint v68 = v67 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        for (uint v71 = v68; v71 < 256u; v71 += 128u) {
            uint4 v73 = b1[v71];
            uint v74 = v73.x;
            float v75 = float(as_type<half>(ushort(v74)));
            float v76 = float(as_type<half>(ushort(v74 >> 16)));
            uint v77 = v73.y;
            float v78 = float(as_type<half>(ushort(v77)));
            float v79 = float(as_type<half>(ushort(v77 >> 16)));
            uint v80 = v73.z;
            float v81 = float(as_type<half>(ushort(v80)));
            float v82 = float(as_type<half>(ushort(v80 >> 16)));
            uint v83 = v73.w;
            float v84 = float(as_type<half>(ushort(v83)));
            float v85 = float(as_type<half>(ushort(v83 >> 16)));
            uint v86 = v59 + v71;
            uint4 v87 = b0[v86];
            float v88 = m0;
            uint v89 = v87.x;
            float v90 = float(as_type<half>(ushort(v89)));
            float v91 = fma(v90, v75, v88);
            float v92 = float(as_type<half>(ushort(v89 >> 16)));
            float v93 = fma(v92, v76, v91);
            uint v94 = v87.y;
            float v95 = float(as_type<half>(ushort(v94)));
            float v96 = fma(v95, v78, v93);
            float v97 = float(as_type<half>(ushort(v94 >> 16)));
            float v98 = fma(v97, v79, v96);
            uint v99 = v87.z;
            float v100 = float(as_type<half>(ushort(v99)));
            float v101 = fma(v100, v81, v98);
            float v102 = float(as_type<half>(ushort(v99 >> 16)));
            float v103 = fma(v102, v82, v101);
            uint v104 = v87.w;
            float v105 = float(as_type<half>(ushort(v104)));
            float v106 = fma(v105, v84, v103);
            float v107 = float(as_type<half>(ushort(v104 >> 16)));
            float v108 = fma(v107, v85, v106);
            m0 = v108;
            uint v109 = v64 + v71;
            uint4 v110 = b0[v109];
            float v111 = m1;
            uint v112 = v110.x;
            float v113 = float(as_type<half>(ushort(v112)));
            float v114 = fma(v113, v75, v111);
            float v115 = float(as_type<half>(ushort(v112 >> 16)));
            float v116 = fma(v115, v76, v114);
            uint v117 = v110.y;
            float v118 = float(as_type<half>(ushort(v117)));
            float v119 = fma(v118, v78, v116);
            float v120 = float(as_type<half>(ushort(v117 >> 16)));
            float v121 = fma(v120, v79, v119);
            uint v122 = v110.z;
            float v123 = float(as_type<half>(ushort(v122)));
            float v124 = fma(v123, v81, v121);
            float v125 = float(as_type<half>(ushort(v122 >> 16)));
            float v126 = fma(v125, v82, v124);
            uint v127 = v110.w;
            float v128 = float(as_type<half>(ushort(v127)));
            float v129 = fma(v128, v84, v126);
            float v130 = float(as_type<half>(ushort(v127 >> 16)));
            float v131 = fma(v130, v85, v129);
            m1 = v131;
        }
        bool v133 = lane == 0u;
        uint v134 = v41 * 2u;
        float v135 = m0;
        float v136 = simd_sum(v135);
        uint v138 = v134 + 0u;
        uint v139 = v138 * 4u;
        uint v140 = v139 + v42;
        if (v133) {
            s0[v140] = v136;
        }
        float v141 = m1;
        float v142 = simd_sum(v141);
        uint v144 = v134 + 1u;
        uint v145 = v144 * 4u;
        uint v146 = v145 + v42;
        if (v133) {
            s0[v146] = v142;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v147 = v42 == 0u;
        bool v148 = v133 && v147;
        uint v150 = v51 + 0u;
        bool v151 = v150 < 1024u;
        bool v152 = v148 && v151;
        if (v152) {
            uint v154 = v134 + 0u;
            uint v155 = v154 * 4u;
            float v156 = s0[v155];
            uint v158 = v155 + 1u;
            float v159 = s0[v158];
            float v160 = v156 + v159;
            uint v162 = v155 + 2u;
            float v163 = s0[v162];
            float v164 = v160 + v163;
            uint v166 = v155 + 3u;
            float v167 = s0[v166];
            float v168 = v164 + v167;
            float v169 = float(half(v168));
            uint v170 = uint(b2[v150]);
            float v171 = float(as_type<half>(ushort(v170)));
            float v172 = v171 + v169;
            uint v173 = uint(as_type<ushort>(half(v172)));
            b2[v150] = ushort(v173);
        }
        uint v175 = v51 + 1u;
        bool v176 = v175 < 1024u;
        bool v177 = v148 && v176;
        if (v177) {
            uint v179 = v134 + 1u;
            uint v180 = v179 * 4u;
            float v181 = s0[v180];
            uint v183 = v180 + 1u;
            float v184 = s0[v183];
            float v185 = v181 + v184;
            uint v187 = v180 + 2u;
            float v188 = s0[v187];
            float v189 = v185 + v188;
            uint v191 = v180 + 3u;
            float v192 = s0[v191];
            float v193 = v189 + v192;
            float v194 = float(half(v193));
            uint v195 = uint(b2[v175]);
            float v196 = float(as_type<half>(ushort(v195)));
            float v197 = v196 + v194;
            uint v198 = uint(as_type<ushort>(half(v197)));
            b2[v175] = ushort(v198);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
