#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_cs_r2_k4(
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
    uint v44 = tgid.x * 1u;
    uint v45 = v44 + v41;
    uint v48 = v45 * 2u;
    uint v53 = v48 + 0u;
    bool v54 = v53 < 1024u;
    uint v55 = v54 ? v53 : 1023u;
    uint v56 = v55 * 256u;
    uint v58 = v48 + 1u;
    bool v59 = v58 < 1024u;
    uint v60 = v59 ? v58 : 1023u;
    uint v61 = v60 * 256u;
    uint v64 = v42 * 32u;
    uint v65 = v64 + lane;
    float m0 = as_type<float>(0x00000000u);
    float m1 = as_type<float>(0x00000000u);
    for (uint v68 = v65; v68 < 256u; v68 += 128u) {
        uint4 v70 = b1[v68];
        uint v71 = v70.x;
        float v72 = float(as_type<half>(ushort(v71)));
        float v73 = float(as_type<half>(ushort(v71 >> 16)));
        uint v74 = v70.y;
        float v75 = float(as_type<half>(ushort(v74)));
        float v76 = float(as_type<half>(ushort(v74 >> 16)));
        uint v77 = v70.z;
        float v78 = float(as_type<half>(ushort(v77)));
        float v79 = float(as_type<half>(ushort(v77 >> 16)));
        uint v80 = v70.w;
        float v81 = float(as_type<half>(ushort(v80)));
        float v82 = float(as_type<half>(ushort(v80 >> 16)));
        uint v83 = v56 + v68;
        uint4 v84 = b0[v83];
        float v85 = m0;
        uint v86 = v84.x;
        float v87 = float(as_type<half>(ushort(v86)));
        float v88 = fma(v87, v72, v85);
        float v89 = float(as_type<half>(ushort(v86 >> 16)));
        float v90 = fma(v89, v73, v88);
        uint v91 = v84.y;
        float v92 = float(as_type<half>(ushort(v91)));
        float v93 = fma(v92, v75, v90);
        float v94 = float(as_type<half>(ushort(v91 >> 16)));
        float v95 = fma(v94, v76, v93);
        uint v96 = v84.z;
        float v97 = float(as_type<half>(ushort(v96)));
        float v98 = fma(v97, v78, v95);
        float v99 = float(as_type<half>(ushort(v96 >> 16)));
        float v100 = fma(v99, v79, v98);
        uint v101 = v84.w;
        float v102 = float(as_type<half>(ushort(v101)));
        float v103 = fma(v102, v81, v100);
        float v104 = float(as_type<half>(ushort(v101 >> 16)));
        float v105 = fma(v104, v82, v103);
        m0 = v105;
        uint v106 = v61 + v68;
        uint4 v107 = b0[v106];
        float v108 = m1;
        uint v109 = v107.x;
        float v110 = float(as_type<half>(ushort(v109)));
        float v111 = fma(v110, v72, v108);
        float v112 = float(as_type<half>(ushort(v109 >> 16)));
        float v113 = fma(v112, v73, v111);
        uint v114 = v107.y;
        float v115 = float(as_type<half>(ushort(v114)));
        float v116 = fma(v115, v75, v113);
        float v117 = float(as_type<half>(ushort(v114 >> 16)));
        float v118 = fma(v117, v76, v116);
        uint v119 = v107.z;
        float v120 = float(as_type<half>(ushort(v119)));
        float v121 = fma(v120, v78, v118);
        float v122 = float(as_type<half>(ushort(v119 >> 16)));
        float v123 = fma(v122, v79, v121);
        uint v124 = v107.w;
        float v125 = float(as_type<half>(ushort(v124)));
        float v126 = fma(v125, v81, v123);
        float v127 = float(as_type<half>(ushort(v124 >> 16)));
        float v128 = fma(v127, v82, v126);
        m1 = v128;
    }
    bool v130 = lane == 0u;
    uint v131 = v41 * 2u;
    float v132 = m0;
    float v133 = simd_sum(v132);
    uint v135 = v131 + 0u;
    uint v136 = v135 * 4u;
    uint v137 = v136 + v42;
    if (v130) {
        s0[v137] = v133;
    }
    float v138 = m1;
    float v139 = simd_sum(v138);
    uint v141 = v131 + 1u;
    uint v142 = v141 * 4u;
    uint v143 = v142 + v42;
    if (v130) {
        s0[v143] = v139;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v144 = v42 == 0u;
    bool v145 = v130 && v144;
    uint v147 = v48 + 0u;
    bool v148 = v147 < 1024u;
    bool v149 = v145 && v148;
    if (v149) {
        uint v151 = v131 + 0u;
        uint v152 = v151 * 4u;
        float v153 = s0[v152];
        uint v155 = v152 + 1u;
        float v156 = s0[v155];
        float v157 = v153 + v156;
        uint v159 = v152 + 2u;
        float v160 = s0[v159];
        float v161 = v157 + v160;
        uint v163 = v152 + 3u;
        float v164 = s0[v163];
        float v165 = v161 + v164;
        float v166 = float(half(v165));
        uint v167 = uint(b2[v147]);
        float v168 = float(as_type<half>(ushort(v167)));
        float v169 = v168 + v166;
        uint v170 = uint(as_type<ushort>(half(v169)));
        b2[v147] = ushort(v170);
    }
    uint v172 = v48 + 1u;
    bool v173 = v172 < 1024u;
    bool v174 = v145 && v173;
    if (v174) {
        uint v176 = v131 + 1u;
        uint v177 = v176 * 4u;
        float v178 = s0[v177];
        uint v180 = v177 + 1u;
        float v181 = s0[v180];
        float v182 = v178 + v181;
        uint v184 = v177 + 2u;
        float v185 = s0[v184];
        float v186 = v182 + v185;
        uint v188 = v177 + 3u;
        float v189 = s0[v188];
        float v190 = v186 + v189;
        float v191 = float(half(v190));
        uint v192 = uint(b2[v172]);
        float v193 = float(as_type<half>(ushort(v192)));
        float v194 = v193 + v191;
        uint v195 = uint(as_type<ushort>(half(v194)));
        b2[v172] = ushort(v195);
    }
}
