#include <metal_stdlib>
using namespace metal;

kernel void qk_rope_append_h8kv2d256_pf196608(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    device ushort *b4 [[buffer(4)]],
    device ushort *b5 [[buffer(5)]],
    device ushort *b6 [[buffer(6)]],
    device ushort *b7 [[buffer(7)]],
    device uint4 *b8 [[buffer(8)]],
    device uint *b9 [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[9];
    threadgroup float s1[256];
    bool v2 = tgid.x < 12u;
    bool v3 = !v2;
    if (v3) {
        uint v4 = tgid.x - 12u;
        uint v7 = v4 * 2048u;
        uint v8 = v7 + lid;
        uint v13 = v8 + 0u;
        bool v14 = v13 < 196608u;
        uint v15 = v14 ? v13 : 196607u;
        uint4 v16 = b8[v15];
        uint v17 = v16.x;
        uint v18 = 0u ^ v17;
        uint v19 = v16.y;
        uint v20 = v18 ^ v19;
        uint v21 = v16.z;
        uint v22 = v20 ^ v21;
        uint v23 = v16.w;
        uint v24 = v22 ^ v23;
        uint v26 = v8 + 256u;
        bool v27 = v26 < 196608u;
        uint v28 = v27 ? v26 : 196607u;
        uint4 v29 = b8[v28];
        uint v30 = v29.x;
        uint v31 = v24 ^ v30;
        uint v32 = v29.y;
        uint v33 = v31 ^ v32;
        uint v34 = v29.z;
        uint v35 = v33 ^ v34;
        uint v36 = v29.w;
        uint v37 = v35 ^ v36;
        uint v39 = v8 + 512u;
        bool v40 = v39 < 196608u;
        uint v41 = v40 ? v39 : 196607u;
        uint4 v42 = b8[v41];
        uint v43 = v42.x;
        uint v44 = v37 ^ v43;
        uint v45 = v42.y;
        uint v46 = v44 ^ v45;
        uint v47 = v42.z;
        uint v48 = v46 ^ v47;
        uint v49 = v42.w;
        uint v50 = v48 ^ v49;
        uint v52 = v8 + 768u;
        bool v53 = v52 < 196608u;
        uint v54 = v53 ? v52 : 196607u;
        uint4 v55 = b8[v54];
        uint v56 = v55.x;
        uint v57 = v50 ^ v56;
        uint v58 = v55.y;
        uint v59 = v57 ^ v58;
        uint v60 = v55.z;
        uint v61 = v59 ^ v60;
        uint v62 = v55.w;
        uint v63 = v61 ^ v62;
        uint v65 = v8 + 1024u;
        bool v66 = v65 < 196608u;
        uint v67 = v66 ? v65 : 196607u;
        uint4 v68 = b8[v67];
        uint v69 = v68.x;
        uint v70 = v63 ^ v69;
        uint v71 = v68.y;
        uint v72 = v70 ^ v71;
        uint v73 = v68.z;
        uint v74 = v72 ^ v73;
        uint v75 = v68.w;
        uint v76 = v74 ^ v75;
        uint v78 = v8 + 1280u;
        bool v79 = v78 < 196608u;
        uint v80 = v79 ? v78 : 196607u;
        uint4 v81 = b8[v80];
        uint v82 = v81.x;
        uint v83 = v76 ^ v82;
        uint v84 = v81.y;
        uint v85 = v83 ^ v84;
        uint v86 = v81.z;
        uint v87 = v85 ^ v86;
        uint v88 = v81.w;
        uint v89 = v87 ^ v88;
        uint v91 = v8 + 1536u;
        bool v92 = v91 < 196608u;
        uint v93 = v92 ? v91 : 196607u;
        uint4 v94 = b8[v93];
        uint v95 = v94.x;
        uint v96 = v89 ^ v95;
        uint v97 = v94.y;
        uint v98 = v96 ^ v97;
        uint v99 = v94.z;
        uint v100 = v98 ^ v99;
        uint v101 = v94.w;
        uint v102 = v100 ^ v101;
        uint v104 = v8 + 1792u;
        bool v105 = v104 < 196608u;
        uint v106 = v105 ? v104 : 196607u;
        uint4 v107 = b8[v106];
        uint v108 = v107.x;
        uint v109 = v102 ^ v108;
        uint v110 = v107.y;
        uint v111 = v109 ^ v110;
        uint v112 = v107.z;
        uint v113 = v111 ^ v112;
        uint v114 = v107.w;
        uint v115 = v113 ^ v114;
        bool v117 = v115 == 2654435769u;
        if (v117) {
            b9[0u] = v115;
        }
        return;
    }
    uint v122 = b0[1u];
    bool v124 = tgid.x < 8u;
    bool v126 = tgid.x < 10u;
    if (v126) {
        uint v131 = tgid.x - 8u;
        uint v132 = tgid.x * 512u;
        uint v133 = v131 * 256u;
        uint v134 = 4096u + v133;
        uint v135 = v124 ? v132 : v134;
        uint v136 = v135 + lid;
        uint v137 = uint(b1[v136]);
        float v138 = float(as_type<half>(ushort(v137)));
        float v139 = v138 * v138;
        float v140 = simd_sum(v139);
        bool v144 = lane == 0u;
        if (v144) {
            s0[sgid] = v140;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v146 = lid == 0u;
        if (v146) {
            float v148 = s0[0u];
            float m0 = v148;
            for (uint v151 = 1u; v151 < 8u; v151 += 1u) {
                float v152 = m0;
                float v153 = s0[v151];
                float v154 = v152 + v153;
                m0 = v154;
            }
            float v155 = m0;
            s0[8u] = v155;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v158 = s0[8u];
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v161 = v158 * as_type<float>(0x3b800000u);
        float v162 = v161 + as_type<float>(0x358637bdu);
        float v163 = sqrt(v162);
        float v165 = as_type<float>(0x3f800000u) / v163;
        uint v166 = uint(b2[lid]);
        uint v167 = uint(b3[lid]);
        uint v168 = v124 ? v166 : v167;
        float v169 = float(as_type<half>(ushort(v168)));
        float v170 = v169 + as_type<float>(0x3f800000u);
        float v171 = v138 * v165;
        float v172 = v171 * v170;
        float v173 = float(half(v172));
        s1[lid] = v173;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v176 = lid < 64u;
        float m1 = v173;
        if (v176) {
            bool v177 = lid < 32u;
            uint v178 = lid - 32u;
            uint v179 = v177 ? lid : v178;
            uint v180 = v122 * 32u;
            uint v181 = v180 + v179;
            uint v182 = uint(b4[v181]);
            uint v183 = uint(b5[v181]);
            float v184 = float(as_type<half>(ushort(v182)));
            float v185 = float(as_type<half>(ushort(v183)));
            float v186 = s1[v179];
            uint v187 = v179 + 32u;
            float v188 = s1[v187];
            float v189 = v186 * v184;
            float v190 = v188 * v185;
            float v191 = v189 - v190;
            float v192 = v186 * v185;
            float v193 = v188 * v184;
            float v194 = v192 + v193;
            float v195 = v177 ? v191 : v194;
            m1 = v195;
        }
        float v196 = m1;
        uint v197 = uint(as_type<ushort>(half(v196)));
        if (v124) {
            b1[v136] = ushort(v197);
        } else {
            uint v198 = v131 * 524288u;
            uint v199 = v122 * 256u;
            uint v200 = v198 + v199;
            uint v201 = v200 + lid;
            b6[v201] = ushort(v197);
        }
    } else {
        uint v202 = tgid.x - 10u;
        uint v204 = v202 * 256u;
        uint v205 = 4608u + v204;
        uint v206 = v205 + lid;
        uint v207 = uint(b1[v206]);
        uint v208 = v202 * 524288u;
        uint v209 = v122 * 256u;
        uint v210 = v208 + v209;
        uint v211 = v210 + lid;
        b7[v211] = ushort(v207);
    }
}
