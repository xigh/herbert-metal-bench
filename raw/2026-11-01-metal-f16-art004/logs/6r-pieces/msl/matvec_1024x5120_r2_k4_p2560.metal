#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x5120_r2_k4_p2560(
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
    threadgroup float s0[8];
    uint v3 = sgid / 4u;
    uint v4 = sgid % 4u;
    for (uint v7 = tgid.x; v7 < 2560u; v7 += 2560u) {
        uint v9 = v7 * 1u;
        uint v10 = v9 + v3;
        uint v13 = v10 * 2u;
        uint v18 = v13 + 0u;
        bool v19 = v18 < 5120u;
        uint v20 = v19 ? v18 : 5119u;
        uint v21 = v20 * 128u;
        uint v23 = v13 + 1u;
        bool v24 = v23 < 5120u;
        uint v25 = v24 ? v23 : 5119u;
        uint v26 = v25 * 128u;
        uint v29 = v4 * 32u;
        uint v30 = v29 + lane;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        for (uint v33 = v30; v33 < 128u; v33 += 128u) {
            uint4 v35 = b1[v33];
            uint v36 = v35.x;
            float v37 = float(as_type<half>(ushort(v36)));
            float v38 = float(as_type<half>(ushort(v36 >> 16)));
            uint v39 = v35.y;
            float v40 = float(as_type<half>(ushort(v39)));
            float v41 = float(as_type<half>(ushort(v39 >> 16)));
            uint v42 = v35.z;
            float v43 = float(as_type<half>(ushort(v42)));
            float v44 = float(as_type<half>(ushort(v42 >> 16)));
            uint v45 = v35.w;
            float v46 = float(as_type<half>(ushort(v45)));
            float v47 = float(as_type<half>(ushort(v45 >> 16)));
            uint v48 = v21 + v33;
            uint4 v49 = b0[v48];
            float v50 = m0;
            uint v51 = v49.x;
            float v52 = float(as_type<half>(ushort(v51)));
            float v53 = fma(v52, v37, v50);
            float v54 = float(as_type<half>(ushort(v51 >> 16)));
            float v55 = fma(v54, v38, v53);
            uint v56 = v49.y;
            float v57 = float(as_type<half>(ushort(v56)));
            float v58 = fma(v57, v40, v55);
            float v59 = float(as_type<half>(ushort(v56 >> 16)));
            float v60 = fma(v59, v41, v58);
            uint v61 = v49.z;
            float v62 = float(as_type<half>(ushort(v61)));
            float v63 = fma(v62, v43, v60);
            float v64 = float(as_type<half>(ushort(v61 >> 16)));
            float v65 = fma(v64, v44, v63);
            uint v66 = v49.w;
            float v67 = float(as_type<half>(ushort(v66)));
            float v68 = fma(v67, v46, v65);
            float v69 = float(as_type<half>(ushort(v66 >> 16)));
            float v70 = fma(v69, v47, v68);
            m0 = v70;
            uint v71 = v26 + v33;
            uint4 v72 = b0[v71];
            float v73 = m1;
            uint v74 = v72.x;
            float v75 = float(as_type<half>(ushort(v74)));
            float v76 = fma(v75, v37, v73);
            float v77 = float(as_type<half>(ushort(v74 >> 16)));
            float v78 = fma(v77, v38, v76);
            uint v79 = v72.y;
            float v80 = float(as_type<half>(ushort(v79)));
            float v81 = fma(v80, v40, v78);
            float v82 = float(as_type<half>(ushort(v79 >> 16)));
            float v83 = fma(v82, v41, v81);
            uint v84 = v72.z;
            float v85 = float(as_type<half>(ushort(v84)));
            float v86 = fma(v85, v43, v83);
            float v87 = float(as_type<half>(ushort(v84 >> 16)));
            float v88 = fma(v87, v44, v86);
            uint v89 = v72.w;
            float v90 = float(as_type<half>(ushort(v89)));
            float v91 = fma(v90, v46, v88);
            float v92 = float(as_type<half>(ushort(v89 >> 16)));
            float v93 = fma(v92, v47, v91);
            m1 = v93;
        }
        bool v95 = lane == 0u;
        uint v96 = v3 * 2u;
        float v97 = m0;
        float v98 = simd_sum(v97);
        uint v100 = v96 + 0u;
        uint v101 = v100 * 4u;
        uint v102 = v101 + v4;
        if (v95) {
            s0[v102] = v98;
        }
        float v103 = m1;
        float v104 = simd_sum(v103);
        uint v106 = v96 + 1u;
        uint v107 = v106 * 4u;
        uint v108 = v107 + v4;
        if (v95) {
            s0[v108] = v104;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v109 = v4 == 0u;
        bool v110 = v95 && v109;
        uint v112 = v13 + 0u;
        bool v113 = v112 < 5120u;
        bool v114 = v110 && v113;
        if (v114) {
            uint v116 = v96 + 0u;
            uint v117 = v116 * 4u;
            float v118 = s0[v117];
            uint v120 = v117 + 1u;
            float v121 = s0[v120];
            float v122 = v118 + v121;
            uint v124 = v117 + 2u;
            float v125 = s0[v124];
            float v126 = v122 + v125;
            uint v128 = v117 + 3u;
            float v129 = s0[v128];
            float v130 = v126 + v129;
            uint v131 = uint(as_type<ushort>(half(v130)));
            b2[v112] = ushort(v131);
        }
        uint v133 = v13 + 1u;
        bool v134 = v133 < 5120u;
        bool v135 = v110 && v134;
        if (v135) {
            uint v137 = v96 + 1u;
            uint v138 = v137 * 4u;
            float v139 = s0[v138];
            uint v141 = v138 + 1u;
            float v142 = s0[v141];
            float v143 = v139 + v142;
            uint v145 = v138 + 2u;
            float v146 = s0[v145];
            float v147 = v143 + v146;
            uint v149 = v138 + 3u;
            float v150 = s0[v149];
            float v151 = v147 + v150;
            uint v152 = uint(as_type<ushort>(half(v151)));
            b2[v133] = ushort(v152);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
