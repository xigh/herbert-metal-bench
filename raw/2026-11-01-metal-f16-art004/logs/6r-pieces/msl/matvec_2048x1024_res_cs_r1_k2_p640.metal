#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_cs_r1_k2_p640(
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
    threadgroup float s0[4];
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
    uint v41 = sgid / 2u;
    uint v42 = sgid % 2u;
    for (uint v45 = tgid.x; v45 < 512u; v45 += 640u) {
        uint v47 = v45 * 2u;
        uint v48 = v47 + v41;
        uint v51 = v48 * 1u;
        uint v56 = v51 + 0u;
        bool v57 = v56 < 1024u;
        uint v58 = v57 ? v56 : 1023u;
        uint v59 = v58 * 256u;
        uint v62 = v42 * 32u;
        uint v63 = v62 + lane;
        float m0 = as_type<float>(0x00000000u);
        for (uint v66 = v63; v66 < 256u; v66 += 64u) {
            uint4 v68 = b1[v66];
            uint v69 = v68.x;
            float v70 = float(as_type<half>(ushort(v69)));
            float v71 = float(as_type<half>(ushort(v69 >> 16)));
            uint v72 = v68.y;
            float v73 = float(as_type<half>(ushort(v72)));
            float v74 = float(as_type<half>(ushort(v72 >> 16)));
            uint v75 = v68.z;
            float v76 = float(as_type<half>(ushort(v75)));
            float v77 = float(as_type<half>(ushort(v75 >> 16)));
            uint v78 = v68.w;
            float v79 = float(as_type<half>(ushort(v78)));
            float v80 = float(as_type<half>(ushort(v78 >> 16)));
            uint v81 = v59 + v66;
            uint4 v82 = b0[v81];
            float v83 = m0;
            uint v84 = v82.x;
            float v85 = float(as_type<half>(ushort(v84)));
            float v86 = fma(v85, v70, v83);
            float v87 = float(as_type<half>(ushort(v84 >> 16)));
            float v88 = fma(v87, v71, v86);
            uint v89 = v82.y;
            float v90 = float(as_type<half>(ushort(v89)));
            float v91 = fma(v90, v73, v88);
            float v92 = float(as_type<half>(ushort(v89 >> 16)));
            float v93 = fma(v92, v74, v91);
            uint v94 = v82.z;
            float v95 = float(as_type<half>(ushort(v94)));
            float v96 = fma(v95, v76, v93);
            float v97 = float(as_type<half>(ushort(v94 >> 16)));
            float v98 = fma(v97, v77, v96);
            uint v99 = v82.w;
            float v100 = float(as_type<half>(ushort(v99)));
            float v101 = fma(v100, v79, v98);
            float v102 = float(as_type<half>(ushort(v99 >> 16)));
            float v103 = fma(v102, v80, v101);
            m0 = v103;
        }
        bool v105 = lane == 0u;
        uint v106 = v41 * 1u;
        float v107 = m0;
        float v108 = simd_sum(v107);
        uint v110 = v106 + 0u;
        uint v111 = v110 * 2u;
        uint v112 = v111 + v42;
        if (v105) {
            s0[v112] = v108;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v113 = v42 == 0u;
        bool v114 = v105 && v113;
        uint v116 = v51 + 0u;
        bool v117 = v116 < 1024u;
        bool v118 = v114 && v117;
        if (v118) {
            uint v120 = v106 + 0u;
            uint v121 = v120 * 2u;
            float v122 = s0[v121];
            uint v124 = v121 + 1u;
            float v125 = s0[v124];
            float v126 = v122 + v125;
            float v127 = float(half(v126));
            uint v128 = uint(b2[v116]);
            float v129 = float(as_type<half>(ushort(v128)));
            float v130 = v129 + v127;
            uint v131 = uint(as_type<ushort>(half(v130)));
            b2[v116] = ushort(v131);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
