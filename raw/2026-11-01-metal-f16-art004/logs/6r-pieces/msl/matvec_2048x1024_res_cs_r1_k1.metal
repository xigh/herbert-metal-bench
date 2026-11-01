#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_cs_r1_k1(
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
    bool v2 = tgid.x < 256u;
    bool v3 = !v2;
    if (v3) {
        uint v4 = tgid.x - 256u;
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
    uint v44 = tgid.x * 4u;
    uint v45 = v44 + v41;
    uint v48 = v45 * 1u;
    uint v53 = v48 + 0u;
    bool v54 = v53 < 1024u;
    uint v55 = v54 ? v53 : 1023u;
    uint v56 = v55 * 256u;
    uint v59 = v42 * 32u;
    uint v60 = v59 + lane;
    float m0 = as_type<float>(0x00000000u);
    for (uint v63 = v60; v63 < 256u; v63 += 32u) {
        uint4 v65 = b1[v63];
        uint v66 = v65.x;
        float v67 = float(as_type<half>(ushort(v66)));
        float v68 = float(as_type<half>(ushort(v66 >> 16)));
        uint v69 = v65.y;
        float v70 = float(as_type<half>(ushort(v69)));
        float v71 = float(as_type<half>(ushort(v69 >> 16)));
        uint v72 = v65.z;
        float v73 = float(as_type<half>(ushort(v72)));
        float v74 = float(as_type<half>(ushort(v72 >> 16)));
        uint v75 = v65.w;
        float v76 = float(as_type<half>(ushort(v75)));
        float v77 = float(as_type<half>(ushort(v75 >> 16)));
        uint v78 = v56 + v63;
        uint4 v79 = b0[v78];
        float v80 = m0;
        uint v81 = v79.x;
        float v82 = float(as_type<half>(ushort(v81)));
        float v83 = fma(v82, v67, v80);
        float v84 = float(as_type<half>(ushort(v81 >> 16)));
        float v85 = fma(v84, v68, v83);
        uint v86 = v79.y;
        float v87 = float(as_type<half>(ushort(v86)));
        float v88 = fma(v87, v70, v85);
        float v89 = float(as_type<half>(ushort(v86 >> 16)));
        float v90 = fma(v89, v71, v88);
        uint v91 = v79.z;
        float v92 = float(as_type<half>(ushort(v91)));
        float v93 = fma(v92, v73, v90);
        float v94 = float(as_type<half>(ushort(v91 >> 16)));
        float v95 = fma(v94, v74, v93);
        uint v96 = v79.w;
        float v97 = float(as_type<half>(ushort(v96)));
        float v98 = fma(v97, v76, v95);
        float v99 = float(as_type<half>(ushort(v96 >> 16)));
        float v100 = fma(v99, v77, v98);
        m0 = v100;
    }
    bool v102 = lane == 0u;
    uint v103 = v41 * 1u;
    float v104 = m0;
    float v105 = simd_sum(v104);
    uint v107 = v103 + 0u;
    uint v108 = v107 * 1u;
    uint v109 = v108 + v42;
    if (v102) {
        s0[v109] = v105;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v110 = v42 == 0u;
    bool v111 = v102 && v110;
    uint v113 = v48 + 0u;
    bool v114 = v113 < 1024u;
    bool v115 = v111 && v114;
    if (v115) {
        uint v117 = v103 + 0u;
        uint v118 = v117 * 1u;
        float v119 = s0[v118];
        float v120 = float(half(v119));
        uint v121 = uint(b2[v113]);
        float v122 = float(as_type<half>(ushort(v121)));
        float v123 = v122 + v120;
        uint v124 = uint(as_type<ushort>(half(v123)));
        b2[v113] = ushort(v124);
    }
}
