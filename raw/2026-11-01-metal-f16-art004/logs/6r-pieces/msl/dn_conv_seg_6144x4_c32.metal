#include <metal_stdlib>
using namespace metal;

kernel void dn_conv_seg_6144x4_c32(
    device uint *b0 [[buffer(0)]],
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
    uint v1 = b0[3u];
    uint v4 = lid % 32u;
    uint v5 = lid / 32u;
    uint v7 = tgid.x * 32u;
    uint v8 = v7 + v4;
    uint v10 = v8 * 4u;
    uint v16 = v1 + 7u;
    uint v17 = v16 / 8u;
    uint v18 = v5 * v17;
    uint v19 = v18 + v17;
    bool v20 = v1 < v19;
    uint v21 = v20 ? v1 : v19;
    bool v22 = v18 < v1;
    uint v24 = v10 + 0u;
    uint v25 = uint(b3[v24]);
    float v26 = float(as_type<half>(ushort(v25)));
    bool v28 = v18 < 4u;
    bool v29 = !v28;
    float m0 = as_type<float>(0x00000000u);
    if (v22) {
        if (v29) {
            uint v31 = v18 - 4u;
            uint v32 = v31 * 8224u;
            uint v33 = v32 + v8;
            uint v34 = uint(b1[v33]);
            float v35 = float(as_type<half>(ushort(v34)));
            m0 = v35;
        } else {
            uint v36 = v10 + 4u;
            uint v37 = v36 + v18;
            uint v38 = v37 - 4u;
            uint v39 = uint(b2[v38]);
            float v40 = float(as_type<half>(ushort(v39)));
            m0 = v40;
        }
    }
    float v41 = m0;
    float m1 = v41;
    uint v43 = v10 + 1u;
    uint v44 = uint(b3[v43]);
    float v45 = float(as_type<half>(ushort(v44)));
    bool v47 = v18 < 3u;
    bool v48 = !v47;
    float m2 = as_type<float>(0x00000000u);
    if (v22) {
        if (v48) {
            uint v50 = v18 - 3u;
            uint v51 = v50 * 8224u;
            uint v52 = v51 + v8;
            uint v53 = uint(b1[v52]);
            float v54 = float(as_type<half>(ushort(v53)));
            m2 = v54;
        } else {
            uint v55 = v10 + 4u;
            uint v56 = v55 + v18;
            uint v57 = v56 - 3u;
            uint v58 = uint(b2[v57]);
            float v59 = float(as_type<half>(ushort(v58)));
            m2 = v59;
        }
    }
    float v60 = m2;
    float m3 = v60;
    uint v62 = v10 + 2u;
    uint v63 = uint(b3[v62]);
    float v64 = float(as_type<half>(ushort(v63)));
    bool v66 = v18 < 2u;
    bool v67 = !v66;
    float m4 = as_type<float>(0x00000000u);
    if (v22) {
        if (v67) {
            uint v69 = v18 - 2u;
            uint v70 = v69 * 8224u;
            uint v71 = v70 + v8;
            uint v72 = uint(b1[v71]);
            float v73 = float(as_type<half>(ushort(v72)));
            m4 = v73;
        } else {
            uint v74 = v10 + 4u;
            uint v75 = v74 + v18;
            uint v76 = v75 - 2u;
            uint v77 = uint(b2[v76]);
            float v78 = float(as_type<half>(ushort(v77)));
            m4 = v78;
        }
    }
    float v79 = m4;
    float m5 = v79;
    uint v81 = v10 + 3u;
    uint v82 = uint(b3[v81]);
    float v83 = float(as_type<half>(ushort(v82)));
    bool v85 = v18 < 1u;
    bool v86 = !v85;
    float m6 = as_type<float>(0x00000000u);
    if (v22) {
        if (v86) {
            uint v88 = v18 - 1u;
            uint v89 = v88 * 8224u;
            uint v90 = v89 + v8;
            uint v91 = uint(b1[v90]);
            float v92 = float(as_type<half>(ushort(v91)));
            m6 = v92;
        } else {
            uint v93 = v10 + 4u;
            uint v94 = v93 + v18;
            uint v95 = v94 - 1u;
            uint v96 = uint(b2[v95]);
            float v97 = float(as_type<half>(ushort(v96)));
            m6 = v97;
        }
    }
    float v98 = m6;
    float m7 = v98;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint v99 = v18; v99 < v21; v99 += 1u) {
        uint v100 = v99 * 8224u;
        uint v101 = v100 + v8;
        uint v102 = uint(b1[v101]);
        float v103 = float(as_type<half>(ushort(v102)));
        float v104 = m3;
        m1 = v104;
        float v105 = m5;
        m3 = v105;
        float v106 = m7;
        m5 = v106;
        m7 = v103;
        float v108 = m1;
        float v109 = fma(v108, v26, as_type<float>(0x00000000u));
        float v110 = m3;
        float v111 = fma(v110, v45, v109);
        float v112 = m5;
        float v113 = fma(v112, v64, v111);
        float v114 = m7;
        float v115 = fma(v114, v83, v113);
        float v116 = -v115;
        float v117 = exp(v116);
        float v119 = as_type<float>(0x3f800000u) + v117;
        float v120 = as_type<float>(0x3f800000u) / v119;
        float v121 = v115 * v120;
        uint v122 = uint(as_type<ushort>(half(v121)));
        b1[v101] = ushort(v122);
    }
    uint v123 = v1 - 1u;
    bool v124 = v123 < v18;
    bool v125 = !v124;
    bool v126 = v123 < v21;
    bool v127 = v125 && v126;
    if (v127) {
        uint v129 = v10 + 0u;
        float v130 = m1;
        uint v131 = uint(as_type<ushort>(half(v130)));
        b2[v129] = ushort(v131);
        uint v133 = v10 + 1u;
        float v134 = m3;
        uint v135 = uint(as_type<ushort>(half(v134)));
        b2[v133] = ushort(v135);
        uint v137 = v10 + 2u;
        float v138 = m5;
        uint v139 = uint(as_type<ushort>(half(v138)));
        b2[v137] = ushort(v139);
        uint v141 = v10 + 3u;
        float v142 = m7;
        uint v143 = uint(as_type<ushort>(half(v142)));
        b2[v141] = ushort(v143);
    }
}
