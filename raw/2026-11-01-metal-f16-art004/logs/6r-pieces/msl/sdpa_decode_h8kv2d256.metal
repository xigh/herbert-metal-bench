#include <metal_stdlib>
using namespace metal;

kernel void sdpa_decode_h8kv2d256(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    device float *b4 [[buffer(4)]],
    device ushort *b5 [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[9];
    threadgroup float s1[256];
    uint v3 = b0[1u];
    uint v5 = v3 + 1u;
    uint v7 = tgid.x / 4u;
    uint v10 = tgid.x * 512u;
    for (uint v12 = lid; v12 < 256u; v12 += 256u) {
        uint v13 = v10 + v12;
        uint v14 = uint(b1[v13]);
        float v15 = float(as_type<half>(ushort(v14)));
        s1[v12] = v15;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v18 = v7 * 524288u;
    uint v19 = tgid.x * 2048u;
    float m0 = as_type<float>(0xff7fffffu);
    for (uint v22 = lid; v22 < v5; v22 += 256u) {
        uint v23 = v22 * 256u;
        uint v24 = v18 + v23;
        float m1 = as_type<float>(0x00000000u);
        for (uint v28 = 0u; v28 < 256u; v28 += 1u) {
            uint v29 = v24 + v28;
            uint v30 = uint(b2[v29]);
            float v31 = float(as_type<half>(ushort(v30)));
            float v32 = s1[v28];
            float v33 = m1;
            float v34 = fma(v32, v31, v33);
            m1 = v34;
        }
        float v35 = m1;
        float v36 = v35 * as_type<float>(0x3d800000u);
        uint v37 = v19 + v22;
        b4[v37] = v36;
        float v38 = m0;
        float v39 = fmax(v38, v36);
        m0 = v39;
    }
    float v40 = m0;
    float v41 = simd_max(v40);
    bool v45 = lane == 0u;
    if (v45) {
        s0[sgid] = v41;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v47 = lid == 0u;
    if (v47) {
        float v49 = s0[0u];
        float m2 = v49;
        for (uint v52 = 1u; v52 < 8u; v52 += 1u) {
            float v53 = m2;
            float v54 = s0[v52];
            float v55 = fmax(v53, v54);
            m2 = v55;
        }
        float v56 = m2;
        s0[8u] = v56;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v59 = s0[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float m3 = as_type<float>(0x00000000u);
    for (uint v61 = lid; v61 < v5; v61 += 256u) {
        uint v62 = v19 + v61;
        float v63 = b4[v62];
        float v64 = v63 - v59;
        float v65 = exp(v64);
        b4[v62] = v65;
        float v66 = m3;
        float v67 = v66 + v65;
        m3 = v67;
    }
    float v68 = m3;
    float v69 = simd_sum(v68);
    bool v73 = lane == 0u;
    if (v73) {
        s0[sgid] = v69;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v75 = lid == 0u;
    if (v75) {
        float v77 = s0[0u];
        float m4 = v77;
        for (uint v80 = 1u; v80 < 8u; v80 += 1u) {
            float v81 = m4;
            float v82 = s0[v80];
            float v83 = v81 + v82;
            m4 = v83;
        }
        float v84 = m4;
        s0[8u] = v84;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v87 = s0[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v89 = as_type<float>(0x3f800000u) / v87;
    for (uint v90 = lid; v90 < 256u; v90 += 256u) {
        float m5 = as_type<float>(0x00000000u);
        for (uint v94 = 0u; v94 < v5; v94 += 1u) {
            uint v95 = v19 + v94;
            float v96 = b4[v95];
            float v97 = v96 * v89;
            uint v98 = v94 * 256u;
            uint v99 = v18 + v98;
            uint v100 = v99 + v90;
            uint v101 = uint(b3[v100]);
            float v102 = float(as_type<half>(ushort(v101)));
            float v103 = m5;
            float v104 = fma(v97, v102, v103);
            m5 = v104;
        }
        float v105 = m5;
        float v106 = float(half(v105));
        uint v107 = v10 + 256u;
        uint v108 = v107 + v90;
        uint v109 = uint(b1[v108]);
        float v110 = float(as_type<half>(ushort(v109)));
        float v111 = -v110;
        float v112 = exp(v111);
        float v114 = as_type<float>(0x3f800000u) + v112;
        float v115 = as_type<float>(0x3f800000u) / v114;
        float v116 = v106 * v115;
        uint v117 = uint(as_type<ushort>(half(v116)));
        uint v118 = tgid.x * 256u;
        uint v119 = v118 + v90;
        b5[v119] = ushort(v117);
    }
}
