#include <metal_stdlib>
using namespace metal;

kernel void qk_rope_append_rows_h8kv2d256(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    device ushort *b4 [[buffer(4)]],
    device ushort *b5 [[buffer(5)]],
    device ushort *b6 [[buffer(6)]],
    device ushort *b7 [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[9];
    threadgroup float s1[256];
    uint v2 = tgid.x / 12u;
    uint v3 = tgid.x % 12u;
    uint v6 = b0[1u];
    uint v7 = v6 + v2;
    uint v9 = v2 * 5120u;
    bool v11 = v3 < 8u;
    bool v13 = v3 < 10u;
    if (v13) {
        uint v18 = v3 - 8u;
        uint v19 = v3 * 512u;
        uint v20 = v18 * 256u;
        uint v21 = 4096u + v20;
        uint v22 = v11 ? v19 : v21;
        uint v23 = v9 + v22;
        uint v24 = v23 + lid;
        uint v25 = uint(b1[v24]);
        float v26 = float(as_type<half>(ushort(v25)));
        float v27 = v26 * v26;
        float v28 = simd_sum(v27);
        bool v32 = lane == 0u;
        if (v32) {
            s0[sgid] = v28;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v34 = lid == 0u;
        if (v34) {
            float v36 = s0[0u];
            float m0 = v36;
            for (uint v39 = 1u; v39 < 8u; v39 += 1u) {
                float v40 = m0;
                float v41 = s0[v39];
                float v42 = v40 + v41;
                m0 = v42;
            }
            float v43 = m0;
            s0[8u] = v43;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v46 = s0[8u];
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v49 = v46 * as_type<float>(0x3b800000u);
        float v50 = v49 + as_type<float>(0x358637bdu);
        float v51 = sqrt(v50);
        float v53 = as_type<float>(0x3f800000u) / v51;
        uint v54 = uint(b2[lid]);
        uint v55 = uint(b3[lid]);
        uint v56 = v11 ? v54 : v55;
        float v57 = float(as_type<half>(ushort(v56)));
        float v58 = v57 + as_type<float>(0x3f800000u);
        float v59 = v26 * v53;
        float v60 = v59 * v58;
        float v61 = float(half(v60));
        s1[lid] = v61;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v64 = lid < 64u;
        float m1 = v61;
        if (v64) {
            bool v65 = lid < 32u;
            uint v66 = lid - 32u;
            uint v67 = v65 ? lid : v66;
            uint v68 = v7 * 32u;
            uint v69 = v68 + v67;
            uint v70 = uint(b4[v69]);
            uint v71 = uint(b5[v69]);
            float v72 = float(as_type<half>(ushort(v70)));
            float v73 = float(as_type<half>(ushort(v71)));
            float v74 = s1[v67];
            uint v75 = v67 + 32u;
            float v76 = s1[v75];
            float v77 = v74 * v72;
            float v78 = v76 * v73;
            float v79 = v77 - v78;
            float v80 = v74 * v73;
            float v81 = v76 * v72;
            float v82 = v80 + v81;
            float v83 = v65 ? v79 : v82;
            m1 = v83;
        }
        float v84 = m1;
        uint v85 = uint(as_type<ushort>(half(v84)));
        if (v11) {
            b1[v24] = ushort(v85);
        } else {
            uint v86 = v18 * 524288u;
            uint v87 = v7 * 256u;
            uint v88 = v86 + v87;
            uint v89 = v88 + lid;
            b6[v89] = ushort(v85);
        }
    } else {
        uint v90 = v3 - 10u;
        uint v92 = v90 * 256u;
        uint v93 = 4608u + v92;
        uint v94 = v9 + v93;
        uint v95 = v94 + lid;
        uint v96 = uint(b1[v95]);
        uint v97 = v90 * 524288u;
        uint v98 = v7 * 256u;
        uint v99 = v97 + v98;
        uint v100 = v99 + lid;
        b7[v100] = ushort(v96);
    }
}
