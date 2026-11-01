#include <metal_stdlib>
using namespace metal;

kernel void qk_rope_append_h8kv2d256(
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
    uint v3 = b0[1u];
    bool v5 = tgid.x < 8u;
    bool v7 = tgid.x < 10u;
    if (v7) {
        uint v12 = tgid.x - 8u;
        uint v13 = tgid.x * 512u;
        uint v14 = v12 * 256u;
        uint v15 = 4096u + v14;
        uint v16 = v5 ? v13 : v15;
        uint v17 = v16 + lid;
        uint v18 = uint(b1[v17]);
        float v19 = float(as_type<half>(ushort(v18)));
        float v20 = v19 * v19;
        float v21 = simd_sum(v20);
        bool v25 = lane == 0u;
        if (v25) {
            s0[sgid] = v21;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v27 = lid == 0u;
        if (v27) {
            float v29 = s0[0u];
            float m0 = v29;
            for (uint v32 = 1u; v32 < 8u; v32 += 1u) {
                float v33 = m0;
                float v34 = s0[v32];
                float v35 = v33 + v34;
                m0 = v35;
            }
            float v36 = m0;
            s0[8u] = v36;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v39 = s0[8u];
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v42 = v39 * as_type<float>(0x3b800000u);
        float v43 = v42 + as_type<float>(0x358637bdu);
        float v44 = sqrt(v43);
        float v46 = as_type<float>(0x3f800000u) / v44;
        uint v47 = uint(b2[lid]);
        uint v48 = uint(b3[lid]);
        uint v49 = v5 ? v47 : v48;
        float v50 = float(as_type<half>(ushort(v49)));
        float v51 = v50 + as_type<float>(0x3f800000u);
        float v52 = v19 * v46;
        float v53 = v52 * v51;
        float v54 = float(half(v53));
        s1[lid] = v54;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v57 = lid < 64u;
        float m1 = v54;
        if (v57) {
            bool v58 = lid < 32u;
            uint v59 = lid - 32u;
            uint v60 = v58 ? lid : v59;
            uint v61 = v3 * 32u;
            uint v62 = v61 + v60;
            uint v63 = uint(b4[v62]);
            uint v64 = uint(b5[v62]);
            float v65 = float(as_type<half>(ushort(v63)));
            float v66 = float(as_type<half>(ushort(v64)));
            float v67 = s1[v60];
            uint v68 = v60 + 32u;
            float v69 = s1[v68];
            float v70 = v67 * v65;
            float v71 = v69 * v66;
            float v72 = v70 - v71;
            float v73 = v67 * v66;
            float v74 = v69 * v65;
            float v75 = v73 + v74;
            float v76 = v58 ? v72 : v75;
            m1 = v76;
        }
        float v77 = m1;
        uint v78 = uint(as_type<ushort>(half(v77)));
        if (v5) {
            b1[v17] = ushort(v78);
        } else {
            uint v79 = v12 * 524288u;
            uint v80 = v3 * 256u;
            uint v81 = v79 + v80;
            uint v82 = v81 + lid;
            b6[v82] = ushort(v78);
        }
    } else {
        uint v83 = tgid.x - 10u;
        uint v85 = v83 * 256u;
        uint v86 = 4608u + v85;
        uint v87 = v86 + lid;
        uint v88 = uint(b1[v87]);
        uint v89 = v83 * 524288u;
        uint v90 = v3 * 256u;
        uint v91 = v89 + v90;
        uint v92 = v91 + lid;
        b7[v92] = ushort(v88);
    }
}
