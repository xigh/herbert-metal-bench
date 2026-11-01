#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x7168_r1_k2_p2560(
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
    threadgroup float s0[4];
    uint v3 = sgid / 2u;
    uint v4 = sgid % 2u;
    for (uint v7 = tgid.x; v7 < 3584u; v7 += 2560u) {
        uint v9 = v7 * 2u;
        uint v10 = v9 + v3;
        uint v13 = v10 * 1u;
        uint v18 = v13 + 0u;
        bool v19 = v18 < 7168u;
        uint v20 = v19 ? v18 : 7167u;
        uint v21 = v20 * 128u;
        uint v24 = v4 * 32u;
        uint v25 = v24 + lane;
        float m0 = as_type<float>(0x00000000u);
        for (uint v28 = v25; v28 < 128u; v28 += 64u) {
            uint4 v30 = b1[v28];
            uint v31 = v30.x;
            float v32 = float(as_type<half>(ushort(v31)));
            float v33 = float(as_type<half>(ushort(v31 >> 16)));
            uint v34 = v30.y;
            float v35 = float(as_type<half>(ushort(v34)));
            float v36 = float(as_type<half>(ushort(v34 >> 16)));
            uint v37 = v30.z;
            float v38 = float(as_type<half>(ushort(v37)));
            float v39 = float(as_type<half>(ushort(v37 >> 16)));
            uint v40 = v30.w;
            float v41 = float(as_type<half>(ushort(v40)));
            float v42 = float(as_type<half>(ushort(v40 >> 16)));
            uint v43 = v21 + v28;
            uint4 v44 = b0[v43];
            float v45 = m0;
            uint v46 = v44.x;
            float v47 = float(as_type<half>(ushort(v46)));
            float v48 = fma(v47, v32, v45);
            float v49 = float(as_type<half>(ushort(v46 >> 16)));
            float v50 = fma(v49, v33, v48);
            uint v51 = v44.y;
            float v52 = float(as_type<half>(ushort(v51)));
            float v53 = fma(v52, v35, v50);
            float v54 = float(as_type<half>(ushort(v51 >> 16)));
            float v55 = fma(v54, v36, v53);
            uint v56 = v44.z;
            float v57 = float(as_type<half>(ushort(v56)));
            float v58 = fma(v57, v38, v55);
            float v59 = float(as_type<half>(ushort(v56 >> 16)));
            float v60 = fma(v59, v39, v58);
            uint v61 = v44.w;
            float v62 = float(as_type<half>(ushort(v61)));
            float v63 = fma(v62, v41, v60);
            float v64 = float(as_type<half>(ushort(v61 >> 16)));
            float v65 = fma(v64, v42, v63);
            m0 = v65;
        }
        bool v67 = lane == 0u;
        uint v68 = v3 * 1u;
        float v69 = m0;
        float v70 = simd_sum(v69);
        uint v72 = v68 + 0u;
        uint v73 = v72 * 2u;
        uint v74 = v73 + v4;
        if (v67) {
            s0[v74] = v70;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v75 = v4 == 0u;
        bool v76 = v67 && v75;
        uint v78 = v13 + 0u;
        bool v79 = v78 < 7168u;
        bool v80 = v76 && v79;
        if (v80) {
            uint v82 = v68 + 0u;
            uint v83 = v82 * 2u;
            float v84 = s0[v83];
            uint v86 = v83 + 1u;
            float v87 = s0[v86];
            float v88 = v84 + v87;
            uint v89 = uint(as_type<ushort>(half(v88)));
            b2[v78] = ushort(v89);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
