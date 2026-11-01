#include <metal_stdlib>
using namespace metal;

kernel void dn_l2_gates_rows_k16v16(
    device ushort *b0 [[buffer(0)]],
    device float *b1 [[buffer(1)]],
    device float *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[5];
    uint v2 = tgid.x / 33u;
    uint v3 = tgid.x % 33u;
    uint v5 = v2 * 8224u;
    bool v8 = v3 < 32u;
    if (v8) {
        uint v10 = v3 * 128u;
        uint v11 = v5 + v10;
        uint v12 = v11 + lid;
        uint v13 = uint(b0[v12]);
        float v14 = float(as_type<half>(ushort(v13)));
        float v15 = v14 * v14;
        float v16 = simd_sum(v15);
        bool v20 = lane == 0u;
        if (v20) {
            s0[sgid] = v16;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v22 = lid == 0u;
        if (v22) {
            float v24 = s0[0u];
            float m0 = v24;
            for (uint v27 = 1u; v27 < 4u; v27 += 1u) {
                float v28 = m0;
                float v29 = s0[v27];
                float v30 = v28 + v29;
                m0 = v30;
            }
            float v31 = m0;
            s0[4u] = v31;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v34 = s0[4u];
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v36 = v34 + as_type<float>(0x358637bdu);
        float v37 = sqrt(v36);
        float v39 = as_type<float>(0x3f800000u) / v37;
        float v40 = v14 * v39;
        uint v41 = uint(as_type<ushort>(half(v40)));
        b0[v12] = ushort(v41);
    } else {
        uint v44 = v2 * 32u;
        for (uint v46 = lid; v46 < 16u; v46 += 128u) {
            uint v49 = 8192u + v46;
            uint v50 = v5 + v49;
            uint v51 = 8208u + v46;
            uint v52 = v5 + v51;
            uint v53 = uint(b0[v50]);
            float v54 = float(as_type<half>(ushort(v53)));
            uint v55 = uint(b0[v52]);
            float v56 = float(as_type<half>(ushort(v55)));
            float v57 = -v56;
            float v58 = exp(v57);
            float v60 = as_type<float>(0x3f800000u) + v58;
            float v61 = as_type<float>(0x3f800000u) / v60;
            float v62 = b2[v46];
            float v63 = v54 + v62;
            float v64 = exp(v63);
            float v66 = as_type<float>(0x3f800000u) + v64;
            float v67 = log(v66);
            bool v69 = v63 > as_type<float>(0x41a00000u);
            float v70 = v69 ? v63 : v67;
            float v71 = b1[v46];
            float v72 = exp(v71);
            float v73 = v72 * v70;
            float v74 = -v73;
            uint v75 = v44 + v46;
            uint v76 = uint(as_type<ushort>(half(v61)));
            b3[v75] = ushort(v76);
            uint v77 = v75 + 16u;
            uint v78 = uint(as_type<ushort>(half(v74)));
            b3[v77] = ushort(v78);
        }
    }
}
