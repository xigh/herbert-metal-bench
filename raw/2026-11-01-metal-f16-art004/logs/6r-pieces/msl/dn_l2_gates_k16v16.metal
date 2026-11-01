#include <metal_stdlib>
using namespace metal;

kernel void dn_l2_gates_k16v16(
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
    bool v3 = tgid.x < 32u;
    if (v3) {
        uint v5 = tgid.x * 128u;
        uint v6 = v5 + lid;
        uint v7 = uint(b0[v6]);
        float v8 = float(as_type<half>(ushort(v7)));
        float v9 = v8 * v8;
        float v10 = simd_sum(v9);
        bool v14 = lane == 0u;
        if (v14) {
            s0[sgid] = v10;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        bool v16 = lid == 0u;
        if (v16) {
            float v18 = s0[0u];
            float m0 = v18;
            for (uint v21 = 1u; v21 < 4u; v21 += 1u) {
                float v22 = m0;
                float v23 = s0[v21];
                float v24 = v22 + v23;
                m0 = v24;
            }
            float v25 = m0;
            s0[4u] = v25;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v28 = s0[4u];
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        float v30 = v28 + as_type<float>(0x358637bdu);
        float v31 = sqrt(v30);
        float v33 = as_type<float>(0x3f800000u) / v31;
        float v34 = v8 * v33;
        uint v35 = uint(as_type<ushort>(half(v34)));
        b0[v6] = ushort(v35);
    } else {
        for (uint v38 = lid; v38 < 16u; v38 += 128u) {
            uint v41 = 8192u + v38;
            uint v42 = 8208u + v38;
            uint v43 = uint(b0[v41]);
            float v44 = float(as_type<half>(ushort(v43)));
            uint v45 = uint(b0[v42]);
            float v46 = float(as_type<half>(ushort(v45)));
            float v47 = -v46;
            float v48 = exp(v47);
            float v50 = as_type<float>(0x3f800000u) + v48;
            float v51 = as_type<float>(0x3f800000u) / v50;
            float v52 = b2[v38];
            float v53 = v44 + v52;
            float v54 = exp(v53);
            float v56 = as_type<float>(0x3f800000u) + v54;
            float v57 = log(v56);
            bool v59 = v53 > as_type<float>(0x41a00000u);
            float v60 = v59 ? v53 : v57;
            float v61 = b1[v38];
            float v62 = exp(v61);
            float v63 = v62 * v60;
            float v64 = -v63;
            uint v65 = uint(as_type<ushort>(half(v51)));
            b3[v38] = ushort(v65);
            uint v66 = v38 + 16u;
            uint v67 = uint(as_type<ushort>(half(v64)));
            b3[v66] = ushort(v67);
        }
    }
}
