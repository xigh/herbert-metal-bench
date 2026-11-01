#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_1024_p1_u(
    device ushort *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[9];
    uint v2 = lid + 0u;
    uint v3 = uint(b0[v2]);
    uint v5 = lid + 256u;
    uint v6 = uint(b0[v5]);
    uint v8 = lid + 512u;
    uint v9 = uint(b0[v8]);
    uint v11 = lid + 768u;
    uint v12 = uint(b0[v11]);
    float v14 = float(as_type<half>(ushort(v3)));
    float v15 = float(as_type<half>(ushort(v6)));
    float v16 = float(as_type<half>(ushort(v9)));
    float v17 = float(as_type<half>(ushort(v12)));
    float v18 = fma(v14, v14, as_type<float>(0x00000000u));
    float v19 = fma(v15, v15, v18);
    float v20 = fma(v16, v16, v19);
    float v21 = fma(v17, v17, v20);
    float v22 = simd_sum(v21);
    bool v26 = lane == 0u;
    if (v26) {
        s0[sgid] = v22;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v28 = lid == 0u;
    if (v28) {
        float v30 = s0[0u];
        float m0 = v30;
        for (uint v33 = 1u; v33 < 8u; v33 += 1u) {
            float v34 = m0;
            float v35 = s0[v33];
            float v36 = v34 + v35;
            m0 = v36;
        }
        float v37 = m0;
        s0[8u] = v37;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v40 = s0[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v43 = v40 * as_type<float>(0x3a800000u);
    float v44 = v43 + as_type<float>(0x358637bdu);
    float v45 = sqrt(v44);
    float v47 = as_type<float>(0x3f800000u) / v45;
    uint v48 = uint(b1[v2]);
    uint v49 = uint(b1[v5]);
    uint v50 = uint(b1[v8]);
    uint v51 = uint(b1[v11]);
    float v52 = float(as_type<half>(ushort(v48)));
    float v54 = v52 + as_type<float>(0x3f800000u);
    float v55 = v14 * v47;
    float v56 = v55 * v54;
    uint v57 = uint(as_type<ushort>(half(v56)));
    b2[v2] = ushort(v57);
    float v58 = float(as_type<half>(ushort(v49)));
    float v60 = v58 + as_type<float>(0x3f800000u);
    float v61 = v15 * v47;
    float v62 = v61 * v60;
    uint v63 = uint(as_type<ushort>(half(v62)));
    b2[v5] = ushort(v63);
    float v64 = float(as_type<half>(ushort(v50)));
    float v66 = v64 + as_type<float>(0x3f800000u);
    float v67 = v16 * v47;
    float v68 = v67 * v66;
    uint v69 = uint(as_type<ushort>(half(v68)));
    b2[v8] = ushort(v69);
    float v70 = float(as_type<half>(ushort(v51)));
    float v72 = v70 + as_type<float>(0x3f800000u);
    float v73 = v17 * v47;
    float v74 = v73 * v72;
    uint v75 = uint(as_type<ushort>(half(v74)));
    b2[v11] = ushort(v75);
}
