#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_rows_1024_p1(
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
    uint v2 = tgid.x * 1024u;
    float m0 = as_type<float>(0x00000000u);
    for (uint v6 = lid; v6 < 1024u; v6 += 256u) {
        uint v7 = v2 + v6;
        uint v8 = uint(b0[v7]);
        float v9 = float(as_type<half>(ushort(v8)));
        float v10 = m0;
        float v11 = fma(v9, v9, v10);
        m0 = v11;
    }
    float v12 = m0;
    float v13 = simd_sum(v12);
    bool v17 = lane == 0u;
    if (v17) {
        s0[sgid] = v13;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    bool v19 = lid == 0u;
    if (v19) {
        float v21 = s0[0u];
        float m1 = v21;
        for (uint v24 = 1u; v24 < 8u; v24 += 1u) {
            float v25 = m1;
            float v26 = s0[v24];
            float v27 = v25 + v26;
            m1 = v27;
        }
        float v28 = m1;
        s0[8u] = v28;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v31 = s0[8u];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float v34 = v31 * as_type<float>(0x3a800000u);
    float v35 = v34 + as_type<float>(0x358637bdu);
    float v36 = sqrt(v35);
    float v38 = as_type<float>(0x3f800000u) / v36;
    for (uint v39 = lid; v39 < 1024u; v39 += 256u) {
        uint v40 = v2 + v39;
        uint v41 = uint(b0[v40]);
        float v42 = float(as_type<half>(ushort(v41)));
        uint v43 = uint(b1[v39]);
        float v44 = float(as_type<half>(ushort(v43)));
        float v46 = v44 + as_type<float>(0x3f800000u);
        float v47 = v42 * v38;
        float v48 = v47 * v46;
        uint v49 = uint(as_type<ushort>(half(v48)));
        b2[v40] = ushort(v49);
    }
}
