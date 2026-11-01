#include <metal_stdlib>
using namespace metal;

kernel void dn_recurrence_k16v16(
    device ushort *b0 [[buffer(0)]],
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
    threadgroup float s0[128];
    threadgroup float s1[128];
    uint v3 = tgid.x / 1u;
    for (uint v9 = lid; v9 < 128u; v9 += 128u) {
        uint v10 = v3 * 128u;
        uint v11 = v10 + v9;
        uint v12 = uint(b0[v11]);
        float v13 = float(as_type<half>(ushort(v12)));
        float v14 = v13 * as_type<float>(0x3db504f3u);
        s0[v9] = v14;
        uint v15 = v11 + 2048u;
        uint v16 = uint(b0[v15]);
        float v17 = float(as_type<half>(ushort(v16)));
        s1[v9] = v17;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint v18 = uint(b1[tgid.x]);
    float v19 = float(as_type<half>(ushort(v18)));
    uint v21 = tgid.x + 16u;
    uint v22 = uint(b1[v21]);
    float v23 = float(as_type<half>(ushort(v22)));
    float v24 = exp(v23);
    uint v26 = tgid.x * 128u;
    uint v27 = 4096u + v26;
    uint v28 = v27 + lid;
    uint v29 = uint(b0[v28]);
    float v30 = float(as_type<half>(ushort(v29)));
    uint v32 = tgid.x * 16384u;
    uint v33 = v32 + lid;
    float m0 = as_type<float>(0x00000000u);
    for (uint v37 = 0u; v37 < 128u; v37 += 1u) {
        uint v38 = v37 * 128u;
        uint v39 = v33 + v38;
        uint v40 = uint(b2[v39]);
        float v41 = float(as_type<half>(ushort(v40)));
        float v42 = v41 * v24;
        uint v43 = uint(as_type<ushort>(half(v42)));
        b2[v39] = ushort(v43);
        float v44 = s1[v37];
        float v45 = m0;
        float v46 = fma(v42, v44, v45);
        m0 = v46;
    }
    float v47 = m0;
    float v48 = v30 - v47;
    float v49 = v48 * v19;
    float m1 = as_type<float>(0x00000000u);
    for (uint v50 = 0u; v50 < 128u; v50 += 1u) {
        uint v51 = v50 * 128u;
        uint v52 = v33 + v51;
        uint v53 = uint(b2[v52]);
        float v54 = float(as_type<half>(ushort(v53)));
        float v55 = s1[v50];
        float v56 = fma(v55, v49, v54);
        uint v57 = uint(as_type<ushort>(half(v56)));
        b2[v52] = ushort(v57);
        float v58 = s0[v50];
        float v59 = m1;
        float v60 = fma(v56, v58, v59);
        m1 = v60;
    }
    float v61 = m1;
    uint v62 = v26 + lid;
    uint v63 = uint(as_type<ushort>(half(v61)));
    b3[v62] = ushort(v63);
}
