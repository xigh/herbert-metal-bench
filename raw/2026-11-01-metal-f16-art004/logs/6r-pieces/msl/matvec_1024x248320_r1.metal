#include <metal_stdlib>
using namespace metal;

kernel void matvec_1024x248320_r1(
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
    uint v3 = tgid.x * 4u;
    uint v4 = v3 + sgid;
    uint v6 = v4 * 1u;
    bool v8 = v6 < 248320u;
    if (v8) {
        uint v12 = v6 + 0u;
        bool v13 = v12 < 248320u;
        uint v14 = v13 ? v12 : 248319u;
        uint v15 = v14 * 128u;
        float m0 = as_type<float>(0x00000000u);
        for (uint v19 = lane; v19 < 128u; v19 += 32u) {
            uint4 v20 = b1[v19];
            uint v21 = v20.x;
            float v22 = float(as_type<half>(ushort(v21)));
            float v23 = float(as_type<half>(ushort(v21 >> 16)));
            uint v24 = v20.y;
            float v25 = float(as_type<half>(ushort(v24)));
            float v26 = float(as_type<half>(ushort(v24 >> 16)));
            uint v27 = v20.z;
            float v28 = float(as_type<half>(ushort(v27)));
            float v29 = float(as_type<half>(ushort(v27 >> 16)));
            uint v30 = v20.w;
            float v31 = float(as_type<half>(ushort(v30)));
            float v32 = float(as_type<half>(ushort(v30 >> 16)));
            uint v33 = v15 + v19;
            uint4 v34 = b0[v33];
            float v35 = m0;
            uint v36 = v34.x;
            float v37 = float(as_type<half>(ushort(v36)));
            float v38 = fma(v37, v22, v35);
            float v39 = float(as_type<half>(ushort(v36 >> 16)));
            float v40 = fma(v39, v23, v38);
            uint v41 = v34.y;
            float v42 = float(as_type<half>(ushort(v41)));
            float v43 = fma(v42, v25, v40);
            float v44 = float(as_type<half>(ushort(v41 >> 16)));
            float v45 = fma(v44, v26, v43);
            uint v46 = v34.z;
            float v47 = float(as_type<half>(ushort(v46)));
            float v48 = fma(v47, v28, v45);
            float v49 = float(as_type<half>(ushort(v46 >> 16)));
            float v50 = fma(v49, v29, v48);
            uint v51 = v34.w;
            float v52 = float(as_type<half>(ushort(v51)));
            float v53 = fma(v52, v31, v50);
            float v54 = float(as_type<half>(ushort(v51 >> 16)));
            float v55 = fma(v54, v32, v53);
            m0 = v55;
        }
        bool v57 = lane == 0u;
        float v58 = m0;
        float v59 = simd_sum(v58);
        uint v61 = v6 + 0u;
        bool v62 = v61 < 248320u;
        bool v63 = v57 && v62;
        if (v63) {
            uint v64 = uint(as_type<ushort>(half(v59)));
            b2[v61] = ushort(v64);
        }
    }
}
