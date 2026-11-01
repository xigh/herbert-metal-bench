#include <metal_stdlib>
using namespace metal;

kernel void dn_conv_6144x4(
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
    bool v2 = tpig.x < 6144u;
    if (v2) {
        uint v4 = tpig.x * 4u;
        uint v5 = uint(b0[tpig.x]);
        float v6 = float(as_type<half>(ushort(v5)));
        uint v9 = v4 + 1u;
        uint v10 = uint(b1[v9]);
        float v11 = float(as_type<half>(ushort(v10)));
        uint v13 = v4 + 2u;
        uint v14 = uint(b1[v13]);
        float v15 = float(as_type<half>(ushort(v14)));
        uint v17 = v4 + 3u;
        uint v18 = uint(b1[v17]);
        float v19 = float(as_type<half>(ushort(v18)));
        uint v21 = v4 + 0u;
        uint v22 = uint(as_type<ushort>(half(v11)));
        b1[v21] = ushort(v22);
        uint v23 = uint(b2[v21]);
        float v24 = float(as_type<half>(ushort(v23)));
        float v25 = fma(v11, v24, as_type<float>(0x00000000u));
        uint v27 = v4 + 1u;
        uint v28 = uint(as_type<ushort>(half(v15)));
        b1[v27] = ushort(v28);
        uint v29 = uint(b2[v27]);
        float v30 = float(as_type<half>(ushort(v29)));
        float v31 = fma(v15, v30, v25);
        uint v33 = v4 + 2u;
        uint v34 = uint(as_type<ushort>(half(v19)));
        b1[v33] = ushort(v34);
        uint v35 = uint(b2[v33]);
        float v36 = float(as_type<half>(ushort(v35)));
        float v37 = fma(v19, v36, v31);
        uint v39 = v4 + 3u;
        uint v40 = uint(as_type<ushort>(half(v6)));
        b1[v39] = ushort(v40);
        uint v41 = uint(b2[v39]);
        float v42 = float(as_type<half>(ushort(v41)));
        float v43 = fma(v6, v42, v37);
        float v44 = -v43;
        float v45 = exp(v44);
        float v47 = as_type<float>(0x3f800000u) + v45;
        float v48 = as_type<float>(0x3f800000u) / v47;
        float v49 = v43 * v48;
        uint v50 = uint(as_type<ushort>(half(v49)));
        b0[tpig.x] = ushort(v50);
    }
}
