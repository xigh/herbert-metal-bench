#include <metal_stdlib>
using namespace metal;

kernel void matvec_3584x1024_res_r2(
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
    uint v6 = v4 * 2u;
    bool v8 = v6 < 1024u;
    if (v8) {
        uint v12 = v6 + 0u;
        bool v13 = v12 < 1024u;
        uint v14 = v13 ? v12 : 1023u;
        uint v15 = v14 * 448u;
        uint v17 = v6 + 1u;
        bool v18 = v17 < 1024u;
        uint v19 = v18 ? v17 : 1023u;
        uint v20 = v19 * 448u;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        for (uint v24 = lane; v24 < 448u; v24 += 32u) {
            uint4 v25 = b1[v24];
            uint v26 = v25.x;
            float v27 = float(as_type<half>(ushort(v26)));
            float v28 = float(as_type<half>(ushort(v26 >> 16)));
            uint v29 = v25.y;
            float v30 = float(as_type<half>(ushort(v29)));
            float v31 = float(as_type<half>(ushort(v29 >> 16)));
            uint v32 = v25.z;
            float v33 = float(as_type<half>(ushort(v32)));
            float v34 = float(as_type<half>(ushort(v32 >> 16)));
            uint v35 = v25.w;
            float v36 = float(as_type<half>(ushort(v35)));
            float v37 = float(as_type<half>(ushort(v35 >> 16)));
            uint v38 = v15 + v24;
            uint4 v39 = b0[v38];
            float v40 = m0;
            uint v41 = v39.x;
            float v42 = float(as_type<half>(ushort(v41)));
            float v43 = fma(v42, v27, v40);
            float v44 = float(as_type<half>(ushort(v41 >> 16)));
            float v45 = fma(v44, v28, v43);
            uint v46 = v39.y;
            float v47 = float(as_type<half>(ushort(v46)));
            float v48 = fma(v47, v30, v45);
            float v49 = float(as_type<half>(ushort(v46 >> 16)));
            float v50 = fma(v49, v31, v48);
            uint v51 = v39.z;
            float v52 = float(as_type<half>(ushort(v51)));
            float v53 = fma(v52, v33, v50);
            float v54 = float(as_type<half>(ushort(v51 >> 16)));
            float v55 = fma(v54, v34, v53);
            uint v56 = v39.w;
            float v57 = float(as_type<half>(ushort(v56)));
            float v58 = fma(v57, v36, v55);
            float v59 = float(as_type<half>(ushort(v56 >> 16)));
            float v60 = fma(v59, v37, v58);
            m0 = v60;
            uint v61 = v20 + v24;
            uint4 v62 = b0[v61];
            float v63 = m1;
            uint v64 = v62.x;
            float v65 = float(as_type<half>(ushort(v64)));
            float v66 = fma(v65, v27, v63);
            float v67 = float(as_type<half>(ushort(v64 >> 16)));
            float v68 = fma(v67, v28, v66);
            uint v69 = v62.y;
            float v70 = float(as_type<half>(ushort(v69)));
            float v71 = fma(v70, v30, v68);
            float v72 = float(as_type<half>(ushort(v69 >> 16)));
            float v73 = fma(v72, v31, v71);
            uint v74 = v62.z;
            float v75 = float(as_type<half>(ushort(v74)));
            float v76 = fma(v75, v33, v73);
            float v77 = float(as_type<half>(ushort(v74 >> 16)));
            float v78 = fma(v77, v34, v76);
            uint v79 = v62.w;
            float v80 = float(as_type<half>(ushort(v79)));
            float v81 = fma(v80, v36, v78);
            float v82 = float(as_type<half>(ushort(v79 >> 16)));
            float v83 = fma(v82, v37, v81);
            m1 = v83;
        }
        bool v85 = lane == 0u;
        float v86 = m0;
        float v87 = simd_sum(v86);
        uint v89 = v6 + 0u;
        bool v90 = v89 < 1024u;
        bool v91 = v85 && v90;
        if (v91) {
            float v92 = float(half(v87));
            uint v93 = uint(b2[v89]);
            float v94 = float(as_type<half>(ushort(v93)));
            float v95 = v94 + v92;
            uint v96 = uint(as_type<ushort>(half(v95)));
            b2[v89] = ushort(v96);
        }
        float v97 = m1;
        float v98 = simd_sum(v97);
        uint v100 = v6 + 1u;
        bool v101 = v100 < 1024u;
        bool v102 = v85 && v101;
        if (v102) {
            float v103 = float(half(v98));
            uint v104 = uint(b2[v100]);
            float v105 = float(as_type<half>(ushort(v104)));
            float v106 = v105 + v103;
            uint v107 = uint(as_type<ushort>(half(v106)));
            b2[v100] = ushort(v107);
        }
    }
}
