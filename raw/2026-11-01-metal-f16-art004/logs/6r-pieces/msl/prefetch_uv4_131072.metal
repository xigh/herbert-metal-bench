#include <metal_stdlib>
using namespace metal;

kernel void prefetch_uv4_131072(
    device uint4 *b0 [[buffer(0)]],
    device uint *b1 [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint v3 = tgid.x * 2048u;
    uint v4 = v3 + lid;
    uint v8 = v4 + 0u;
    bool v9 = v8 < 131072u;
    uint v11 = v9 ? v8 : 131071u;
    uint4 v12 = b0[v11];
    uint v13 = v12.x;
    uint v14 = 0u ^ v13;
    uint v15 = v12.y;
    uint v16 = v14 ^ v15;
    uint v17 = v12.z;
    uint v18 = v16 ^ v17;
    uint v19 = v12.w;
    uint v20 = v18 ^ v19;
    uint v22 = v4 + 256u;
    bool v23 = v22 < 131072u;
    uint v25 = v23 ? v22 : 131071u;
    uint4 v26 = b0[v25];
    uint v27 = v26.x;
    uint v28 = v20 ^ v27;
    uint v29 = v26.y;
    uint v30 = v28 ^ v29;
    uint v31 = v26.z;
    uint v32 = v30 ^ v31;
    uint v33 = v26.w;
    uint v34 = v32 ^ v33;
    uint v36 = v4 + 512u;
    bool v37 = v36 < 131072u;
    uint v39 = v37 ? v36 : 131071u;
    uint4 v40 = b0[v39];
    uint v41 = v40.x;
    uint v42 = v34 ^ v41;
    uint v43 = v40.y;
    uint v44 = v42 ^ v43;
    uint v45 = v40.z;
    uint v46 = v44 ^ v45;
    uint v47 = v40.w;
    uint v48 = v46 ^ v47;
    uint v50 = v4 + 768u;
    bool v51 = v50 < 131072u;
    uint v53 = v51 ? v50 : 131071u;
    uint4 v54 = b0[v53];
    uint v55 = v54.x;
    uint v56 = v48 ^ v55;
    uint v57 = v54.y;
    uint v58 = v56 ^ v57;
    uint v59 = v54.z;
    uint v60 = v58 ^ v59;
    uint v61 = v54.w;
    uint v62 = v60 ^ v61;
    uint v64 = v4 + 1024u;
    bool v65 = v64 < 131072u;
    uint v67 = v65 ? v64 : 131071u;
    uint4 v68 = b0[v67];
    uint v69 = v68.x;
    uint v70 = v62 ^ v69;
    uint v71 = v68.y;
    uint v72 = v70 ^ v71;
    uint v73 = v68.z;
    uint v74 = v72 ^ v73;
    uint v75 = v68.w;
    uint v76 = v74 ^ v75;
    uint v78 = v4 + 1280u;
    bool v79 = v78 < 131072u;
    uint v81 = v79 ? v78 : 131071u;
    uint4 v82 = b0[v81];
    uint v83 = v82.x;
    uint v84 = v76 ^ v83;
    uint v85 = v82.y;
    uint v86 = v84 ^ v85;
    uint v87 = v82.z;
    uint v88 = v86 ^ v87;
    uint v89 = v82.w;
    uint v90 = v88 ^ v89;
    uint v92 = v4 + 1536u;
    bool v93 = v92 < 131072u;
    uint v95 = v93 ? v92 : 131071u;
    uint4 v96 = b0[v95];
    uint v97 = v96.x;
    uint v98 = v90 ^ v97;
    uint v99 = v96.y;
    uint v100 = v98 ^ v99;
    uint v101 = v96.z;
    uint v102 = v100 ^ v101;
    uint v103 = v96.w;
    uint v104 = v102 ^ v103;
    uint v106 = v4 + 1792u;
    bool v107 = v106 < 131072u;
    uint v109 = v107 ? v106 : 131071u;
    uint4 v110 = b0[v109];
    uint v111 = v110.x;
    uint v112 = v104 ^ v111;
    uint v113 = v110.y;
    uint v114 = v112 ^ v113;
    uint v115 = v110.z;
    uint v116 = v114 ^ v115;
    uint v117 = v110.w;
    uint v118 = v116 ^ v117;
    bool v120 = v118 == 2654435769u;
    if (v120) {
        b1[0u] = v118;
    }
}
