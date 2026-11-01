#include <metal_stdlib>
using namespace metal;

kernel void sdpa_decode_merge_h8d256_c8(
    device ushort *b0 [[buffer(0)]],
    device float *b1 [[buffer(1)]],
    device ushort *b2 [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint v4 = tgid.x * 8u;
    uint v5 = v4 * 258u;
    uint v9 = v5 + 256u;
    float v10 = b1[v9];
    float v11 = fmax(as_type<float>(0xff7fffffu), v10);
    uint v13 = v5 + 514u;
    float v14 = b1[v13];
    float v15 = fmax(v11, v14);
    uint v17 = v5 + 772u;
    float v18 = b1[v17];
    float v19 = fmax(v15, v18);
    uint v21 = v5 + 1030u;
    float v22 = b1[v21];
    float v23 = fmax(v19, v22);
    uint v25 = v5 + 1288u;
    float v26 = b1[v25];
    float v27 = fmax(v23, v26);
    uint v29 = v5 + 1546u;
    float v30 = b1[v29];
    float v31 = fmax(v27, v30);
    uint v33 = v5 + 1804u;
    float v34 = b1[v33];
    float v35 = fmax(v31, v34);
    uint v37 = v5 + 2062u;
    float v38 = b1[v37];
    float v39 = fmax(v35, v38);
    float v41 = v10 - v39;
    float v42 = exp(v41);
    uint v44 = v5 + 257u;
    float v45 = b1[v44];
    float v46 = fma(v45, v42, as_type<float>(0x00000000u));
    float v47 = v14 - v39;
    float v48 = exp(v47);
    uint v50 = v5 + 515u;
    float v51 = b1[v50];
    float v52 = fma(v51, v48, v46);
    float v53 = v18 - v39;
    float v54 = exp(v53);
    uint v56 = v5 + 773u;
    float v57 = b1[v56];
    float v58 = fma(v57, v54, v52);
    float v59 = v22 - v39;
    float v60 = exp(v59);
    uint v62 = v5 + 1031u;
    float v63 = b1[v62];
    float v64 = fma(v63, v60, v58);
    float v65 = v26 - v39;
    float v66 = exp(v65);
    uint v68 = v5 + 1289u;
    float v69 = b1[v68];
    float v70 = fma(v69, v66, v64);
    float v71 = v30 - v39;
    float v72 = exp(v71);
    uint v74 = v5 + 1547u;
    float v75 = b1[v74];
    float v76 = fma(v75, v72, v70);
    float v77 = v34 - v39;
    float v78 = exp(v77);
    uint v80 = v5 + 1805u;
    float v81 = b1[v80];
    float v82 = fma(v81, v78, v76);
    float v83 = v38 - v39;
    float v84 = exp(v83);
    uint v86 = v5 + 2063u;
    float v87 = b1[v86];
    float v88 = fma(v87, v84, v82);
    float v90 = as_type<float>(0x3f800000u) / v88;
    uint v93 = tgid.x * 512u;
    for (uint v94 = lid; v94 < 256u; v94 += 256u) {
        uint v97 = v5 + 0u;
        uint v98 = v97 + v94;
        float v99 = b1[v98];
        float v100 = fma(v99, v42, as_type<float>(0x00000000u));
        uint v102 = v5 + 258u;
        uint v103 = v102 + v94;
        float v104 = b1[v103];
        float v105 = fma(v104, v48, v100);
        uint v107 = v5 + 516u;
        uint v108 = v107 + v94;
        float v109 = b1[v108];
        float v110 = fma(v109, v54, v105);
        uint v112 = v5 + 774u;
        uint v113 = v112 + v94;
        float v114 = b1[v113];
        float v115 = fma(v114, v60, v110);
        uint v117 = v5 + 1032u;
        uint v118 = v117 + v94;
        float v119 = b1[v118];
        float v120 = fma(v119, v66, v115);
        uint v122 = v5 + 1290u;
        uint v123 = v122 + v94;
        float v124 = b1[v123];
        float v125 = fma(v124, v72, v120);
        uint v127 = v5 + 1548u;
        uint v128 = v127 + v94;
        float v129 = b1[v128];
        float v130 = fma(v129, v78, v125);
        uint v132 = v5 + 1806u;
        uint v133 = v132 + v94;
        float v134 = b1[v133];
        float v135 = fma(v134, v84, v130);
        float v136 = v135 * v90;
        float v137 = float(half(v136));
        uint v138 = v93 + 256u;
        uint v139 = v138 + v94;
        uint v140 = uint(b0[v139]);
        float v141 = float(as_type<half>(ushort(v140)));
        float v142 = -v141;
        float v143 = exp(v142);
        float v145 = as_type<float>(0x3f800000u) + v143;
        float v146 = as_type<float>(0x3f800000u) / v145;
        float v147 = v137 * v146;
        uint v148 = uint(as_type<ushort>(half(v147)));
        uint v149 = tgid.x * 256u;
        uint v150 = v149 + v94;
        b2[v150] = ushort(v148);
    }
}
