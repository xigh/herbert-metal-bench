#include <metal_stdlib>
using namespace metal;

kernel void matvec_2048x1024_res_r4(
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
    uint v6 = v4 * 4u;
    bool v8 = v6 < 1024u;
    if (v8) {
        uint v12 = v6 + 0u;
        bool v13 = v12 < 1024u;
        uint v14 = v13 ? v12 : 1023u;
        uint v15 = v14 * 256u;
        uint v17 = v6 + 1u;
        bool v18 = v17 < 1024u;
        uint v19 = v18 ? v17 : 1023u;
        uint v20 = v19 * 256u;
        uint v22 = v6 + 2u;
        bool v23 = v22 < 1024u;
        uint v24 = v23 ? v22 : 1023u;
        uint v25 = v24 * 256u;
        uint v27 = v6 + 3u;
        bool v28 = v27 < 1024u;
        uint v29 = v28 ? v27 : 1023u;
        uint v30 = v29 * 256u;
        float m0 = as_type<float>(0x00000000u);
        float m1 = as_type<float>(0x00000000u);
        float m2 = as_type<float>(0x00000000u);
        float m3 = as_type<float>(0x00000000u);
        for (uint v34 = lane; v34 < 256u; v34 += 32u) {
            uint4 v35 = b1[v34];
            uint v36 = v35.x;
            float v37 = float(as_type<half>(ushort(v36)));
            float v38 = float(as_type<half>(ushort(v36 >> 16)));
            uint v39 = v35.y;
            float v40 = float(as_type<half>(ushort(v39)));
            float v41 = float(as_type<half>(ushort(v39 >> 16)));
            uint v42 = v35.z;
            float v43 = float(as_type<half>(ushort(v42)));
            float v44 = float(as_type<half>(ushort(v42 >> 16)));
            uint v45 = v35.w;
            float v46 = float(as_type<half>(ushort(v45)));
            float v47 = float(as_type<half>(ushort(v45 >> 16)));
            uint v48 = v15 + v34;
            uint4 v49 = b0[v48];
            float v50 = m0;
            uint v51 = v49.x;
            float v52 = float(as_type<half>(ushort(v51)));
            float v53 = fma(v52, v37, v50);
            float v54 = float(as_type<half>(ushort(v51 >> 16)));
            float v55 = fma(v54, v38, v53);
            uint v56 = v49.y;
            float v57 = float(as_type<half>(ushort(v56)));
            float v58 = fma(v57, v40, v55);
            float v59 = float(as_type<half>(ushort(v56 >> 16)));
            float v60 = fma(v59, v41, v58);
            uint v61 = v49.z;
            float v62 = float(as_type<half>(ushort(v61)));
            float v63 = fma(v62, v43, v60);
            float v64 = float(as_type<half>(ushort(v61 >> 16)));
            float v65 = fma(v64, v44, v63);
            uint v66 = v49.w;
            float v67 = float(as_type<half>(ushort(v66)));
            float v68 = fma(v67, v46, v65);
            float v69 = float(as_type<half>(ushort(v66 >> 16)));
            float v70 = fma(v69, v47, v68);
            m0 = v70;
            uint v71 = v20 + v34;
            uint4 v72 = b0[v71];
            float v73 = m1;
            uint v74 = v72.x;
            float v75 = float(as_type<half>(ushort(v74)));
            float v76 = fma(v75, v37, v73);
            float v77 = float(as_type<half>(ushort(v74 >> 16)));
            float v78 = fma(v77, v38, v76);
            uint v79 = v72.y;
            float v80 = float(as_type<half>(ushort(v79)));
            float v81 = fma(v80, v40, v78);
            float v82 = float(as_type<half>(ushort(v79 >> 16)));
            float v83 = fma(v82, v41, v81);
            uint v84 = v72.z;
            float v85 = float(as_type<half>(ushort(v84)));
            float v86 = fma(v85, v43, v83);
            float v87 = float(as_type<half>(ushort(v84 >> 16)));
            float v88 = fma(v87, v44, v86);
            uint v89 = v72.w;
            float v90 = float(as_type<half>(ushort(v89)));
            float v91 = fma(v90, v46, v88);
            float v92 = float(as_type<half>(ushort(v89 >> 16)));
            float v93 = fma(v92, v47, v91);
            m1 = v93;
            uint v94 = v25 + v34;
            uint4 v95 = b0[v94];
            float v96 = m2;
            uint v97 = v95.x;
            float v98 = float(as_type<half>(ushort(v97)));
            float v99 = fma(v98, v37, v96);
            float v100 = float(as_type<half>(ushort(v97 >> 16)));
            float v101 = fma(v100, v38, v99);
            uint v102 = v95.y;
            float v103 = float(as_type<half>(ushort(v102)));
            float v104 = fma(v103, v40, v101);
            float v105 = float(as_type<half>(ushort(v102 >> 16)));
            float v106 = fma(v105, v41, v104);
            uint v107 = v95.z;
            float v108 = float(as_type<half>(ushort(v107)));
            float v109 = fma(v108, v43, v106);
            float v110 = float(as_type<half>(ushort(v107 >> 16)));
            float v111 = fma(v110, v44, v109);
            uint v112 = v95.w;
            float v113 = float(as_type<half>(ushort(v112)));
            float v114 = fma(v113, v46, v111);
            float v115 = float(as_type<half>(ushort(v112 >> 16)));
            float v116 = fma(v115, v47, v114);
            m2 = v116;
            uint v117 = v30 + v34;
            uint4 v118 = b0[v117];
            float v119 = m3;
            uint v120 = v118.x;
            float v121 = float(as_type<half>(ushort(v120)));
            float v122 = fma(v121, v37, v119);
            float v123 = float(as_type<half>(ushort(v120 >> 16)));
            float v124 = fma(v123, v38, v122);
            uint v125 = v118.y;
            float v126 = float(as_type<half>(ushort(v125)));
            float v127 = fma(v126, v40, v124);
            float v128 = float(as_type<half>(ushort(v125 >> 16)));
            float v129 = fma(v128, v41, v127);
            uint v130 = v118.z;
            float v131 = float(as_type<half>(ushort(v130)));
            float v132 = fma(v131, v43, v129);
            float v133 = float(as_type<half>(ushort(v130 >> 16)));
            float v134 = fma(v133, v44, v132);
            uint v135 = v118.w;
            float v136 = float(as_type<half>(ushort(v135)));
            float v137 = fma(v136, v46, v134);
            float v138 = float(as_type<half>(ushort(v135 >> 16)));
            float v139 = fma(v138, v47, v137);
            m3 = v139;
        }
        bool v141 = lane == 0u;
        float v142 = m0;
        float v143 = simd_sum(v142);
        uint v145 = v6 + 0u;
        bool v146 = v145 < 1024u;
        bool v147 = v141 && v146;
        if (v147) {
            float v148 = float(half(v143));
            uint v149 = uint(b2[v145]);
            float v150 = float(as_type<half>(ushort(v149)));
            float v151 = v150 + v148;
            uint v152 = uint(as_type<ushort>(half(v151)));
            b2[v145] = ushort(v152);
        }
        float v153 = m1;
        float v154 = simd_sum(v153);
        uint v156 = v6 + 1u;
        bool v157 = v156 < 1024u;
        bool v158 = v141 && v157;
        if (v158) {
            float v159 = float(half(v154));
            uint v160 = uint(b2[v156]);
            float v161 = float(as_type<half>(ushort(v160)));
            float v162 = v161 + v159;
            uint v163 = uint(as_type<ushort>(half(v162)));
            b2[v156] = ushort(v163);
        }
        float v164 = m2;
        float v165 = simd_sum(v164);
        uint v167 = v6 + 2u;
        bool v168 = v167 < 1024u;
        bool v169 = v141 && v168;
        if (v169) {
            float v170 = float(half(v165));
            uint v171 = uint(b2[v167]);
            float v172 = float(as_type<half>(ushort(v171)));
            float v173 = v172 + v170;
            uint v174 = uint(as_type<ushort>(half(v173)));
            b2[v167] = ushort(v174);
        }
        float v175 = m3;
        float v176 = simd_sum(v175);
        uint v178 = v6 + 3u;
        bool v179 = v178 < 1024u;
        bool v180 = v141 && v179;
        if (v180) {
            float v181 = float(half(v176));
            uint v182 = uint(b2[v178]);
            float v183 = float(as_type<half>(ushort(v182)));
            float v184 = v183 + v181;
            uint v185 = uint(as_type<ushort>(half(v184)));
            b2[v178] = ushort(v185);
        }
    }
}
