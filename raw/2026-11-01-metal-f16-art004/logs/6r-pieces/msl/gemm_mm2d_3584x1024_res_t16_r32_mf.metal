#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void gemm_mm2d_3584x1024_res_t16_r32_mf(
    device uint *b0 [[buffer(0)]],
    device uint4 *b1 [[buffer(1)]],
    device uint4 *b2 [[buffer(2)]],
    device ushort *b3 [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[512];
    uint v1 = b0[3u];
    uint v5 = v1 + 31u;
    uint v6 = v5 / 32u;
    uint v7 = tgid.x % v6;
    uint v8 = tgid.x / v6;
    uint v12 = v7 * 32u;
    uint v13 = v8 * 16u;
    uint v14 = v1 - v12;
    bool v15 = 32u < v14;
    uint v16 = v15 ? 32u : v14;
    uint v17 = 1024u - v13;
    bool v18 = 16u < v17;
    uint v19 = v18 ? 16u : v17;
    {
        auto tX = tensor((device half *)b2 + v12 * 3584u, dextents<int32_t, 2>(3584, int(v16)), array<int, 2>({1, 3584}));
        auto tW = tensor((device half *)b1 + v13 * 3584u, dextents<int32_t, 2>(3584, int(v19)), array<int, 2>({1, 3584}));
        mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(32, 16, static_cast<int>(dynamic_extent), false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<4>> mm;
        auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
        mm.run(tX, tW, cT);
        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(s0, dextents<int32_t, 2>(16, 32));
        cT.store(tC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint v23 = lid; v23 < 512u; v23 += 128u) {
        uint v24 = v23 / 16u;
        uint v25 = v23 % 16u;
        bool v26 = v24 < v16;
        bool v27 = v25 < v19;
        bool v28 = v26 && v27;
        if (v28) {
            float v29 = s0[v23];
            uint v30 = v12 + v24;
            uint v31 = v13 + v25;
            uint v32 = v30 * 1024u;
            uint v33 = v32 + v31;
            float v34 = float(half(v29));
            uint v35 = uint(b3[v33]);
            float v36 = float(as_type<half>(ushort(v35)));
            float v37 = v36 + v34;
            uint v38 = uint(as_type<ushort>(half(v37)));
            b3[v33] = ushort(v38);
        }
    }
}
