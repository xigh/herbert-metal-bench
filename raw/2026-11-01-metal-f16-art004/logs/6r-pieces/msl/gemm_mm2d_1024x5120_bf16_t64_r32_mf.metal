#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void gemm_mm2d_1024x5120_bf16_t64_r32_mf(
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
    threadgroup float s0[2048];
    uint v1 = b0[3u];
    uint v5 = v1 + 31u;
    uint v6 = v5 / 32u;
    uint v7 = tgid.x % v6;
    uint v8 = tgid.x / v6;
    uint v12 = v7 * 32u;
    uint v13 = v8 * 64u;
    uint v14 = v1 - v12;
    bool v15 = 32u < v14;
    uint v16 = v15 ? 32u : v14;
    uint v17 = 5120u - v13;
    bool v18 = 64u < v17;
    uint v19 = v18 ? 64u : v17;
    {
        auto tX = tensor((device half *)b2 + v12 * 1024u, dextents<int32_t, 2>(1024, int(v16)), array<int, 2>({1, 1024}));
        auto tW = tensor((device half *)b1 + v13 * 1024u, dextents<int32_t, 2>(1024, int(v19)), array<int, 2>({1, 1024}));
        mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(32, 64, static_cast<int>(dynamic_extent), false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<4>> mm;
        auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
        mm.run(tX, tW, cT);
        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(s0, dextents<int32_t, 2>(64, 32));
        cT.store(tC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint v23 = lid; v23 < 2048u; v23 += 128u) {
        uint v24 = v23 / 64u;
        uint v25 = v23 % 64u;
        bool v26 = v24 < v16;
        bool v27 = v25 < v19;
        bool v28 = v26 && v27;
        if (v28) {
            float v29 = s0[v23];
            uint v30 = v12 + v24;
            uint v31 = v13 + v25;
            uint v32 = v30 * 5120u;
            uint v33 = v32 + v31;
            uint v34 = uint(as_type<ushort>(half(v29)));
            b3[v33] = ushort(v34);
        }
    }
}
