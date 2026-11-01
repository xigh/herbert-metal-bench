#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void gemm_nax_2048x1024_res_128x128_b0(
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
    const uint m = b0[3];
    const uint mb = (m + 127u) / 128u;
    const uint gm = tgid.x % mb, gn = tgid.x / mb;
    const ushort sgm = ushort(sgid) / 4, sgn = ushort(sgid) % 4;
    const ushort qid = ushort(lane) >> 2;
    const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);
    const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;
    const uint m0 = gm * 128u + sgm * 32u, n0 = gn * 128u + sgn * 32u;
    typedef vec<half, 8> v8; typedef vec<half, 4> v4; typedef vec<float, 8> f8;
    const device half *xa00 = (const device half *)b2 + min(m0 + 0u + fm, m - 1u) * 2048u + fn;
    const device half *wb00 = (const device half *)b1 + min(n0 + 0u + fm, 1023u) * 2048u + fn;
    const device half *xa01 = (const device half *)b2 + min(m0 + 8u + fm, m - 1u) * 2048u + fn;
    const device half *wb01 = (const device half *)b1 + min(n0 + 8u + fm, 1023u) * 2048u + fn;
    const device half *xa10 = (const device half *)b2 + min(m0 + 16u + fm, m - 1u) * 2048u + fn;
    const device half *wb10 = (const device half *)b1 + min(n0 + 16u + fm, 1023u) * 2048u + fn;
    const device half *xa11 = (const device half *)b2 + min(m0 + 24u + fm, m - 1u) * 2048u + fn;
    const device half *wb11 = (const device half *)b1 + min(n0 + 24u + fm, 1023u) * 2048u + fn;
    f8 d00 = f8(0.0f), d01 = f8(0.0f), d10 = f8(0.0f), d11 = f8(0.0f);
    for (uint k0 = 0; k0 < 2048u; k0 += 32u) {
        v8 a00; { v4 x0 = *(const device v4 *)(xa00 + k0 + 0u); v4 x1 = *(const device v4 *)(xa01 + k0 + 0u); a00 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 a01; { v4 x0 = *(const device v4 *)(xa00 + k0 + 16u); v4 x1 = *(const device v4 *)(xa01 + k0 + 16u); a01 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 a10; { v4 x0 = *(const device v4 *)(xa10 + k0 + 0u); v4 x1 = *(const device v4 *)(xa11 + k0 + 0u); a10 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 a11; { v4 x0 = *(const device v4 *)(xa10 + k0 + 16u); v4 x1 = *(const device v4 *)(xa11 + k0 + 16u); a11 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 b00; { v4 x0 = *(const device v4 *)(wb00 + k0 + 0u); v4 x1 = *(const device v4 *)(wb01 + k0 + 0u); b00 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 b01; { v4 x0 = *(const device v4 *)(wb00 + k0 + 16u); v4 x1 = *(const device v4 *)(wb01 + k0 + 16u); b01 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 b10; { v4 x0 = *(const device v4 *)(wb10 + k0 + 0u); v4 x1 = *(const device v4 *)(wb11 + k0 + 0u); b10 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 b11; { v4 x0 = *(const device v4 *)(wb10 + k0 + 16u); v4 x1 = *(const device v4 *)(wb11 + k0 + 16u); b11 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = a00[e]; cb[e] = b00[e]; cb[8 + e] = b10[e]; cc[e] = d00[e]; cc[8 + e] = d01[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { d00[e] = cc[e]; d01[e] = cc[8 + e]; }
        }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = a01[e]; cb[e] = b01[e]; cb[8 + e] = b11[e]; cc[e] = d00[e]; cc[8 + e] = d01[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { d00[e] = cc[e]; d01[e] = cc[8 + e]; }
        }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = a10[e]; cb[e] = b00[e]; cb[8 + e] = b10[e]; cc[e] = d10[e]; cc[8 + e] = d11[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { d10[e] = cc[e]; d11[e] = cc[8 + e]; }
        }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = a11[e]; cb[e] = b01[e]; cb[8 + e] = b11[e]; cc[e] = d10[e]; cc[8 + e] = d11[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { d10[e] = cc[e]; d11[e] = cc[8 + e]; }
        }
    }
    for (ushort i = 0; i < 2; ++i) { const uint r = m0 + 0u + fm + 8u * i; if (r < m) for (ushort j = 0; j < 4; ++j) {
        const uint c = n0 + 0u + fn + j;
        if (c < 1024u) { const float acc = d00[i * 4 + j]; const uint at = r * 1024u + c; { const float rr = float(as_type<half>(ushort(uint(as_type<ushort>(half(acc)))))); const float of = float(as_type<half>(ushort(b3[at]))); b3[at] = ushort(uint(as_type<ushort>(half(of + rr)))); } }
    } }
    for (ushort i = 0; i < 2; ++i) { const uint r = m0 + 0u + fm + 8u * i; if (r < m) for (ushort j = 0; j < 4; ++j) {
        const uint c = n0 + 16u + fn + j;
        if (c < 1024u) { const float acc = d01[i * 4 + j]; const uint at = r * 1024u + c; { const float rr = float(as_type<half>(ushort(uint(as_type<ushort>(half(acc)))))); const float of = float(as_type<half>(ushort(b3[at]))); b3[at] = ushort(uint(as_type<ushort>(half(of + rr)))); } }
    } }
    for (ushort i = 0; i < 2; ++i) { const uint r = m0 + 16u + fm + 8u * i; if (r < m) for (ushort j = 0; j < 4; ++j) {
        const uint c = n0 + 0u + fn + j;
        if (c < 1024u) { const float acc = d10[i * 4 + j]; const uint at = r * 1024u + c; { const float rr = float(as_type<half>(ushort(uint(as_type<ushort>(half(acc)))))); const float of = float(as_type<half>(ushort(b3[at]))); b3[at] = ushort(uint(as_type<ushort>(half(of + rr)))); } }
    } }
    for (ushort i = 0; i < 2; ++i) { const uint r = m0 + 16u + fm + 8u * i; if (r < m) for (ushort j = 0; j < 4; ++j) {
        const uint c = n0 + 16u + fn + j;
        if (c < 1024u) { const float acc = d11[i * 4 + j]; const uint at = r * 1024u + c; { const float rr = float(as_type<half>(ushort(uint(as_type<ushort>(half(acc)))))); const float of = float(as_type<half>(ushort(b3[at]))); b3[at] = ushort(uint(as_type<ushort>(half(of + rr)))); } }
    } }
}
