#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void sdpa_nax_h8kv2d256(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device uint4 *b2 [[buffer(2)]],
    device uint4 *b3 [[buffer(3)]],
    device ushort *b4 [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float xs[4096];
    const uint m = b0[3];
    const uint p0 = b0[1];
    const uint h = tgid.x % 8u;
    const uint r0 = (tgid.x / 8u) * 64u;
    const uint kvh = h / 4u;
    const ushort rg = ushort(sgid) / 2, dg = ushort(sgid) % 2;
    const ushort qid = ushort(lane) >> 2;
    const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);
    const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;
    const uint row0 = r0 + rg * 16u + fm;
    const uint dcol = dg * 128u + fn;
    const device half *qh = (const device half *)b1;
    const device half *kh = (const device half *)b2 + kvh * 524288u;
    const device half *vh = (const device half *)b3 + kvh * 524288u;
    typedef vec<half, 8> v8; typedef vec<half, 4> v4;
    v8 q[8];
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 0u) : v4(0); q[0][0] = x[0]; q[0][1] = x[1]; q[0][2] = x[2]; q[0][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 0u) : v4(0); q[0][4] = x[0]; q[0][5] = x[1]; q[0][6] = x[2]; q[0][7] = x[3]; }
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 16u) : v4(0); q[1][0] = x[0]; q[1][1] = x[1]; q[1][2] = x[2]; q[1][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 16u) : v4(0); q[1][4] = x[0]; q[1][5] = x[1]; q[1][6] = x[2]; q[1][7] = x[3]; }
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 32u) : v4(0); q[2][0] = x[0]; q[2][1] = x[1]; q[2][2] = x[2]; q[2][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 32u) : v4(0); q[2][4] = x[0]; q[2][5] = x[1]; q[2][6] = x[2]; q[2][7] = x[3]; }
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 48u) : v4(0); q[3][0] = x[0]; q[3][1] = x[1]; q[3][2] = x[2]; q[3][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 48u) : v4(0); q[3][4] = x[0]; q[3][5] = x[1]; q[3][6] = x[2]; q[3][7] = x[3]; }
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 64u) : v4(0); q[4][0] = x[0]; q[4][1] = x[1]; q[4][2] = x[2]; q[4][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 64u) : v4(0); q[4][4] = x[0]; q[4][5] = x[1]; q[4][6] = x[2]; q[4][7] = x[3]; }
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 80u) : v4(0); q[5][0] = x[0]; q[5][1] = x[1]; q[5][2] = x[2]; q[5][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 80u) : v4(0); q[5][4] = x[0]; q[5][5] = x[1]; q[5][6] = x[2]; q[5][7] = x[3]; }
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 96u) : v4(0); q[6][0] = x[0]; q[6][1] = x[1]; q[6][2] = x[2]; q[6][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 96u) : v4(0); q[6][4] = x[0]; q[6][5] = x[1]; q[6][6] = x[2]; q[6][7] = x[3]; }
    { const uint r = row0 + 0u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 112u) : v4(0); q[7][0] = x[0]; q[7][1] = x[1]; q[7][2] = x[2]; q[7][3] = x[3]; }
    { const uint r = row0 + 8u; v4 x = r < m ? *(const device v4 *)(qh + r * 5120u + h * 512u + dcol + 112u) : v4(0); q[7][4] = x[0]; q[7][5] = x[1]; q[7][6] = x[2]; q[7][7] = x[3]; }
    typedef vec<float, 8> f8; f8 o[8];
    for (ushort i = 0; i < 8; ++i) o[i] = f8(0.0f);
    float mx0 = -FLT_MAX, mx1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;
    const uint qp0 = p0 + row0, qp1 = qp0 + 8u;
    const uint end = p0 + min(r0 + 64u, m);
    for (uint k0 = 0; k0 < end; k0 += 32u) {
        f8 s0 = f8(0.0f), s1 = f8(0.0f);
        v8 ka0; { const device half *p = kh + (k0 + fm) * 256u + dcol + 0u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb0; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 0u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[0][e]; cb[e] = ka0[e]; cb[8 + e] = kb0[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        v8 ka1; { const device half *p = kh + (k0 + fm) * 256u + dcol + 16u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb1; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 16u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[1][e]; cb[e] = ka1[e]; cb[8 + e] = kb1[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        v8 ka2; { const device half *p = kh + (k0 + fm) * 256u + dcol + 32u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka2 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb2; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 32u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb2 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[2][e]; cb[e] = ka2[e]; cb[8 + e] = kb2[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        v8 ka3; { const device half *p = kh + (k0 + fm) * 256u + dcol + 48u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka3 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb3; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 48u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb3 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[3][e]; cb[e] = ka3[e]; cb[8 + e] = kb3[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        v8 ka4; { const device half *p = kh + (k0 + fm) * 256u + dcol + 64u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka4 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb4; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 64u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb4 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[4][e]; cb[e] = ka4[e]; cb[8 + e] = kb4[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        v8 ka5; { const device half *p = kh + (k0 + fm) * 256u + dcol + 80u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka5 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb5; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 80u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb5 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[5][e]; cb[e] = ka5[e]; cb[8 + e] = kb5[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        v8 ka6; { const device half *p = kh + (k0 + fm) * 256u + dcol + 96u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka6 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb6; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 96u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb6 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[6][e]; cb[e] = ka6[e]; cb[8 + e] = kb6[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        v8 ka7; { const device half *p = kh + (k0 + fm) * 256u + dcol + 112u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); ka7 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 kb7; { const device half *p = kh + (k0 + 16u + fm) * 256u + dcol + 112u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); kb7 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = q[7][e]; cb[e] = ka7[e]; cb[8 + e] = kb7[e]; cc[e] = s0[e]; cc[8 + e] = s1[e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { s0[e] = cc[e]; s1[e] = cc[8 + e]; }
        }
        threadgroup float *slot = xs + (rg * 2u + dg) * 512u + ushort(lane) * 16u;
        for (ushort e = 0; e < 8; ++e) { slot[e] = s0[e]; slot[8 + e] = s1[e]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *peer = xs + rg * 1024u + ushort(lane) * 16u;
        for (ushort e = 0; e < 8; ++e) { s0[e] = peer[e]; s1[e] = peer[8 + e]; }
        for (ushort pg = 1; pg < 2; ++pg) for (ushort e = 0; e < 8; ++e) { s0[e] += peer[pg * 512u + e]; s1[e] += peer[pg * 512u + 8u + e]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sc = 0.0625f;
        float lm0 = -FLT_MAX, lm1 = -FLT_MAX;
        for (ushort e = 0; e < 8; ++e) {
            const uint kp = k0 + fn + (e & 3);
            const uint qp = e < 4 ? qp0 : qp1;
            s0[e] = kp <= qp ? s0[e] * sc : -FLT_MAX;
            s1[e] = kp + 16u <= qp ? s1[e] * sc : -FLT_MAX;
            if (e < 4) lm0 = fmax(lm0, fmax(s0[e], s1[e])); else lm1 = fmax(lm1, fmax(s0[e], s1[e]));
        }
        lm0 = fmax(lm0, simd_shuffle_xor(lm0, 1)); lm0 = fmax(lm0, simd_shuffle_xor(lm0, 8));
        lm1 = fmax(lm1, simd_shuffle_xor(lm1, 1)); lm1 = fmax(lm1, simd_shuffle_xor(lm1, 8));
        const float mn0 = fmax(mx0, lm0), mn1 = fmax(mx1, lm1);
        const float c0 = exp(mx0 - mn0), c1 = exp(mx1 - mn1);
        float ls0 = 0.0f, ls1 = 0.0f;
        v8 p0f, p1f;
        for (ushort e = 0; e < 8; ++e) {
            const float mn = e < 4 ? mn0 : mn1;
            const float a = exp(s0[e] - mn), b = exp(s1[e] - mn);
            if (e < 4) ls0 += a + b; else ls1 += a + b;
            p0f[e] = half(a); p1f[e] = half(b);
        }
        ls0 += simd_shuffle_xor(ls0, 1); ls0 += simd_shuffle_xor(ls0, 8);
        ls1 += simd_shuffle_xor(ls1, 1); ls1 += simd_shuffle_xor(ls1, 8);
        l0 = fma(l0, c0, ls0); l1 = fma(l1, c1, ls1);
        mx0 = mn0; mx1 = mn1;
        if (!simd_all(c0 == 1.0f && c1 == 1.0f)) {
            for (ushort i = 0; i < 8; ++i) for (ushort e = 0; e < 8; ++e) o[i][e] *= e < 4 ? c0 : c1;
        }
        v8 va0_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 0u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va0_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb0_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 16u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb0_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p0f[e]; cb[e] = va0_0[e]; cb[8 + e] = vb0_0[e]; cc[e] = o[0][e]; cc[8 + e] = o[1][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[0][e] = cc[e]; o[1][e] = cc[8 + e]; }
        }
        v8 va0_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 0u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va0_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb0_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 16u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb0_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p1f[e]; cb[e] = va0_1[e]; cb[8 + e] = vb0_1[e]; cc[e] = o[0][e]; cc[8 + e] = o[1][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[0][e] = cc[e]; o[1][e] = cc[8 + e]; }
        }
        v8 va2_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 32u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va2_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb2_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 48u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb2_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p0f[e]; cb[e] = va2_0[e]; cb[8 + e] = vb2_0[e]; cc[e] = o[2][e]; cc[8 + e] = o[3][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[2][e] = cc[e]; o[3][e] = cc[8 + e]; }
        }
        v8 va2_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 32u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va2_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb2_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 48u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb2_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p1f[e]; cb[e] = va2_1[e]; cb[8 + e] = vb2_1[e]; cc[e] = o[2][e]; cc[8 + e] = o[3][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[2][e] = cc[e]; o[3][e] = cc[8 + e]; }
        }
        v8 va4_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 64u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va4_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb4_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 80u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb4_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p0f[e]; cb[e] = va4_0[e]; cb[8 + e] = vb4_0[e]; cc[e] = o[4][e]; cc[8 + e] = o[5][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[4][e] = cc[e]; o[5][e] = cc[8 + e]; }
        }
        v8 va4_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 64u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va4_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb4_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 80u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb4_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p1f[e]; cb[e] = va4_1[e]; cb[8 + e] = vb4_1[e]; cc[e] = o[4][e]; cc[8 + e] = o[5][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[4][e] = cc[e]; o[5][e] = cc[8 + e]; }
        }
        v8 va6_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 96u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va6_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb6_0; { const device half *p = vh + (k0 + 0u + fm) * 256u + dcol + 112u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb6_0 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p0f[e]; cb[e] = va6_0[e]; cb[8 + e] = vb6_0[e]; cc[e] = o[6][e]; cc[8 + e] = o[7][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[6][e] = cc[e]; o[7][e] = cc[8 + e]; }
        }
        v8 va6_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 96u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); va6_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        v8 vb6_1; { const device half *p = vh + (k0 + 16u + fm) * 256u + dcol + 112u; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + 2048u); vb6_1 = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }
        {
            mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<half, half, float>();
            auto cb = op.get_right_input_cooperative_tensor<half, half, float>();
            auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { ca[e] = p1f[e]; cb[e] = va6_1[e]; cb[8 + e] = vb6_1[e]; cc[e] = o[6][e]; cc[8 + e] = o[7][e]; }
            op.run(ca, cb, cc);
            #pragma clang loop unroll(full)
            for (ushort e = 0; e < 8; ++e) { o[6][e] = cc[e]; o[7][e] = cc[8 + e]; }
        }
    }
    device half *oh = (device half *)b4;
    { const uint r = row0 + 0u; if (r < m) { const float l = l0;
        for (ushort id = 0; id < 8; ++id) for (ushort j = 0; j < 4; ++j) {
            const uint d = dcol + id * 16u + j;
            const float ov = float(half(o[id][0 + j] / l));
            const float gv = float(qh[r * 5120u + h * 512u + 256u + d]);
            const float sg = 1.0f / (1.0f + exp(-gv));
            oh[r * 2048u + h * 256u + d] = half(ov * sg);
        }
    } }
    { const uint r = row0 + 8u; if (r < m) { const float l = l1;
        for (ushort id = 0; id < 8; ++id) for (ushort j = 0; j < 4; ++j) {
            const uint d = dcol + id * 16u + j;
            const float ov = float(half(o[id][4 + j] / l));
            const float gv = float(qh[r * 5120u + h * 512u + 256u + d]);
            const float sg = 1.0f / (1.0f + exp(-gv));
            oh[r * 2048u + h * 256u + d] = half(ov * sg);
        }
    } }
}
