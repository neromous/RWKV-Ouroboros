#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;
#define MM8_ONE_JSPLIT 16
#define MM8_ONE_TILE 256
#define EMBSPLIT 256
#define EMBBLOCK 16

//---------------
//
// Optimized mm8 operations
//
//---------------

template <typename DTYPE>
__global__ void kernelc_mm8_one(
    const unsigned long long N, const unsigned long long M,
    const DTYPE *__restrict__ const x,
    const uint8_t *__restrict__ const w, const unsigned long long w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const unsigned long long offset,
    unsigned long long tokenlength)
{

    for (unsigned long long token = 0; token < tokenlength; token++)
    {
        const unsigned long long k = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned long long j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
        const unsigned long long j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

        if (k < M)
        {
            float y_local = 0;
            for (unsigned long long j = j0; j < j1; ++j)
            {
                y_local += float(x[j + N * token]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
            }
            atomicAdd(reinterpret_cast<float *>(&y[k + M * token]), *reinterpret_cast<float *>(&y_local));
        }
    }
}

void cudac_mm8_one(unsigned long long N, unsigned long long M,
                   float *x,
                   uint8_t *w, unsigned long long w_stride,
                   float *y,
                   float *r,
                   float *o,
                   unsigned long long offset,
                     unsigned long long tokenlength)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset, tokenlength);
}

/*
template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_] = {0};

    __syncthreads();
    u[i] = float(_u[i]);
    w[i] = float(_w[i]);
    __syncthreads();

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);
    }
}
*/

//---------------
//
// Main forward kernels
//
//---------------

template <typename F>
__global__ void kernel_forward_inference(
    const int B, const int T, const int C, const int H,
    float *__restrict__ _state,
    const F *__restrict__ const _r, const F *__restrict__ const _k,
    const F *__restrict__ const _v, const float *__restrict__ _w,
    const F *__restrict__ _u, F *__restrict__ _y
) {
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;
    _state += h*_N_*_N_ + i*_N_; // wrong if B > 1 !!!

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];

    float state[_N_];
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        state[j] = _state[j];

    __syncthreads();
    u[i] = float(_u[i]);
    w[i] = _w[i];
    __syncthreads();

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);
    }

    __syncthreads();
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];
    __syncthreads();
}

//---------------
//
// Main backwards kernels
//
//---------------

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H, float *__restrict__ _state,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const float *__restrict__ __w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gw, F *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;
    __w += h*_N_;

    __shared__ float w_[_N_], u_[_N_];
    __shared__ float r[_N_], k[_N_], v[_N_], gy[_N_];
    __syncthreads();
    w_[i] = _w[i];
    u_[i] = float(_u[i]);
    __syncthreads();

    const float w = w_[i];
    const float ww = __w[i];
    const float u = u_[i];

    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    #pragma unroll
    for (int j = 0; j < _N_; j++) {
        state[j] = _state[j];
        // saaaa[j] = _state[j];
        // sbbbb[j] = _state[j];
        // scccc[j] = _state[j];
        // sdddd[j] = _state[j];
    }

    float gw = 0, gu = 0;
    const int t000 = b*T*C + h*_N_ + i;
    const int t111 = (b+1)*T*C + h*_N_ + i;
    const int t222 = t111 - 2*C;

    for (int t = t000; t < t111; t += C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float k = float(_k[t]);
        float gr = 0, gu_ = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k * v[j];

            gr += (u * x + s) * gy[j];
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = F(gr);
        gu += float(_r[t]) * gu_;
    }
    _gu[b*C + h*_N_ + i] = F(gu);

    for (int t = t000; t < t222; t += C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t + 2*C]);
        __syncthreads();

        const float k = float(_k[t]);
        float gw_ = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float& s2 = sbbbb[j];
            float x = k * v[j];

            float tmp = w * (x + s);
            s = tmp;
            s2 = tmp + w * s2;
            gw_ += s2 * gy[j];
        }
        gw += float(_r[t + 2*C]) * gw_;
    }
    _gw[b*C + h*_N_ + i] = F(ww * gw);

    for (int t = t111 - C; t >= t000; t -= C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float rr = float(_r[t]);
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = rr * gy[j];

            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = F(gk);
    }

    for (int t = t111 - C; t >= t000; t -= C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float gyy = float(_gy[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = sdddd[j];
            float x = gyy * r[j];

            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = F(gv);
    }
}

//---------------
//
// Forward / backward type aliases
//
//---------------

void cuda_forward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    kernel_forward_inference<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, k, v, w, u, y);
}
void cuda_forward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *k, fp16 *v, float *w, fp16 *u, fp16 *y)
{
    assert(H*_N_ == C);
    kernel_forward_inference<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, k, v, w, u, y);
}
void cuda_forward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *k, fp32 *v, float *w, fp32 *u, fp32 *y)
{
    assert(H*_N_ == C);
    kernel_forward_inference<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, k, v, w, u, y);
}

void cuda_backward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}

void cuda_backward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *k, fp16 *v, float *w, float *ww, fp16 *u, fp16 *gy, fp16 *gr, fp16 *gk, fp16 *gv, fp16 *gw, fp16 *gu)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}

void cuda_backward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *k, fp32 *v, float *w, float *ww, fp32 *u, fp32 *gy, fp32 *gr, fp32 *gk, fp32 *gv, fp32 *gw, fp32 *gu)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}