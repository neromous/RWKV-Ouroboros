#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#define MIN_VALUE (-1e38)
typedef at::BFloat16 bf16;

__global__ void kernel_forward(const int B, const int T, const int C,
                               const float *__restrict__ const _w, const bf16 *__restrict__ const _u, const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v,  const float *__restrict__ const last_state,
                               bf16 *__restrict__ const _y, float *__restrict__ const new_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int state_offset = (_b * C + _c)*3;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    bf16 *__restrict__ const y = _y + _offset;

    float aa, bb, pp;
    if (last_state == NULL) {
        aa = 0, bb = 0, pp = MIN_VALUE;
    } else {
        aa = last_state[state_offset+0];
        bb = last_state[state_offset+1];
        pp = last_state[state_offset+2];
    }
    // aa and bb are running sums divided by exp(pp) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = bf16((e1 * aa + e2 * vv) / (e1 * bb + e2));

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    if (new_state != NULL) {
        new_state[state_offset+0] = aa;
        new_state[state_offset+1] = bb;
        new_state[state_offset+2] = pp;
    }
}

__global__ void kernel_backward(const int B, const int T, const int C,
                                const float *__restrict__ const _w, const bf16 *__restrict__ const _u, const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v, const float *__restrict__ const last_state,
                                const bf16 *__restrict__ const _y, const bf16 *__restrict__ const _gy, const float *__restrict__ const gnew_state,
                                bf16 *__restrict__ const _gw, bf16 *__restrict__ const _gu, bf16 *__restrict__ const _gk, bf16 *__restrict__ const _gv, float *__restrict__ const glast_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int state_offset  = (_b * C + _c)*3;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    const bf16 *__restrict__ const y = _y + _offset;
    const bf16 *__restrict__ const gy = _gy + _offset;
    bf16 *__restrict__ const gk = _gk + _offset;
    bf16 *__restrict__ const gv = _gv + _offset;


    float q[Tmax], r[Tmax];

    float gw = 0, gu = 0, ga = 0, gb = 0;
    float aa, bb, pp;
    if (last_state == NULL) {
        aa = 0, bb = 0, pp = MIN_VALUE;
    } else {
        aa = last_state[state_offset+0];
        bb = last_state[state_offset+1];
        pp = last_state[state_offset+2];
    }
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float yy = float(y[ii]);

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        const float qq = float(gy[ii]) / (e1 * bb + e2);
        gw += (ga - gb * yy) * e1 * qq;
        gu += (vv - yy) * e2 * qq;
        q[i] = qq;
        r[i] = ww - p;

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        ga = e1 * (aa + ga);
        gb = e1 * (bb + gb);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    //For clearity, I use gaa, gbb, gpp here.
    //Note that gaa, gbb are NOT actual gradient of aa, bb
    //In fact, gaa*exp(gpp) is the actual gradient of original a (i.e., aa*exp(pp)).
    //And gaa*exp(pp+gpp) is the actual gradient of aa.
    //In brief, gaa, gbb, gpp are NOT gradients. DO NOT use them directly update initial state.
    float gaa = 0, gbb = 0, gpp = MIN_VALUE; 
    if (gnew_state != NULL) {
        gaa = gnew_state[state_offset+0];
        gbb = gnew_state[state_offset+1];
        gpp = gnew_state[state_offset+2];
        if (gaa == 0 && gbb == 0) gpp = MIN_VALUE;
        gw += (gaa * ga + gbb * gb) * exp(pp+gpp);
    }

    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] = bf16(gw * _w[_c]); // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = bf16(gu);
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float yy = float(y[ii]);
        const float qq = q[i];
        const float rr = r[i];

        float e1 = qq * exp(rr);
        float e2 = exp(kk + gpp);
        gk[ii] = bf16(e1 * (vv - yy) + e2 * (gaa * vv + gbb));
        gv[ii] = bf16(e1 + e2 * gaa);

        const float ww = w + gpp;
        const float www = rr - u - kk;
        const float p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        gaa = e1 * gaa + e2;
        gbb = e1 * gbb - e2 * yy;    
        gpp = p;
    }

    // glast_state[2] is not the gradient w.r.t of last_state[2]
    // pp (index 2) in last_state is just an exponent for aa and bb
    // so there are really only 2 elements to differentiate on
    // Similary gpp (glast_state index 2) is just an exponent for gaa and gbb
    if (glast_state != NULL) {
        glast_state[state_offset+0] = gaa;
        glast_state[state_offset+1] = gbb;
        glast_state[state_offset+2] = gpp;
    }

}

void cuda_forward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, float *last_state, 
    bf16 *y, float *new_state) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, last_state, y, new_state);
}

void cuda_backward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, float *last_state, 
    bf16 *y, bf16 *gy, float *gnew_state, 
    bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv, float *glast_state) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, last_state, y, gy, gnew_state, gw, gu, gk, gv, glast_state);
}

