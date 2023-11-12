#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#define MIN_VALUE (-1e38)
typedef at::BFloat16 bf16;

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), file, line);
		exit(int(err));
	}
}

__global__ void kernel_forward(const int B, const int T, const int C,
                               const float *__restrict__ const _w, const bf16 *__restrict__ const _u, const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v,
                               bf16 *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    bf16 *__restrict__ const y = _y + _offset;

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    float aa = 0, bb = 0, pp = MIN_VALUE;
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
}

__global__ void kernel_backward(const int B, const int T, const int C,
                                const float *__restrict__ const _w, const bf16 *__restrict__ const _u, const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v,
                                const bf16 *__restrict__ const _y, const bf16 *__restrict__ const _gy,
                                bf16 *__restrict__ const _gw, bf16 *__restrict__ const _gu, bf16 *__restrict__ const _gk, bf16 *__restrict__ const _gv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    const bf16 *__restrict__ const y = _y + _offset;
    const bf16 *__restrict__ const gy = _gy + _offset;
    bf16 *__restrict__ const gk = _gk + _offset;
    bf16 *__restrict__ const gv = _gv + _offset;

    float q[Tmax], r[Tmax];

    float gw = 0, gu = 0, aa = 0, bb = 0, ga = 0, gb = 0, pp = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float yy = float(y[ii]);

        float ww = u + kk; //应该不会炸
        float p = max(pp, ww); //pp、ww不炸则肯定不会炸，pp已经炸了会传播过来
        float e1 = exp(pp - p); //减去较大值，所以指数必然小于等于0，e1、e2小于等于1，且必然有一个为1，所以这两行都不会炸
        float e2 = exp(ww - p);//不会炸
        const float qq = float(gy[ii]) / (e1 * bb + e2);//假如e1等于1，e2和bb都特别小时会炸，并且这一行似乎是唯一一个可能出错的行
        gw += (ga - gb * yy) * e1 * qq; //取决于qq炸不炸
        // gw = min(gw, 1e2);
        // gw = max(gw, -1e2);
        // gw = min(gw,MAX_gw); //grad clip
        // gw = max(gw,MIN_gw);
        gu += (vv - yy) * e2 * qq; //取决于qq炸不炸
        // gu = min(gu, 1e2);
        // gu = max(gu, -1e2);
        // gu = min(gu,MAX_gu);
        // gu = max(gu,MIN_gu);
        q[i] = qq;
        r[i] = ww - p;

        ww = w + pp; //会不会炸取决于pp大小
        p = max(ww, kk);
        e1 = exp(ww - p); //e1、e2小于等于1，道理同上
        e2 = exp(kk - p);
        ga = e1 * (aa + ga);
        gb = e1 * (bb + gb);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
        // pp = min(pp, 1e2);
        // pp = max(pp, -1e2);
    }
    const int _offsetBC = _b * C + _c;
    // assert(gw == 0);
    _gw[_offsetBC] = bf16(gw * _w[_c]); // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = bf16(gu);

    aa = 0, bb = 0, pp = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float yy = float(y[ii]);
        const float qq = q[i];
        const float rr = r[i];

        // float e1 = 1;
        // float e2 = 1;
        float e1 = qq * exp(rr);
        float e2 = exp(kk + pp);
        // bf16 gk_temp = bf16(e1 * (vv - yy) + e2 * (aa * vv + bb));
        // bf16 gv_temp = bf16(e1 + e2 * aa);
        // gk[ii] = gk_temp;
        // gv[ii] = gv_temp;
        gk[ii] = (e1 * (vv - yy) + e2 * (aa * vv + bb));
        gv[ii] = (e1 + e2 * aa);
        // gk[ii] = bf16(e1 * (vv - yy) + e2 * (aa * vv + bb));
        // gv[ii] = bf16(e1 + e2 * aa);
        // gk[ii] = 1.14514;
        // gv[ii] = 1.14514;
        // gk[ii] = bf16(qq);
        // gv[ii] = bf16(rr);

        const float ww = w + pp;
        const float www = rr - u - kk;
        const float p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        aa = e1 * aa + e2;
        bb = e1 * bb - e2 * yy;
        pp = p;
        // gk[ii] = bf16(qq);
        // gv[ii] = bf16(p);
    }
}
//注释法调试记录
//1. 只保留_gw、_gu赋值，删除两个循环和q、r数组初始化，似乎没有出错。_gw能正确赋值
//2. 恢复q、r初始化，能正确赋值
//3. 恢复gw、gu的计算循环，能正确赋值，看来问题在第二个循环中
//4. 砍掉第二个循环后半部分（ww=w+pp及之后），不正确赋值，看来在靠前部分。
//5. 定位到gk[ii]、gv[ii]赋值部分，注意，只计算不赋值没有出错（存疑）。只赋字面值也不出错，只要等于一个变量就出错
//6. 现在定位看看等于哪些变量会出错：e1会出错，e2不会. qq和rr一并输出时会，
//7. 奇怪，为啥qq、rr一起输出会，不一起输出时不会？

void cuda_forward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

void cuda_backward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv);
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("Fuck cuda\n");
    
}
