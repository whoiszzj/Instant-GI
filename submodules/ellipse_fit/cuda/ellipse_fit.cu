#include "ellipse_fit.h"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <float.h>

__device__ void solve_linear_6x6(float S[6][6], float a[6]) {
    // 高斯消元法，约束 a[5]=1
    // S * a = 0, a[5]=1
    float A[6][7];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
            A[i][j] = S[i][j];
        A[i][6] = 0.0f;
    }
    // 约束 a[5]=1
    for (int j = 0; j < 6; ++j)
        A[5][j] = 0.0f;
    A[5][5] = 1.0f;
    A[5][6] = 1.0f;

    // 高斯消元
    for (int i = 0; i < 6; ++i) {
        // 主元归一化
        float pivot = A[i][i];
        if (fabsf(pivot) < 1e-8) continue;
        for (int j = i; j < 7; ++j)
            A[i][j] /= pivot;
        // 消元
        for (int k = 0; k < 6; ++k) {
            if (k == i) continue;
            float factor = A[k][i];
            for (int j = i; j < 7; ++j)
                A[k][j] -= factor * A[i][j];
        }
    }
    for (int i = 0; i < 6; ++i)
        a[i] = A[i][6];
}

__device__ void ellipse_params_from_a(const float a[6], float& cx, float& cy, float& A, float& B, float& angle) {
    // 椭圆一般式: a0*x^2 + a1*x*y + a2*y^2 + a3*x + a4*y + a5 = 0
    // 解析出中心(cx, cy)、长短轴A/B、旋转角angle
    float b = a[1], c = a[2], d = a[3], f = a[4], g = a[5], a0 = a[0];

    float num = b*b - 4*a0*c;
    if (fabsf(num) < 1e-8) { cx = cy = A = B = angle = 0; return; }

    cx = (2*c*d - b*f) / num;
    cy = (2*a0*f - b*d) / num;

    float up = 2*(a0*f*f + c*d*d + g*b*b - 2*b*d*f - a0*c*g);
    float down1 = (b*b - 4*a0*c) * ((c - a0)*sqrtf(1 + 4*b*b/((a0-c)*(a0-c))) - (c + a0));
    float down2 = (b*b - 4*a0*c) * ((a0 - c)*sqrtf(1 + 4*b*b/((a0-c)*(a0-c))) - (c + a0));
    if (down1 == 0 || down2 == 0) { cx = cy = A = B = angle = 0; return; }
    A = sqrtf(fabsf(up / down1));
    B = sqrtf(fabsf(up / down2));

    angle = 0.5f * atanf(b / (a0 - c));
    // 角度转为度
    angle = angle * 180.0f / 3.14159265358979323846f;
}

__device__ void swapf(float& a, float& b) { float t = a; a = b; b = t; }

// 简单2x2线性方程组求解
__device__ void solve2x2(float A[2][2], float b[2], float x[2]) {
    float det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    if (fabsf(det) < 1e-8) { x[0]=x[1]=0; return; }
    x[0] = (b[0]*A[1][1] - b[1]*A[0][1]) / det;
    x[1] = (A[0][0]*b[1] - A[1][0]*b[0]) / det;
}

// 简单3x3线性方程组求解（高斯消元）
__device__ void solve3x3(float A[3][3], float b[3], float x[3]) {
    float M[3][4];
    for(int i=0;i<3;++i) {
        for(int j=0;j<3;++j) M[i][j]=A[i][j];
        M[i][3]=b[i];
    }
    // 消元
    for(int i=0;i<3;++i) {
        float pivot = M[i][i];
        if(fabsf(pivot)<1e-8) continue;
        for(int j=i;j<4;++j) M[i][j]/=pivot;
        for(int k=0;k<3;++k) {
            if(k==i) continue;
            float f=M[k][i];
            for(int j=i;j<4;++j) M[k][j]-=f*M[i][j];
        }
    }
    for(int i=0;i<3;++i) x[i]=M[i][3];
}

// CUDA kernel
__global__ void fit_ellipse_kernel(const float* points, float* results, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 1. 读取点
    float px[6], py[6];
    float cx=0, cy=0;
    for(int i=0;i<6;++i) {
        px[i]=points[idx*12+i*2+0];
        py[i]=points[idx*12+i*2+1];
        cx+=px[i];
        cy+=py[i];
    }
    cx/=6.0f; cy/=6.0f;

    // 2. 归一化
    float s=0;
    for(int i=0;i<6;++i) s+=fabsf(px[i]-cx)+fabsf(py[i]-cy);
    float scale = 100.0f/(s>FLT_EPSILON?s:FLT_EPSILON);

    // 3. 构造A, b
    float A[6][5], b[6];
    for(int i=0;i<6;++i) {
        float x=(px[i]-cx)*scale, y=(py[i]-cy)*scale;
        b[i]=10000.0f;
        A[i][0]=-x*x;
        A[i][1]=-y*y;
        A[i][2]=-x*y;
        A[i][3]=x;
        A[i][4]=y;
    }

    // 4. 用最小二乘法解A·gfp=b，gfp为5维参数
    // 这里用正规方程 A^T A x = A^T b
    float ATA[5][5]={0}, ATb[5]={0};
    for(int i=0;i<6;++i)
        for(int j=0;j<5;++j) {
            ATb[j]+=A[i][j]*b[i];
            for(int k=0;k<5;++k)
                ATA[j][k]+=A[i][j]*A[i][k];
        }
    // 高斯消元解5x5
    float gfp[5]={0};
    // 这里只写伪代码，建议用cuSolver或手写高斯消元
    // solve5x5(ATA, ATb, gfp);

    // 5. 求中心
    float AA[2][2], bb[2], rp[5]={0};
    AA[0][0]=2*gfp[0];
    AA[0][1]=gfp[2];
    AA[1][0]=gfp[2];
    AA[1][1]=2*gfp[1];
    bb[0]=gfp[3];
    bb[1]=gfp[4];
    solve2x2(AA, bb, rp); // rp[0]=cx, rp[1]=cy

    // 6. 重新拟合A-C
    float A3[6][3], b3[6];
    for(int i=0;i<6;++i) {
        float x=(px[i]-cx)*scale - rp[0], y=(py[i]-cy)*scale - rp[1];
        b3[i]=1.0f;
        A3[i][0]=x*x;
        A3[i][1]=y*y;
        A3[i][2]=x*y;
    }
    float ATA3[3][3]={0}, ATb3[3]={0};
    for(int i=0;i<6;++i)
        for(int j=0;j<3;++j) {
            ATb3[j]+=A3[i][j]*b3[i];
            for(int k=0;k<3;++k)
                ATA3[j][k]+=A3[i][j]*A3[i][k];
        }
    float gfp3[3]={0};
    solve3x3(ATA3, ATb3, gfp3);

    // 7. 解析角度和长短轴
    float min_eps=1e-8f;
    rp[4]=-0.5f*atan2f(gfp3[2], gfp3[1]-gfp3[0]);
    float t;
    if(fabsf(gfp3[2])>min_eps)
        t=gfp3[2]/sinf(-2.0f*rp[4]);
    else
        t=gfp3[1]-gfp3[0];
    rp[2]=fabsf(gfp3[0]+gfp3[1]-t);
    if(rp[2]>min_eps) rp[2]=sqrtf(2.0f/rp[2]);
    rp[3]=fabsf(gfp3[0]+gfp3[1]+t);
    if(rp[3]>min_eps) rp[3]=sqrtf(2.0f/rp[3]);

    // 8. 还原中心和尺度
    float center_x=(rp[0]/scale)+cx, center_y=(rp[1]/scale)+cy;
    float width=rp[2]*2.0f/scale, height=rp[3]*2.0f/scale;
    float angle=rp[4]*180.0f/3.14159265358979323846f;
    if(width>height) { swapf(width, height); angle+=90.0f; }
    if(angle<-180.0f) angle+=360.0f;
    if(angle>360.0f) angle-=360.0f;

    results[idx*5+0]=center_x;
    results[idx*5+1]=center_y;
    results[idx*5+2]=width;
    results[idx*5+3]=height;
    results[idx*5+4]=angle;
}

torch::Tensor fit_ellipses_cuda(torch::Tensor points) {
    int N = points.size(0);
    auto results = torch::zeros({N, 5}, points.options().device(points.device()).dtype(torch::kFloat32));
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    fit_ellipse_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        results.data_ptr<float>(),
        N
    );
    return results;
}
