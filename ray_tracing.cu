#include "cuda.h"
#include "book.h"
#include "cpu_bitmap.h"
#define DIM 1024
#define rnd(x) (x*rand()/RAND_MAX)
#define INF 2e10f

struct Sphere{
    float r,g,b;
    float radius;
    float x,y,z;
    //定义射线与球相交函数，返回相交点到图像的距离
    __device__ float hit(float ox, float oy, float oz, float *n){
        float dx = ox - x;
        float dy = oy - y;
        if(dx*dx + dy*dy < radius*radius){
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz / sqrtf(radius*radius);
            return dz + z;
        }
        return -INF;
    }
}

#define SPHERES 20

__constant__ Sphere s[SPHERES]

__global__ void kernel(unsigned char *ptr){
    int x = threadsIdx.x + blockIdx.x * blockDim.x;
    int y = threadsIdx.x + blockIdx.x * blockDim.x;
    int offset = x + y * blockDim.x * gridDim.x;
    //将图像坐标转换为以图像中心为原点的直角坐标
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);
    //对该线程要处理的像素值进行初始化
    float r = 0,g = 0,b=0;
    float maxz = -INF;
    //对每个线程遍历一遍所有球，判断是否与球相交，返回距离画面最近的点，并给该像素进行赋值
    for(int i =0;i<SPHERES;i++){
        float n;
        float t = s[i].hit(ox,oy,&n);
        if(t>maxz){
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maz = t ;

        }
    }   
    //输出
    ptr[offset * 4 +0] = (int)(r * 255);
    ptr[offset * 4 +1] = (int)(g * 255);
    ptr[offset * 4 +2] = (int)(b * 255);
    ptr[offset * 4 +3] = 255;
}

struct DataBlock{
    unsigned char *dev_bitmap;
}

int main(void){
    DataBlock data;
    cudaEvent_t start stop;
    
}
