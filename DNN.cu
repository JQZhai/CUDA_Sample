#include "cuda_runtime.h"
#include <cuda.h>
#include <device_function.h>
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

__global__ void sobel_gpu(unsigned char *in, unsigned char *out, int imgHeight, int imgWidth) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int index = y * imgWidth + x;
    int Gx = 0;
    int Gy = 0;
    unsigned char x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if(x>0 && x < imgWidth && y>0 && y<imgHeight){
        x0 = in[(y-1)*imgWidth+x-1];
        
    }
}


void sobel_cpu(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth){
    int Gx = 0;
    int Gy = 0;
    for (int i = 1; i < )
}

int main()