#include "cuda_runtime.h"
#include <cuda.h>
#include <device_functions.h>
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

__global__ void sobel_gpu(unsigned char* in, unsigned char* out, int imgHeight, int imgWidth){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = y * imgWidth + x;
    int Gx = 0;
    int Gy = 0;
    unsigned char x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if(x > 0 && x < imgWidth && y >0 && y < imgHeight){
        x0 = in[(y-1)*imgWidth + x - 1];
        x1 = in[(y-1)*imgWidth + x];
        x2 = in[(y-1)*imgWidth + x + 1];
        x3 = in[index - 1];
        x4 = in[index];
        x5 = in[index + 1];
        x6 = in[(y+1)*imgWidth + x -1];
        x7 = in[(y+1)*imgWidth + x];
        x8 = in[(y+1)*imgWidth + x +1];
        Gx = x0 + 2*x3 + x6 - x2 - 2*x5 - x8;
        Gy = x0 + 2*x1 + x2 - x6 - 2*x7 - x8;
        out[index] = (abs(Gx)+abs(Gy)) / 2;
    }
}

void sobel_cpu(Mat srcImg, Mat dstImg){
    int Gx = 0;
    int Gy = 0;
    for(int i = 1; i < imgHeight - 1; i ++){
        uchar* dataUp = srcImg.ptr<uchar>(i-1);
        uchar* data = srcImg.ptr<uchar>(i);
        uchar* dataDown = srcImg.ptr<uchar>(i+1);
        uchar* out = dstImg.ptr<uchar>(i);
        for (int j = 1; j < imgWidth; j ++){
            Gx = (dataUp[j+1] + 2*data[j+1] + dataDown[j+1] - dataUp[j-1] - 2*data[j-1] - dataDown[j+1]);
            Gy = (dataUp[j-1] + 2*dataUp[j] + dataUp[j+1] - dataDown[j-1] - 2*dataDown[j] - dataDown[j+1]);
        }
    }
}

int main(){
    Mat grayImg = imread("1.jpg");
    int imgWidth = grayImg.cols;
    int imgHeight = grayImg.rows;

    Mat gaussImg;
    GaussianBlur(grayImg, gaussImg, size(3,3), 0, 0, BORDER_DEFAULT);
    Mat dst_cpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    sobel_cpu(gaussImg, dst_cpu, imgHeight, imgWidth);

    size_t num = sizeof(unsigned char) * imgHeight * imgWidth;
    unsigned char* in_gpu;
    unsigned char* out_gpu;
    cudaMalloc((void **)&in_gpu, num);
    cudaMalloc((void **)&out_gpu, num);

    dim3 threadPerBlock(32, 32);
    dim3 blockPerGrid((imgWidth + threadPerBlock.x -1) / threadPerBlock.x,
    (imgHeight + threadPerBlock.y -1) / threadPerBlock.y);

    cudaMemcpy(in_gpu, gaussImg.data, num, cudaMemcpyHostToDevice);

    sobel_gpu<<<blockPerGrid, threadPerBlock>>>(in_gpu, out_gpu, imgHeight, imgWidth);

    cudaMemcpy(dst_gpu, out_gpu, num, cudaMemcpyDeviceToHost);

    imshow("gpu", dst_gpu);
    imshow("cpu", dst_cpu);

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    
    return 0;
}