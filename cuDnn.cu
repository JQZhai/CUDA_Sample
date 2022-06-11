#include "cuda_runtime.h"
#include <cudnn.h>
#include <cuda.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

float3 data_kernel[]{
    make_float3(-1.0f,-1.0f,-1.0f),make_float3(0.0f,0.0f,0.0f),make_float3(1.0f,1.0f,1.0f),
    make_float3(-2.0f,-2.0f,-2.0f),make_float3(0.0f,0.0f,0.0f),make_float3(2.0f,2.0f,2.0f),
    make_float3(-1.0f,-1.0f,-1.0f),make_float3(0.0f,0.0f,0.0f),make_float3(1.0f,1.0f,1.0f),
    make_float3(-1.0f,-1.0f,-1.0f),make_float3(0.0f,0.0f,0.0f),make_float3(1.0f,1.0f,1.0f),
    make_float3(-2.0f,-2.0f,-2.0f),make_float3(0.0f,0.0f,0.0f),make_float3(2.0f,2.0f,2.0f),
    make_float3(-1.0f,-1.0f,-1.0f),make_float3(0.0f,0.0f,0.0f),make_float3(1.0f,1.0f,1.0f),
    make_float3(-1.0f,-1.0f,-1.0f),make_float3(0.0f,0.0f,0.0f),make_float3(1.0f,1.0f,1.0f),
    make_float3(-2.0f,-2.0f,-2.0f),make_float3(0.0f,0.0f,0.0f),make_float3(2.0f,2.0f,2.0f),
    make_float3(-1.0f,-1.0f,-1.0f),make_float3(0.0f,0.0f,0.0f),make_float3(1.0f,1.0f,1.0f)
};

int main(){
    Mat img = imread("1.jpg");
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    imt imgChannel = img.channels();

    Mat dat_gpu(imgHeight, imgWidth, CV_8UC3, Scalar(0,0,0));
    size_t num = imgChannel * imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char *in_gpu;
    unsigned char *out_gpu;
    float *filt_data;
    cudaMalloc((viod**)&filt_data, 3*3*3*sizeof(float3));
    cudaMalloc((void**)&in_gpu, num);
    cudaMalloc((void**)&out_gpu, num);

    cudnnHandle_t handle;
    cudnnCreate(&handle);
    // cudnnSetStream(handle,stream1);
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,1,3,imgHeight,imgWidth);

    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,1,3,imgHeight,imgWidth);

    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4Descriptor(kernel_descriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,3,3,3,3);

    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgoperf_t algo;
    cudnnGetconvolutionForwardAlgorithm_v7(handle,input_descriptor,kernel_descriptor,
    conv_descriptor,output_descriptor,1,0,&algo);
    
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,input_descriptor,kernel_descriptor,conv_descriptor,output_descriptor,
    algo.algo,&workspace_size);
    void *workspace = nullptr;
    cudaMalloc(&workspace,workspace_size);
    cudaMemcpy((void*)filt_data,(void*)data_kernel,3*3*3*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(in_gpu,img.data,num,cudaMemcpyHostToDevice);

    auto alpha = 1.0f,beta=0.0f;
    cudnnGetConvolutionForward(handle,&alpha,input_descriptor,in_gpu,kernel_descriptor,filt_data,conv_descriptor,algo.algo,workspace,workspace_size,&beta,output_descriptor,out_gpu);

    cudaMemcpy(dst_gpu.data,out_gpu,num,cudaMemcpyDeviceToHost);

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroy(handle);
    return 0;

}