#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH  16

__global__ void MatrixMul(float *Md, float *Nd, float *Pd, const int WIDTH)
{
    int col = TILE_WIDTH*blockIdx.x+threadIdx.x;
    int row = TILE_WIDTH*blockIdx.y+threadIdx.y;
    for (int k = 0; k < WIDTH; k++)
    {
        Pd[row*WIDTH+col]+=Md[row*WIDTH+k]*Nd[k*WIDTH+col];
    }
}

__global__ void MatrixMulShare(float *Md, float *Nd, float *Pd, const int WIDTH)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int Pervalue = 0;
    int col = TILE_WIDTH*blockIdx.x+threadIdx.x;
    int row = TILE_WIDTH*blockIdx.y+threadIdx.y;
    for(int m =0;m<WIDTH/TILE_WIDTH;m++)
    {
       Mds[threadIdx.y][threadIdx.x] = Md[row*WIDTH+(m*TILE_WIDTH+threadIdx.x)];
       Nds[threadIdx.y][threadIdx.x] = Nd[(m*TILE_WIDTH+threadIdx.y)*WIDTH+col];
       __syncthreads();
       for(int k = 0;k<TILE_WIDTH;k++)
       {
           Pervalue+=Mds[threadIdx.x][k]+Nds[threadIdx.y][k];
       } 
       __syncthreads();
    }
    Pd[row*WIDTH+col] = Pervalue;
}
int main()
{
    const int WIDTH = 512;
    float array1_h[WIDTH][WIDTH], array2_h[WIDTH][WIDTH];
    float reasult_array_h[WIDTH][WIDTH];
    float *array1_d, *array2_d, *reasult_array_d;
    int i,j;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(i = 0; i < WIDTH; i++)
    {
        for(j = 0; j < WIDTH; j ++)
        {
            array1_h[i][j] = 1;
            array2_h[i][j] = 2;
        }
    }
    //在GPU上分配Array
    int size = WIDTH * WIDTH * sizeof(int);
    cudaMalloc((void**) &array1_d, size);
    cudaMalloc((void**) &array2_d, size);
    cudaMemcpy(array1_d, array1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(array2_d, array2_h, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &reasult_array_d, size);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
    dim3 dimGrid(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH,1);
    cudaEventRecord(start, 0);
    // MatrixMul<<<dimGrid,dimBlock>>>(array1_d, array2_d, reasult_array_d, WIDTH);
    MatrixMulShare<<<dimGrid,dimBlock>>>(array1_d, array2_d, reasult_array_d, WIDTH);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%f\n",elapsedTime);
    cudaMemcpy(reasult_array_h,reasult_array_d,size,cudaMemcpyDeviceToHost);
    cudaFree(array1_d);
    cudaFree(array2_d);
    cudaFree(reasult_array_d);
    return 0;
}