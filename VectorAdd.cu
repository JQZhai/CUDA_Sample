#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
int main(void)
{
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("Vector addition of %d element.\n", numElements);

    float *h_a =(float *)malloc(size);
    float *h_b =(float *)malloc(size);
    float *h_c =(float *)malloc(size);
    for(int i = 0; i < numElements; i ++)
    {
        h_a[i] = rand()/(float)RAND_MAX;
        h_b[i] = rand()/(float)RAND_MAX;
    }
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    printf("copy input data from the host to device memory\n");
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    int ThreadPreBlock = 256;
    int BlockPreGrid = (numElements + ThreadPreBlock -1)/ThreadPreBlock;
    vectorAdd<<<BlockPreGrid, ThreadPreBlock>>>(d_a, d_b, d_c, numElements);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < numElements; i++)
    {
        if(fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
        {
            fprintf(stderr,"num %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    printf("PASSED\n");
    return 0;
}