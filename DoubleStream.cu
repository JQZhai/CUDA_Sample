 #include <stdio.h>

 //(A+B)/2=C
 
 #define N (2048*2048)//每个流执行数据大小
 #define FULL (N*20)//全部数据大小
 
 __global__ void kernel(int *a, int *b, int *c)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N)
    {
        c[idx] = (a[idx]+b[idx])/2;
    }
}
int main(void)
{
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap)
    {
        printf("paltform not support overlap");
        return 0;
    }
    //初始化计时器
    cudaEvent_t start, stop;
    float elapsedTime;
    //声明流和buffer指针
    cudaStream_t stream0;
    cudaStream_t stream1;
    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;
    //开始计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //初始化流
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    //GPU端内存申请
    cudaMalloc((void **)&dev_a0, N*sizeof(int));
    cudaMalloc((void **)&dev_b0, N*sizeof(int));
    cudaMalloc((void **)&dev_c0, N*sizeof(int));
    cudaMalloc((void **)&dev_a1, N*sizeof(int));
    cudaMalloc((void **)&dev_b1, N*sizeof(int));
    cudaMalloc((void **)&dev_c1, N*sizeof(int));
    //cpu端分配内存
    cudaHostAlloc((void**)&host_a, FULL*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL*sizeof(int), cudaHostAllocDefault);
    for(int i =0;i<FULL;i++){
        host_a[i]=rand();
        host_b[i]=rand();
    }
    cudaEventRecord(start,0);
    for(int i=0;i<FULL;i+=N*2){
        //将数据从cpu锁页内存传输给显存
        cudaMemcpyAsync(dev_a0,host_a+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(dev_a1,host_a+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(dev_b0,host_b+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(dev_b1,host_b+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1);
        kernel<<<N/256,256,0,stream0>>>(dev_a0,dev_b0,dev_c0);
        kernel<<<N/256,256,0,stream1>>>(dev_a1,dev_b1,dev_c1);
        //将计算结果从GPU显存传输给cpu内存
        cudaMemcpyAsync(host_c+i,dev_c0,N*sizeof(int),cudaMemcpyDeviceToHost,stream0);
        cudaMemcpyAsync(host_c+i+N,dev_c1,N*sizeof(int),cudaMemcpyDeviceToHost,stream1);
    }
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime,start,stop);
        printf("Time:%3.1f ms\n",elapsedTime);
        cudaFree(dev_a0);
        cudaFree(dev_b0);
        cudaFree(dev_c0);
        cudaFree(dev_a1);
        cudaFree(dev_b1);
        cudaFree(dev_c1);
        cudaFreeHost(host_a);
        cudaFreeHost(host_b);
        cudaFreeHost(host_c);
        cudaStreamDestroy(stream0);
        cudaStreamDestroy(stream1);
    return 0;
}