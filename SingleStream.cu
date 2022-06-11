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
    cudaStream_t stream;
    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;
    //开始计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //初始化流
    cudaStreamCreate(&stream);
    //GPU端内存申请
    cudaMalloc((void **)&dev_a, N*sizeof(int));
    cudaMalloc((void **)&dev_b, N*sizeof(int));
    cudaMalloc((void **)&dev_c, N*sizeof(int));
    //cpu端分配内存
    cudaHostAlloc((void**)&host_a, FULL*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL*sizeof(int), cudaHostAllocDefault);
    for(int i =0;i<FULL;i++){
        host_a[i]=rand();
        host_b[i]=rand();
    }
    cudaEventRecord(start,0);
    for(int i=0;i<FULL;i+=N){
        //将数据从cpu锁页内存传输给显存
        cudaMemcpyAsync(dev_a,host_a+i,N*sizeof(int),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(dev_b,host_b+i,N*sizeof(int),cudaMemcpyHostToDevice,stream);
        kernel<<<N/256,256,0,stream>>>(dev_a,dev_b,dev_c);
        //将计算结果从GPU显存传输给cpu内存
        cudaMemcpyAsync(host_c+i,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost,stream);
    }
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime,start,stop);
        printf("Time:%3.1f ms\n",elapsedTime);
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaFreeHost(host_a);
        cudaFreeHost(host_b);
        cudaFreeHost(host_c);
        cudaStreamDestroy(stream);
    return 0;
}