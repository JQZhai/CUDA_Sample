#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

using namespace std;

//c = a * b
int main(){
    int M = 4;//a的行数，c的行数
    int N = 4;//a的列数，b的行数
    int K = 4;//b的列数，c的列数
    float *host_a = (float *)malloc(sizeof(float)*M*N);
    float *host_b = (float *)malloc(sizeof(float)*N*K);
    float *host_c = (float *)malloc(sizeof(float)*M*K);
    for (int i=0;i<M*N;i++){
        host_a[i]=i;
    }
    for (int i=0;i<N*K;i++){
        host_b[i]=i;
    }
    cout<<"a:"<<endl;
    for (int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            cout<<host_a[j+i*N]<<" ";
            if((j+1+i*N)%M==0)
            {
                cout<<endl;
            }
        }
    }
    cout<<endl;
    cout<<"b:"<<endl;
    for (int i=0;i<K;i++){
        for(int j=0;j<N;j++){
            cout<<host_b[j+i*K]<<" ";
            if((j+1+i*K)%N==0)
            {
                cout<<endl;
            }
        }
    }
    cout<<endl;
    float *d_a, *d_b,*d_c;
    cudaMalloc((void **)&d_a,sizeof(float)*M*N);
    cudaMalloc((void **)&d_b,sizeof(float)*K*N);
    cudaMalloc((void **)&d_c,sizeof(float)*M*K);

    cudaMemcpy(d_a,host_a,sizeof(float)*M*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,host_b,sizeof(float)*N*K,cudaMemcpyHostToDevice);
    float alpha =1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,M,N,K,&alpha,d_a,M,d_b,N,&beta,d_c,N);
    cudaMemcpy(host_c,d_c,M*K*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<"c:"<<endl;
    for (int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            cout<<host_c[i+j*N]<<" ";
            if((j+1+i*N)%M==0)
            {
                cout<<endl;
            }
        }
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(host_a);
    free(host_b);
    free(host_c);
}