#pragma once
#include <stdio.h>

#define CHECK_CUDA_ERROR(call)                                             \
do {                                                            \
    const cudaError_t errorCode = call;                         \
    if (errorCode != cudaSuccess) {                             \
        printf("CUDA Error:\n");                                \
        printf("    File:   %s\n", __FILE__);                   \
        printf("    Line:   %d\n", __LINE__);                   \
        printf("    Error code:     %d\n", errorCode);          \
        printf("    Error text:     %s\n",                      \
            cudaGetErrorString(errorCode));                     \
        exit(1);                                                \
    }                                                           \
}                                                               \
while (0)

#define CHECK_CUBLAS_STATUS(call)                                             \
do {                                                            \
    const cublasStatus_t statusCode = call;                         \
    if (statusCode != CUBLAS_STATUS_SUCCESS) {                             \
        printf("CUDA Error:\n");                                \
        printf("    File:   %s\n", __FILE__);                   \
        printf("    Line:   %d\n", __LINE__);                   \
        printf("    Status code:     %d\n", statusCode);          \
        exit(1);                                                \
    }                                                           \
}                                                               \
while (0)

