#include "fuzzyCmeans.h"
#include "cuda_kernels.cuh"
#include <cmath>
#include <cstdio>


namespace clustering {
namespace fuzzyCmeans {

namespace 
{
    constexpr int block_size = 256;   
}


template <typename DataType>
void launchUpdateClustersKernel(const DataType *d_data, 
    const float *membership,
    DataType *d_clusters, 
    const int num_features,
    const int num_samples,
    const int num_clusters,
    const float scale,
    cudaStream_t stream)
{
    dim3 grid(num_features, num_clusters);
    dim3 block(block_size);
    updateClustersKernel<DataType><<<grid, block, 0, stream>>>(d_data, membership, d_clusters, num_features, num_samples, scale);
}

template <typename DataType>
void launchCalculateLossAndUpdateMembershipKernel(const DataType *d_data, 
    const DataType *d_clusters, 
    float *membership,
    float *distances,
    const int num_features,
    const int num_clusters,
    const int num_samples,
    const float scale, 
    cudaStream_t stream)
{
    calculateDistancesAndUpdateMembershipKernel<DataType><<<num_samples, block_size, 0, stream>>>(d_data, d_clusters, membership, 
        distances, num_features, num_clusters, num_samples, scale);
}

template <template <typename T> class ReductionOp, typename T>
void cubDeviceReduce(const T *d_in, T *d_out, const int num_items, T init, char *d_cache_buf, cudaStream_t stream)
{
    auto op_func = ReductionOp<T>();
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    // cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op_func, init, stream);
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op_func, init, stream);

    // Run reduction, use the preallocte buffer instead of allocate temporary storage
    cub::DeviceReduce::Reduce((void *)d_cache_buf, temp_storage_bytes, d_in, d_out, num_items, op_func, init, stream);
}


template <typename DataType>
__global__ void argmaxKernel(const DataType *mat, int *labels, const int cols)
{
    DataType val = -1e7f;
    int max_idx = -1;
    for (int i=threadIdx.x; i<cols; i+=blockDim.x) {
        max_idx = (val < mat[blockIdx.x * cols + i] ? i : max_idx);
        val = (val < mat[blockIdx.x * cols + i] ? mat[blockIdx.x * cols + i] : val);
    }
    DataType max_val = blockAllReduce<MaxOp, DataType>(val);
    if (max_val == val) labels[blockIdx.x] = max_idx;
}

template <typename DataType>
void launchArgmaxKernel(const DataType *mat, int *labels, const int rows, const int cols, cudaStream_t stream = 0)
{
    argmaxKernel<DataType><<<rows, block_size, 0, stream>>>(mat, labels, cols);
}


template <typename DataType>
void launchFit(const DataType *d_data, DataType *d_clusters, float *d_membership, float *d_distances, float *d_loss, char *d_cache_buf, 
    const int num_clusters, const int num_samples, const int num_features, const float scale, cudaStream_t calculate_stream)
{
    launchUpdateClustersKernel<DataType>(d_data, d_membership, d_clusters, num_features, num_samples, num_clusters, scale, calculate_stream);

    launchCalculateLossAndUpdateMembershipKernel<DataType>(d_data, d_clusters, d_membership, d_distances, num_features, num_clusters,
        num_samples, scale, calculate_stream);
    cubDeviceReduce<SumOp, float>(d_distances, d_loss, num_samples * num_clusters, 0.0f, d_cache_buf, calculate_stream);
}



template <typename DataType>
FuzzyCmeans<DataType>::FuzzyCmeans(float *h_membership, int num_clusters, int num_features, int num_samples, float scale, int max_iters) :
    m_num_clusters(num_clusters), m_num_features(num_features), m_num_samples(num_samples), m_scale(scale), m_max_iters(max_iters), 
    m_optTarget(1e7f), m_eplison(1e-10f)
{
    int mem_size = sizeof(DataType) * (m_num_clusters * m_num_features) + sizeof(float) * (m_num_samples * m_num_clusters) + 
        sizeof(int) * m_num_samples;
    m_clusters = (DataType *)malloc(mem_size);
    m_membership = (float *)(m_clusters + m_num_clusters * m_num_features);
    m_labels = (int *)(m_membership + m_num_samples * m_num_clusters);

    int cache_size = m_num_samples * m_num_clusters;
    mem_size = sizeof(DataType) * (m_num_samples * num_features) + sizeof(DataType) * (m_num_clusters * m_num_features) + 
        sizeof(float) * (m_num_samples * m_num_clusters) + sizeof(float) * (m_num_samples * m_num_clusters) + 
        sizeof(int) * m_num_samples + sizeof(float) * (1 + cache_size);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, mem_size));
    d_clusters = (DataType *)(d_data + m_num_samples * num_features);
    d_membership = (float *)(d_clusters + m_num_clusters * m_num_features);
    d_distances = (float *)(d_membership + m_num_samples * m_num_clusters);
    d_labels = (int *)(d_distances + m_num_samples * m_num_clusters);
    d_loss = (float *)(d_labels + m_num_samples);

    CHECK_CUDA_ERROR(cudaMemcpy(d_membership, h_membership, sizeof(float) * (m_num_samples * m_num_clusters), cudaMemcpyHostToDevice));

    printf("num_samples: %d  num_clusters: %d  num_features: %d\n", num_samples, num_clusters, num_features);
}

template <typename DataType>
void FuzzyCmeans<DataType>::fit(const DataType *v_data)
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    cudaEventQuery(start);

    printf("***********************in fit*********************\n");
    float h_loss;

    cudaStream_t calculate_stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&calculate_stream));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, v_data, sizeof(DataType) * m_num_samples * m_num_features, cudaMemcpyHostToDevice, calculate_stream));
    char *d_cache_buf = (char *)(d_loss + 1);

    float lastLoss = 0.0f;
    for (int i = 0; i < m_max_iters; ++i)
    {
        launchFit<DataType>(d_data, d_clusters, d_membership, d_distances, d_loss, d_cache_buf, m_num_clusters, m_num_samples, m_num_features,
            m_scale, calculate_stream);
        
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&h_loss, d_loss, sizeof(float) * 1, cudaMemcpyDeviceToHost, calculate_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(calculate_stream));
        this->m_optTarget = h_loss;
        if (std::abs(lastLoss - this->m_optTarget) < this->m_eplison) {
            printf("break!!!    lastLoss: %g  m_optTarget: %g\n", lastLoss, this->m_optTarget);
            break;
        }
        lastLoss = this->m_optTarget;
        printf("Iters: %d  current loss: %g\n", i+1, this->m_optTarget);
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    printf("Time = %g ms.\n", elapsedTime);

    launchArgmaxKernel<float>(d_membership, d_labels, m_num_samples, m_num_clusters, calculate_stream);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(m_clusters, d_clusters, sizeof(DataType) * m_num_clusters * m_num_features, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(m_membership, d_membership, sizeof(float) * m_num_samples * m_num_clusters, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(m_labels, d_labels, sizeof(int) * m_num_samples, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaStreamDestroy(calculate_stream));
}

template <typename DataType>
FuzzyCmeans<DataType>::~FuzzyCmeans()
{
    free(m_clusters);
    CHECK_CUDA_ERROR(cudaFree(d_data));
}

template void launchFit<float>(const float *d_data, float *d_clusters, float *d_membership, float *d_distances, float *d_loss, char *d_cache_buf, 
    const int num_clusters, const int num_samples, const int num_features, const float scale, cudaStream_t calculate_stream);

template void launchArgmaxKernel<float>(const float *mat, int *labels, const int rows, const int cols, cudaStream_t stream);


template void launchArgmaxKernel<double>(const double *mat, int *labels, const int rows, const int cols, cudaStream_t stream);

template class FuzzyCmeans<float>;

template class FuzzyCmeans<double>;


}   // FuzzyCmeans
}   // Clustering
