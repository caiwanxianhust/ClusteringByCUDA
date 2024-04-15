#include "kmeans.h"
#include "cuda_kernels.cuh"

#include <stdio.h>

namespace clustering {

template <typename DataType>
KmeansGPU<DataType>::KmeansGPU(DataType *h_clusters, int num_clusters, int num_features, int num_samples, int max_iters) 
    : m_num_clusters(num_clusters), m_num_features(num_features), m_num_samples(num_samples), m_max_iters(max_iters), 
    m_optTarget(1e7f), m_eplison(1e-4f)
{
    m_sample_classes = new int[m_num_samples]{0};
    m_clusters = new DataType[m_num_clusters * m_num_features];
    for (int i = 0; i < this->m_num_clusters * this->m_num_features; ++i)
    {
        m_clusters[i] = h_clusters[i];
    }
    int data_buf_size = m_num_samples * m_num_features;
    int cluster_buf_size = m_num_clusters * m_num_features;
    int cache_size = m_num_samples;
    int mem_size = sizeof(DataType) * (data_buf_size + cluster_buf_size) + sizeof(int) * (m_num_samples) +
                   sizeof(float) * (m_num_samples + m_num_samples) + sizeof(int) * m_num_clusters + sizeof(float) * cache_size;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, mem_size));

    d_clusters = (DataType *)(d_data + data_buf_size);
    d_sample_classes = (int *)(d_clusters + cluster_buf_size);
    d_min_dist = (float *)(d_sample_classes + m_num_samples);
    d_loss = (float *)(d_min_dist + m_num_samples);
    d_cluster_size = (int *)(d_loss + m_num_samples);

    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters, h_clusters, sizeof(DataType) * cluster_buf_size, cudaMemcpyHostToDevice));

    printf("num_samples: %d  num_clusters: %d  num_features: %d\n", num_samples, num_clusters, num_features);
}

template <typename DataType>
void KmeansGPU<DataType>::fit(const DataType *v_data)
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    cudaEventQuery(start);

    printf("***********************in fit*********************\n");
    float *h_loss = new float[m_num_samples]{0.0};

    cudaStream_t calculate_stream, update_stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&calculate_stream));
    CHECK_CUDA_ERROR(cudaStreamCreate(&update_stream));

    cudaEvent_t calculate_event, update_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&calculate_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&update_event));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, v_data, sizeof(DataType) * m_num_samples * m_num_features, cudaMemcpyHostToDevice, calculate_stream));
    // CHECK_CUDA_ERROR(cudaEventRecord(update_event, update_stream));
    char *d_cache_buf = (char *)(d_cluster_size + m_num_clusters);

    float lastLoss = 0.0f;
    for (int i = 0; i < m_max_iters; ++i)
    {
        launchFit<DataType>(d_data, d_clusters, d_sample_classes, d_cluster_size, d_min_dist, d_loss, d_cache_buf,
                            m_num_clusters, m_num_samples, m_num_features, calculate_stream, update_stream, 
                            calculate_event, update_event);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_loss, d_loss, sizeof(float) * m_num_samples, cudaMemcpyDeviceToHost, calculate_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(calculate_stream));
        this->m_optTarget = h_loss[0];
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


    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_clusters, d_clusters, sizeof(DataType) * m_num_clusters * m_num_features, cudaMemcpyDeviceToHost, calculate_stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_sample_classes, d_sample_classes, sizeof(int) * m_num_samples, cudaMemcpyDeviceToHost, calculate_stream));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventDestroy(update_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(calculate_event));
    CHECK_CUDA_ERROR(cudaStreamDestroy(calculate_stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(update_stream));
    delete [] h_loss;
}

template <typename DataType>
KmeansGPU<DataType>::~KmeansGPU() 
{
    delete [] m_sample_classes;
    delete [] m_clusters;
    CHECK_CUDA_ERROR(cudaFree(d_data));
}

template <typename DataType>
float KmeansGPU<DataType>::accuracy(int *label)
{
    float errCnt = 0;
    for (int i = 0; i < m_num_samples; ++i)
        if (m_sample_classes[i] != label[i])
            errCnt += 1.0;
    return 1.0 - errCnt / m_num_samples;
}


template class KmeansGPU<float>;

template class KmeansGPU<double>;

/*
template KmeansGPU<float>::KmeansGPU(float *h_clusters, int num_clusters, int num_features, int num_samples, int max_iters);
template void KmeansGPU<float>::fit(const float *v_data);

template KmeansGPU<double>::KmeansGPU(double *h_clusters, int num_clusters, int num_features, int num_samples, int max_iters);
template void KmeansGPU<double>::fit(const double *v_data);
*/

}   // clustering
