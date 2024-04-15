#pragma once
#include "error.cuh"
#include <cub/cub.cuh>


template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T warpAllReduce(T val)
{
    auto func = ReductionOp<T>();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = func(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__inline__ __device__
    T
    blockAllReduce(T val)
{
    static __shared__ T shared[32];
    __shared__ T result;
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpAllReduce<ReductionOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpAllReduce<ReductionOp, T>(val);
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}

template <typename DataType>
__inline__ __device__ float calculateDistance(const DataType *sample, const DataType *cluster, const int num_features)
{
    float sub_square_sum = 0.0f;
    float sub_val;

#pragma unroll
    for (int idx=threadIdx.x; idx<num_features; idx+=blockDim.x) {
        sub_val = (float)(__ldg(sample + idx) - __ldg(cluster + idx));
        sub_square_sum += sub_val * sub_val;
    }
    __syncthreads();

    sub_square_sum = blockAllReduce<SumOp, float>(sub_square_sum);
    return sub_square_sum;
}


/**
 * @brief 根据聚类中心计算损失，并求新的隶属度矩阵
 *  gird: [num_samples, 1, 1]   block: [block_size, 1, 1]
 * @param d_data           [num_samples, num_features] 
 * @param d_clusters       [num_clusters, num_features]     聚类中心 
 * @param membership       [num_clusters, num_samples]      隶属度矩阵  
 * @param distances        [num_samples, num_clusters]      distance 矩阵  
 * @param scale                                             模糊系数
 * @return void
*/
template <typename DataType>
__global__ void calculateDistancesAndUpdateMembershipKernel(const DataType *d_data, 
    const DataType *d_clusters, 
    float *membership,
    float *distances,
    const int num_features,
    const int num_clusters,
    const int num_samples,
    const float scale)
{
    const DataType *sample = d_data + blockIdx.x * num_features;
    float distance_ij;
    __shared__ float s_distance_ik[256];

    for (int j=0; j<num_clusters; ++j) {
        const DataType *cluster = d_clusters + j * num_features;
        distance_ij = calculateDistance<DataType>(sample, cluster, num_features);
        s_distance_ik[j] = distance_ij;
    }
    __syncthreads();

    if (threadIdx.x < num_clusters) {
        float old_u_ij_m = __powf(__ldg(membership + threadIdx.x * num_samples + blockIdx.x), scale);
        distances[blockIdx.x * num_clusters + threadIdx.x] = s_distance_ik[threadIdx.x] * old_u_ij_m;

        // if (blockIdx.x == 0) printf("clusterId: %d  distance: %g\n", threadIdx.x, distances[blockIdx.x * num_clusters + threadIdx.x]);

        float distance_ik_sum = 0.0f;
        for (int j=0; j<num_clusters; ++j) {
            distance_ik_sum += __powf((s_distance_ik[threadIdx.x] / s_distance_ik[j]), (1.0f/(scale - 1.0f))); 
        }
        float u_ij = 1.0 / distance_ik_sum;
        membership[threadIdx.x * num_samples + blockIdx.x] = u_ij;
    }
}



/**
 * @brief 根据隶属度矩阵求聚类中心
 *  gird: [num_features, num_clusters, 1]   block: [block_size, 1, 1]
 * @param d_data           [num_samples, num_features] 
 * @param d_clusters       [num_clusters, num_features]     聚类中心 
 * @param membership       [num_clusters, num_samples]      隶属度矩阵
 * @param num_features
 * @param num_samples
 * @param scale                                             模糊系数
 * @return void
*/
template <typename DataType>
__global__ void updateClustersKernel(const DataType *d_data, 
    const float *membership,
    DataType *d_clusters, 
    const int num_features,
    const int num_samples, 
    const float scale)
{
    int cluster_id = blockIdx.y;
    int feature_id = blockIdx.x;
    float val = 0.0f;
    float u_ij_m_sum = 0.0f;
    float u_ij_m;
    for (int i=threadIdx.x; i<num_samples; i+=blockDim.x) {
        u_ij_m = __powf(__ldg(membership + cluster_id * num_samples + i), scale);
        // if (i == 0 && feature_id == 0) printf("cluster_id: %d  u_ij_m: %g\n", cluster_id, u_ij_m);
        val += u_ij_m * __ldg(d_data + i * num_features + feature_id);
        u_ij_m_sum += u_ij_m;
    }
    __syncthreads();

    val = blockAllReduce<SumOp, DataType>(val);
    u_ij_m_sum = blockAllReduce<SumOp, DataType>(u_ij_m_sum);
    if (threadIdx.x == 0) {
        d_clusters[cluster_id * num_features + feature_id] = (DataType)(val / u_ij_m_sum);
        // if (feature_id < 10) printf("cluster_id: %d feature_id: %d  val: %g \n", cluster_id, feature_id, 
        //     d_clusters[cluster_id * num_features + feature_id]);
    } 
}