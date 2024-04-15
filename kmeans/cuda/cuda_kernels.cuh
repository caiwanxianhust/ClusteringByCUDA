#pragma once
#include "error.cuh"
#include <type_traits>
#include <cub/cub.cuh>

namespace clustering {

namespace
{
    constexpr int block_size = 128;
}

template<typename T, int N>
struct GetPackType {
    using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed
{
    __device__ Packed()
    {
        // do nothing
    }
    union
    {
        PackType<T, pack_size> storage;
        T elem[pack_size]; // 这里联合体只有一个成员，为了方便后期扩展
    };
};



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
__inline__ __device__ T WarpReduce(T val)
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
    blockReduce(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduce<ReductionOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = WarpReduce<ReductionOp, T>(val);
    return val;
}


template <typename DataType>
__global__ void calDistKernel(
    const DataType *d_data,
    const DataType *d_clusters, // [num_clusters, num_features]
    float *d_distance,          // [num_samples, num_clusters]
    const int num_clusters,
    const int clusterNo,
    const int num_samples,
    const int num_features)
{
    // grid_size = num_samples, block_size = 128
    int sample_offset = num_features * blockIdx.x;
    int cluster_offset = num_features * clusterNo;

    __shared__ float s_c[1024];
    s_c[threadIdx.x] = 0.0f;
    __syncthreads();

    float val;
    for (int i = threadIdx.x; i < num_features; i += blockDim.x)
    {
        val = (float)(d_data[sample_offset + i] - d_clusters[cluster_offset + i]);
        s_c[threadIdx.x] += val * val;
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncwarp();
    }

    if (threadIdx.x == 0)
        d_distance[blockIdx.x * num_clusters + clusterNo] = sqrtf(s_c[0]);
}

__global__ void reduceMin(
    float *d_distance,
    int *d_sampleClasses,
    int *d_cluster_size,
    int num_clusters,
    int num_samples,
    float *d_min_dist)
{
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < num_samples)
    {
        float minDist = d_distance[n * num_clusters + 0];
        int minIdx = 0;
        float tmp;
        for (int i = 1; i < num_clusters; i++)
        {
            tmp = __ldg(&d_distance[n * num_clusters + i]);
            if (tmp < minDist)
            {
                minDist = tmp;
                minIdx = i;
            }
        }
        d_sampleClasses[n] = minIdx;
        d_min_dist[n] = minDist;
    }
}

__global__ void reduceSum(
    float *d_min_dist,
    float *d_loss,
    int num_samples)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float s_y[];
    float y = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (; n < num_samples; n += stride)
        y += d_min_dist[n];
    s_y[threadIdx.x] = y;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0)
        d_loss[blockIdx.x] = s_y[0];
}

__global__ void countCluster(int *d_sampleClasses, int *d_cluster_size, int num_samples)
{
    // grid_size = (num_samples - 1) / block_size, block_size = block_size(128)
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < num_samples)
    {
        int clusterID = d_sampleClasses[n];
        atomicAdd(&(d_cluster_size[clusterID]), 1);
    }
}

template <typename DataType>
__global__ void updateClusters(
    const DataType *d_data,
    DataType *d_clusters,
    int *d_sampleClasses,
    int *d_cluster_size,
    const int num_samples,
    const int num_features)
{

    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int clusterId = d_sampleClasses[n];
    int clustercnt = d_cluster_size[clusterId];
    extern __shared__ DataType s_c[]; // [num_clusters, num_features]
    for (int i = 0; i < num_features; ++i)
    {
        s_c[clusterId * num_features + i] = 0.0f;
    }
    __syncthreads();
    if (n < num_samples)
    {
        for (int i = 0; i < num_features; ++i)
        {
            atomicAdd(&s_c[clusterId * num_features + i], d_data[n * num_features + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < num_features; ++i)
        {
            atomicAdd(&d_clusters[clusterId * num_features + i], s_c[clusterId * num_features + i] / clustercnt);
        }
    }
}

template <typename T>
__global__ void init(T *x, const T value, const int N)
{
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N)
        x[n] = value;
}


template <typename T>
__global__ void initV2(T *x, const T value, const int N)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll
    for (; n < N; n += gridDim.x * blockDim.x)
        x[n] = value;
}

template <typename DataType>
__device__ float calDistV2(const DataType *d_data,
                           const DataType *d_clusters, // [num_clusters, num_features]
                           const int clusterNo, const int num_features)
{
    // grid_size = num_samples, block_size = 256
    const int sample_offset = num_features * blockIdx.x;
    const int cluster_offset = num_features * clusterNo;

    float distance = 0.0f;
    float sub_val;

#pragma unroll
    for (int i = threadIdx.x; i < num_features; i += blockDim.x)
    {
        sub_val = (float)(d_data[sample_offset + i] - d_clusters[cluster_offset + i]);
        distance += sub_val * sub_val;
    }
    __syncthreads();

    distance = blockReduce<SumOp, float>(distance);
    return distance;
}

template <typename DataType>
__device__ float calDistV3(const DataType *d_data,
                           const DataType *d_clusters, // [num_clusters, num_features]
                           const int num_features, 
                           int lane_id)
{
    float distance = 0.0f;
    float sub_val;

#pragma unroll
    for (int i = lane_id; i < num_features; i += 32)
    {
        sub_val = (float)(d_data[i] - d_clusters[i]);
        distance += sub_val * sub_val;
    }
    __syncwarp();

    distance = WarpReduce<SumOp, float>(distance);
    return distance;
}

template <typename DataType, int pack_size>
__device__ float calDistPacked(const DataType *d_data,
                               const DataType *d_clusters, // [num_clusters, num_features]
                               const int clusterNo, const int num_features)
{
    // grid_size = num_samples, block_size = 256
    const int sample_offset = num_features * blockIdx.x;
    const int cluster_offset = num_features * clusterNo;

    const PackType<DataType, pack_size> *buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_data + sample_offset);
    const PackType<DataType, pack_size> *cluster_buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_clusters + cluster_offset);
    int num_packs = num_features / pack_size;

    float distance = 0.0f;
    float sub_val;
    Packed<DataType, pack_size> data_pack;
    Packed<DataType, pack_size> cluster_pack;

#pragma unroll
    for (int pack_id = threadIdx.x; pack_id < num_packs; pack_id += blockDim.x)
    {
        data_pack.storage = *(buf + pack_id);
        cluster_pack.storage = *(cluster_buf + pack_id);
#pragma unroll
        for (int elem_id = 0; elem_id < pack_size; ++elem_id)
        {
            sub_val = (float)(data_pack.elem[elem_id] - cluster_pack.elem[elem_id]);
            distance += sub_val * sub_val;
        }
    }
    __syncthreads();

    distance = blockReduce<SumOp, float>(distance);
    return distance;
}

template <typename DataType, int pack_size>
__device__ float calDistWarpPacked(const DataType *d_data,
                               const DataType *d_clusters, // [num_clusters, num_features]
                               const int num_features,
                               int lane_id)
{
    const PackType<DataType, pack_size> *buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_data);
    const PackType<DataType, pack_size> *cluster_buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_clusters);
    int num_packs = num_features / pack_size;

    float distance = 0.0f;
    float sub_val;
    Packed<DataType, pack_size> data_pack;
    Packed<DataType, pack_size> cluster_pack;

#pragma unroll
    for (int pack_id = lane_id; pack_id < num_packs; pack_id += 32)
    {
        data_pack.storage = *(buf + pack_id);
        cluster_pack.storage = *(cluster_buf + pack_id);
#pragma unroll
        for (int elem_id = 0; elem_id < pack_size; ++elem_id)
        {
            sub_val = (float)(data_pack.elem[elem_id] - cluster_pack.elem[elem_id]);
            distance += sub_val * sub_val;
        }
    }
    __syncwarp();
    
    distance = WarpReduce<SumOp, float>(distance);
    return distance;
}

template <typename DataType>
__global__ void calClustersDistkernel(const DataType *d_data,
                                      const DataType *d_clusters, // [num_clusters, num_features]
                                      int *d_sample_classes,      // [nsamples, ]
                                      float *d_min_dist,          // [nsamples, ]
                                      const int num_features,
                                      const int num_clusters)
{
    // grid_size = num_samples, block_size = 256
    float min_dist = 1e9f;
    float dist;
    int min_idx;

#pragma unroll
    for (int i = 0; i < num_clusters; ++i)
    {
        // if (blockIdx.x == 0 && i < 3 && threadIdx.x == 0) printf("sample_id: %d  c_id: %d  dx: %g  cx: %g\n", blockIdx.x, i, d_data[num_features * blockIdx.x], 
        //     d_clusters[i * num_features]);
        dist = calDistV2<DataType>(d_data, d_clusters, i, num_features);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx = i;
        }
    }

    if (threadIdx.x == 0)
    {
        // printf("sample_id: %d  min_dist: %g\n", blockIdx.x, min_dist);
        d_sample_classes[blockIdx.x] = min_idx;
        d_min_dist[blockIdx.x] = sqrtf(min_dist);
    }
}

template <typename DataType, int pack_size>
__global__ void calClustersDistPackedkernel(const DataType *d_data,
                                            const DataType *d_clusters, // [num_clusters, num_features]
                                            int *d_sample_classes,      // [nsamples, ]
                                            float *d_min_dist,          // [nsamples, ]
                                            int *d_clusterSize,         // [nsamples, ]
                                            const int num_features,
                                            const int num_clusters)
{
    // grid_size = num_samples, block_size = 256
    float min_dist = 1e9f;
    float dist;
    int min_idx;

#pragma unroll
    for (int i = 0; i < num_clusters; ++i)
    {
        dist = calDistPacked<DataType, pack_size>(d_data, d_clusters, i, num_features);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx = i;
        }
    }

    if (threadIdx.x == 0)
    {
        d_sample_classes[blockIdx.x] = min_idx;
        d_min_dist[blockIdx.x] = sqrtf(min_dist);
        atomicAdd(&(d_clusterSize[min_idx]), 1);
    }
}

template <typename DataType>
__global__ void calClustersDistWarpkernel(const DataType *d_data,
                                                const DataType *d_clusters, // [num_clusters, num_features]
                                                int *d_sample_classes,      // [nsamples, ]
                                                float *d_min_dist,          // [nsamples, ]
                                                int *d_clusterSize,         // [nsamples, ]
                                                const int num_features,
                                                const int num_clusters,
                                                const int num_samples)
{
    // 每个 warp 计算一个样本
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int warp_num_per_block = blockDim.x >> 5;
    int row_id = blockIdx.x * warp_num_per_block + warp_id;
    if (row_id < num_samples) {
        const DataType *row_data = d_data + row_id * num_features;
        float min_dist = 1e9f;
        float dist;
        int min_idx = 999;
        int cluster_offset = 0;

#pragma unroll
        for (int i = 0; i < num_clusters; ++i)
        {
            dist = calDistV3<DataType>(row_data, d_clusters + cluster_offset, num_features, lane_id);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_idx = i;
            }
            cluster_offset += num_features;
        }

        if (lane_id == 0)
        {
            // printf("sample_id: %d  min_idx: %d  min_dist: %g\n", row_id, min_idx, min_dist);
            d_sample_classes[row_id] = min_idx;
            d_min_dist[row_id] = sqrtf(min_dist);
            atomicAdd(&(d_clusterSize[min_idx]), 1);
        }
    }
}

template <typename DataType, int pack_size>
__global__ void calClustersDistWarpPackedkernel(const DataType *d_data,
                                                const DataType *d_clusters, // [num_clusters, num_features]
                                                int *d_sample_classes,      // [nsamples, ]
                                                float *d_min_dist,          // [nsamples, ]
                                                int *d_clusterSize,         // [nsamples, ]
                                                const int num_features,
                                                const int num_clusters,
                                                const int num_samples)
{
    // 每个 warp 计算一个样本
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int warp_num_per_block = blockDim.x >> 5;
    int row_id = blockIdx.x * warp_num_per_block + warp_id;
    if (row_id < num_samples) {
        const DataType *row_data = d_data + row_id * num_features;
        float min_dist = 1e9f;
        float dist;
        int min_idx;
        int cluster_offset = 0;

#pragma unroll
        for (int i = 0; i < num_clusters; ++i)
        {
            dist = calDistWarpPacked<DataType, pack_size>(row_data, d_clusters + cluster_offset, num_features, lane_id);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_idx = i;
            }
            cluster_offset += num_features;
        }

        if (lane_id == 0)
        {
            d_sample_classes[row_id] = min_idx;
            d_min_dist[row_id] = sqrtf(min_dist);
            atomicAdd(&(d_clusterSize[min_idx]), 1);
        }
    }
}


template <template <typename> class ReductionOp>
__global__ void vec1DReduce(float *vec, float *reduce, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;

    auto func = ReductionOp<float>();

#pragma unroll
    for (; n < N; n += blockDim.x * gridDim.x)
        val = func(val, vec[n]);
    __syncthreads();

    float block_sum = blockReduce<ReductionOp, float>(val);
    if (threadIdx.x == 0)
        reduce[blockIdx.x] = block_sum;
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

    // Allocate temporary storage
    // cudaMallocAsync((void **)&d_temp_storage, temp_storage_bytes, stream);

    // Run reduction, use the preallocte buffer instead of allocate temporary storage
    cub::DeviceReduce::Reduce((void *)d_cache_buf, temp_storage_bytes, d_in, d_out, num_items, op_func, init, stream);
}


__global__ void histCount(int *d_sample_classes, // [N, ]
                          int *d_clusterSize,    // [num_clusters, ]
                          const int num_clusters, const int N)
{
    // block_size = 256, grid_size = (num_samples - 1) / block_size + 1;
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int s_histo[256];
    if (threadIdx.x < num_clusters)
        s_histo[threadIdx.x] = 0;
    __syncthreads();

#pragma unroll
    for (; n < N; n += gridDim.x * blockDim.x)
    {
        atomicAdd(&s_histo[d_sample_classes[n]], 1);
    }
    __syncthreads();
    if (threadIdx.x < num_clusters)
        atomicAdd(&d_clusterSize[threadIdx.x], s_histo[threadIdx.x]);
}

template <typename DataType>
__global__ void update(
    const DataType *d_data,
    DataType *d_clusters,
    int *d_sampleClasses,
    int *d_cluster_size,
    const int num_samples,
    const int num_features)
{
    // grid_size = num_samples, block_size = block_size
    int clusterId = d_sampleClasses[blockIdx.x];
    int clustercnt = d_cluster_size[clusterId];

#pragma unroll
    for (int i = threadIdx.x; i < num_features; i += blockDim.x)
    {
        atomicAdd(&(d_clusters[clusterId * num_features + i]), d_data[num_features * blockIdx.x + i] / clustercnt);
    }
}

/**
 * @brief 矩阵转置，输入矩阵为行主序， from (m,n) to (n,m)
 *
 * @tparam T
 * @param dst       输出矩阵
 * @param src       源矩阵
 * @param m         矩阵行数
 * @param n         矩阵列数
 * gird: ((m+31)/32, (n+31)/32)
 * block: (32, 32)
 * @return __global__
 */
template <typename T>
__global__ void transposMatrixKernel(T *dst, const T *src, const int m, const int n)
{
	__shared__ T s_tile[32][33];
	int bx = blockIdx.x * 32;
	int by = blockIdx.y * 32;
	int x = bx + threadIdx.x;
	int y = by + threadIdx.y;
	//s_tile[threadIdx.y][threadIdx.x] = (T)(0.0f);
	if (x < n && y < m) {
		s_tile[threadIdx.y][threadIdx.x] = src[y * n + x];
	}
	__syncthreads();

	y = bx + threadIdx.y;
	x = by + threadIdx.x;
	if (x < m && y < n) {
		dst[y * m + x] = s_tile[threadIdx.x][threadIdx.y];
	}
}


template <typename DataType>
void launchFit(const DataType *d_data, DataType *d_clusters, int *d_sample_classes,
               int *d_cluster_size, float *d_min_dist, float *d_loss, char *d_cache_buf, const int num_clusters,
               const int num_samples, const int num_features, cudaStream_t calculate_stream,
               cudaStream_t update_stream, cudaEvent_t calculate_event, cudaEvent_t update_event)
{
    
    initV2<int><<<1, 1024, 0, calculate_stream>>>(d_cluster_size, 0.0f, num_clusters);

    if (num_features % 2 && (sizeof(DataType) * 2 <= 128))
    {
        calClustersDistPackedkernel<DataType, 2><<<num_samples, block_size, 0, calculate_stream>>>(d_data, d_clusters,
            d_sample_classes, d_min_dist, d_cluster_size, num_features, num_clusters);
    }
    else
    {
        int warp_num_per_block = block_size >> 5; 
        int grid_size = (num_samples + warp_num_per_block - 1) / warp_num_per_block;

        calClustersDistWarpkernel<DataType><<<grid_size, block_size, 0, calculate_stream>>>(d_data, d_clusters,
            d_sample_classes, d_min_dist, d_cluster_size, num_features, num_clusters, num_samples);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(calculate_event, calculate_stream));
    
    // vec1DReduce<SumOp><<<block_size, block_size, 0, calculate_stream>>>(d_min_dist, d_loss, num_samples);
    // vec1DReduce<SumOp><<<1, block_size, 0, calculate_stream>>>(d_loss, d_loss, block_size);

    cubDeviceReduce<SumOp, float>(d_min_dist, d_loss, num_samples, float(0), d_cache_buf, calculate_stream);

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(update_stream, calculate_event));

    initV2<DataType><<<1, 1024, 0, update_stream>>>(d_clusters, 0.0f, num_clusters * num_features);
    update<DataType><<<num_samples, block_size, 0, update_stream>>>(d_data, d_clusters,
                                                                    d_sample_classes, d_cluster_size, num_samples, num_features);
    CHECK_CUDA_ERROR(cudaEventRecord(update_event, update_stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(calculate_stream, update_event));
} 


}   // clustering

