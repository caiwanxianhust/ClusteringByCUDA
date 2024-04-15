#ifndef KMEANS_H
#define KMEANS_H


namespace clustering {

// using DataType = float;

template <typename DataType>
class AbstractKmeans {
public:
    
    virtual void fit(const DataType *) = 0;
    virtual float accuracy(int *) = 0;
    virtual ~AbstractKmeans() {}
};

template <typename DataType>
class KmeansGPU : public AbstractKmeans<DataType>
{
public:
    KmeansGPU(DataType *h_clusters, int num_clusters, int num_features, int num_samples, int max_iters = 50);
    virtual ~KmeansGPU();
    void fit(const DataType *v_data);
    float accuracy(int *label);

    int m_num_clusters;
    int m_num_features;
    int m_num_samples;
    int m_max_iters;
    float m_eplison;
    float m_optTarget;

    DataType *m_clusters;
    int *m_sample_classes;
    DataType *d_data;                       // [num_samples, num_features]
    DataType *d_clusters;                   // [num_clusters, num_features]
    int *d_sample_classes;                  // [num_samples, ]
    float *d_min_dist;                      // [num_samples, ]
    float *d_loss;                          // [num_samples, ]
    int *d_cluster_size;                    // [num_clusters, ]

private:
    KmeansGPU(const KmeansGPU &model);
    KmeansGPU &operator=(const KmeansGPU &model);
};

}   // clustering


#endif