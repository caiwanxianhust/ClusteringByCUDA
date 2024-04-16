#ifndef FUZYCMEANS_H
#define FUZYCMEANS_H


namespace clustering {
namespace fuzzyCmeans {

template <typename DataType>
class AbstractFuzzyCmeans {
public:
    
    virtual void fit(const DataType *) = 0;
    virtual ~AbstractFuzzyCmeans() {}
};

template <typename DataType>
class FuzzyCmeans : public AbstractFuzzyCmeans<DataType>
{
public:
    FuzzyCmeans(float *h_membership, int num_clusters, int num_features, int num_samples, float scale, int max_iters = 50);
    virtual ~FuzzyCmeans();
    void fit(const DataType *v_data);

    int m_num_clusters;
    int m_num_features;
    int m_num_samples;
    int m_max_iters;
    float m_eplison;
    float m_optTarget;
    float m_scale;

    DataType *m_clusters;
    float *m_membership;
    int *m_labels;
    int *d_labels;
    float *d_membership;
    DataType *d_data;                       // [num_samples, num_features]
    DataType *d_clusters;                   // [num_clusters, num_features]
    float *d_distances;                     // [num_samples, num_clusters]
    float *d_loss;                          // [1]

private:
    FuzzyCmeans(const FuzzyCmeans &model);
    FuzzyCmeans &operator=(const FuzzyCmeans &model);
};

}   // fuzzyCmeans
}   // clustering


#endif