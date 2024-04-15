#include "clustering.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>



template <typename T>
void printVecInVec(const T *clusters, const int n, const int m, const std::string &str)
{
    for (int i = 0; i < n; ++i)
    {
        std::cout << str << ' ' << i + 1 << std::endl;
        for (int j = 0; j < m; ++j)
        {
            std::cout << clusters[i * m + j] << "  ";
        }
        std::cout << std::endl;
    }
}


int readCoordinate(const char *file, float *data, int *label, const int n_features, int N) {
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail()) {
        std::cout << "No such file or directory: sample_1e6_fea_100_class_4_lable_1_ninfo_8.csv" << std::endl;
        exit(1);
    }
    std::string line;
    int n = 0;
    while (std::getline(ifs, line) && n < N) {
        std::stringstream sstream(line);
        if (line.empty()) continue;
        int m = 0;
        std::string s_fea;
        while (std::getline(sstream, s_fea, ',')) {
            if (m < n_features) data[n * n_features + m] = std::stod(s_fea);
            else label[n] = std::stoi(s_fea);
            m++;
        }
        n++;
    }
    ifs.close();
    return n;
}

template<typename DataType>
void timing(
    float *data, 
    int *label, 
    float *clusters, 
    const int num_clusters, 
    const int n_features, 
    const int n_samples) {
    
    clustering::KmeansGPU<float> *model = new clustering::KmeansGPU<float>(clusters, num_clusters, n_features, n_samples);

    std::cout << "*********starting fitting*********" << std::endl;

    model->fit(data);

    std::cout << "*********    accuracy  **********" << std::endl;
    std::cout << "model accuracy : " << model->accuracy(label) << std::endl;
    printVecInVec<int>(model->m_sample_classes, 1, 30, "sampleClasses_10");
    printVecInVec<int>(label, 1, 30, "origin labels");

    delete model;
}



void launchDemo(int max_num_samples) {
    constexpr int n_features = 500;
    const int data_buf_size = max_num_samples * n_features;
    float *data = new float[data_buf_size];
    int *label = new int[data_buf_size];

    const char *file = "./sample_1e5_fea_500_class_8_lable_1_ninfo_8.csv";
    int num_samples = readCoordinate(file, data, label, n_features, max_num_samples);
    std::cout << "num of samples : " << num_samples << std::endl;

    int cidx[] = {0, 1, 3, 4, 5, 9, 10, 11};
    constexpr int num_clusters = 8;
    constexpr int cluster_buf_size = num_clusters * n_features;
    float clusters[cluster_buf_size] = {0};
    for (int i=0 ; i<num_clusters; ++i) {
        for (int j=0; j<n_features; ++j) {
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }
    
    timing<float>(data, label, clusters, num_clusters, n_features, num_samples);
    
    delete[] data;
    delete[] label;

}

int main() 
{

    launchDemo(100000);
    return 0;
}