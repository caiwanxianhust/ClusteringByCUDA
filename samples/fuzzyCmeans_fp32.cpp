#include "clustering.h"
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

template <typename T>
void printVecInVec(const T *clusters, const int n, const int m, const int end_n, const int end_m, const char *str);


template <typename T>
void printVecInVec(const T *clusters, const int n, const int m, const int end_n, const int end_m, const char *str)
{
    printf("%s:\n[\n", str);
    for (int i = 0; i < end_n; ++i)
    {
        printf("[");
        for (int j = 0; j < end_m; ++j)
        {
            printf("%g  ", (float)clusters[i * m + j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

int readCoordinate(const char *file, float *data, int *label, const int n_features, int N) {
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail()) {
        printf("No such file or directory: %s\n", file);
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

int readMembership(const char *file, float *memship, const int n_samples, const int n_clusters) 
{
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail()) {
        printf("No such file or directory: %s\n", file);
        exit(1);
    }
    std::string line;
    int n = 0;
    int m;
    while (std::getline(ifs, line) && n < n_samples) {
        std::stringstream sstream(line);
        if (line.empty()) continue;
        m = 0;
        std::string s_fea;
        while (std::getline(sstream, s_fea, ',') && m < n_clusters) {
            memship[m * n_samples + n] = std::stod(s_fea);
            m++;
        }
        n++;
    }
    ifs.close();
    return n;
}

void init_membership(float *membership, const int num_clusters, const int num_samples)
{
    std::srand(1234);
    float rand_val, rand_sum;
    for (int i = 0; i < num_clusters; i++) {
        rand_sum = 0.0f;
        for (int j = 0; j < num_samples; ++j) {
            rand_val = (float)((std::rand() % 1000) / 1000.0f);
            rand_sum += rand_val;
            membership[i * num_samples + j] = rand_val;
        }
        for (int j = 0; j < num_samples; ++j) {
            membership[i * num_samples + j] /= rand_sum;
        }
    }
}



int main()
{
    using DataType = float;
    const int num_samples = 100000;
    const int num_features = 500;
    const int num_clusters = 8;
    float *memship = new float[num_samples * num_clusters];
    const char *memship_file = "./member_ship_1e5_8_for_1e5_500_8.csv";
    readMembership(memship_file, memship, num_samples, num_clusters);
    printVecInVec(memship, num_clusters, num_features, num_clusters, 10, "init memship");

    DataType *data = new DataType[num_samples * num_features];
    int *label = new int[num_samples];
    const char *data_file = "./fuzzycmeans_samples_1e5_500_8.csv";
    readCoordinate(data_file, data, label, num_features, num_samples);
    printf("num of samples : %d\n", num_samples);

    clustering::fuzzyCmeans::FuzzyCmeans<DataType> *model = 
        new clustering::fuzzyCmeans::FuzzyCmeans<DataType>(memship, num_clusters, num_features, num_samples, 2.0f, 50);

    model->fit(data);

    printVecInVec<int>(model->m_labels, 1, num_samples, 1, 30, "pred labels");
    printVecInVec<int>(label, 1, num_samples, 1, 30, "true labels");

    printVecInVec(model->m_membership, num_clusters, num_samples, num_clusters, 10, "pred memship");

    delete model;
    delete [] label;
    delete [] data;
    delete [] memship;
    return 0;
}