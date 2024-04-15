set (sample_kmeans_fp32_files
    kmeans_fp32.cpp
)

set (sample_fuzzyCmeans_fp32_files
    fuzzyCmeans_fp32.cpp
)

set (SAMPLES_HEADER_DIR
    ${PROJECT_SOURCE_DIR}
)

include_directories(
    ${SAMPLES_HEADER_DIR}
)

link_directories(
    ${PROJECT_SOURCE_DIR}/kmeans/lib
    ${PROJECT_SOURCE_DIR}/fuzzyCmeans/lib
)

add_executable(kmeans_fp32 ${sample_kmeans_fp32_files})

# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(kmeans_fp32 PUBLIC cxx_std_14)

message("-- Assign include directories (include_directories=${SAMPLES_HEADER_DIR})")

target_link_libraries(kmeans_fp32 PUBLIC -lkmeans)


add_executable(fuzzyCmeans_fp32 ${sample_fuzzyCmeans_fp32_files})

# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(fuzzyCmeans_fp32 PUBLIC cxx_std_14)

message("-- Assign include directories (include_directories=${SAMPLES_HEADER_DIR})")

target_link_libraries(fuzzyCmeans_fp32 PUBLIC -lfuzzyCmeans)