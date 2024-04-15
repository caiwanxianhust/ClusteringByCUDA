set (cuda_files
    kmeans.cu
)

set (KEMANS_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/kmeans
    ${PROJECT_SOURCE_DIR}/kmeans/cuda
)

include_directories(
    ${KEMANS_HEADER_DIR}
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/kmeans/lib)

add_library(kmeans SHARED ${cuda_files})

# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(kmeans PUBLIC cxx_std_14)

message("-- Assign kmeans include directories (include_directories=${KEMANS_HEADER_DIR})")

target_compile_options(kmeans PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")

message("-- in kmeans cuda Assign arch (arch=${CMAKE_CUDA_ARCHITECTURES})")

set_target_properties(kmeans PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(kmeans PRIVATE cublas)
# target_link_libraries(kmeans PRIVATE cublasLt)
target_link_libraries(kmeans PRIVATE cudart)
