###############################################################################
#
# Top level CMakeLists.txt file for the SYCL-RNG repo
# Hacker: Mate Ferenc Nagy-Egri
# Last modified: 2018.08.01. (yyyy.mm.dd.)
#
###############################################################################

#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of 'cmake' necessary to build this project
cmake_minimum_required (VERSION 3.17.1)

# The project name and version, release numbers
project (DraftBotOptimization LANGUAGES CXX
                              VERSION 0.0.1)

# Typically you don't care so much for a third party library's tests to be
# run from your own project's code.
set(JSON_BuildTests OFF CACHE INTERNAL "")
# If you only include this third party in PRIVATE source files, you do not
# need to install it when your main project gets installed.
set(JSON_Install OFF CACHE INTERNAL "")
# Disable Implicit JSON conversions
set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
# Don't use include(nlohmann_json/CMakeLists.txt) since that carries with it
# unintended consequences that will break the build.  It's generally
# discouraged (although not necessarily well documented as such) to use
# include(...) for pulling in other CMake projects anyways.
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

add_subdirectory(extern/json)
find_package(Threads REQUIRED)
find_package(OpenMP)

set(OPTIMIZATION_ALGORITHMS Genetic CrossEntropy DifferentialEvolution CmaEs)

option(USE_SYCL "Use the SYCL implementation" OFF)
option(OPTIMIZE_RATINGS "Optimize ratings as well as weights" OFF)

foreach(ALGORITHM IN LISTS OPTIMIZATION_ALGORITHMS)
    set(TARGET_NAME "DraftBot${ALGORITHM}Optimization")
    add_executable(${TARGET_NAME} draftbot_optimization.cpp parse_json.cpp algorithms/shared/util.cpp)
    if(ALGORITHM STREQUAL "Genetic")
        target_sources(${TARGET_NAME} PRIVATE algorithms/genetic.cpp)
    elseif(ALGORITHM STREQUAL "CrossEntropy")
        target_sources(${TARGET_NAME} PRIVATE algorithms/cross_entropy_method.cpp)
    elseif(ALGORITHM STREQUAL "DifferentialEvolution")
        target_sources(${TARGET_NAME} PRIVATE algorithms/differential_evolution.cpp)
    elseif(ALGORITHM STREQUAL "CmaEs")
        target_sources(${TARGET_NAME} PRIVATE algorithms/cma_es.cpp)
        target_link_libraries(${TARGET_NAME} PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cudart.lib"
                                                     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cublas.lib"
                                                     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cusparse.lib"
                                                     C:/opt/lib/openblas.lib C:/opt/lib/magma.lib C:/opt/lib/magma_sparse.lib)
        target_include_directories (${TARGET_NAME} PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include")
    endif()
    target_include_directories(${TARGET_NAME} PRIVATE extern/concurrentqueue C:/opt/include)

    target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(OPTIMIZE_RATINGS)
        target_compile_definitions(${TARGET_NAME} PRIVATE OPTIMIZE_RATINGS)
    endif()

    if (MSVC)
        target_link_options(${TARGET_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
        target_compile_options(${TARGET_NAME} PRIVATE /DWIN32 /D_WINDOWS -W4 -MP -DMAGMA_NOAFFINITY /FS)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -O2 -g -march=native)
        target_link_options(${TARGET_NAME} PRIVATE -O2 -g)
    endif()

    set_target_properties (${TARGET_NAME} PROPERTIES CXX_STANDARD 14
                                                     CXX_STANDARD_REQUIRED ON
                                                     CXX_EXTENSIONS OFF)
endforeach()

if(USE_SYCL)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(ComputeCpp)
    foreach(ALGORITHM IN LISTS OPTIMIZATION_ALGORITHMS)
        set(TARGET_NAME "DraftBot${ALGORITHM}Optimization")
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_SYCL EIGEN_USE_SYCL=1 EIGEN_SYCL_ASYNC_EXECUTION=1  EIGEN_NO_CUDA)
        target_include_directories (${TARGET_NAME} PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include")
        add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES draftbot_optimization.cpp)
    endforeach()
endif()