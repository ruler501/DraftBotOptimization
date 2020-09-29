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
cmake_minimum_required (VERSION 3.18)

set(OPTIMIZATION_ALGORITHMS "Genetic;CrossEntropy;DifferentialEvolution;CmaEs" CACHE STRING "The algorithms to compile")

option(USE_SYCL "Use the SYCL implementation" OFF)
option(OPTIMIZE_RATINGS "Optimize ratings as well as weights" OFF)
option(USE_ASAN "Compile with AddressSanitizer" OFF)
option(CONSIDER_NON_BASICS "Consider non basic lands in calculations" OFF)
option(USE_CUDA "USE_ASAN the CUDA Implementation" OFF)

if (USE_CUDA)
    project (DraftBotOptimization LANGUAGES CXX CUDA
                                  VERSION 0.0.1)
else()
    project (DraftBotOptimization LANGUAGES CXX
                                  VERSION 0.0.1)
endif()
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

foreach(ALGORITHM IN LISTS OPTIMIZATION_ALGORITHMS)
    set(TARGET_NAME "DraftBot${ALGORITHM}Optimization")
    add_executable(${TARGET_NAME} draftbot_optimization.cpp parse_json.cpp algorithms/shared/util.cpp draftbot_optimization.h algorithms/shared/util.h algorithms/shared/parameters.h)
    if(ALGORITHM STREQUAL "Genetic")
        target_sources(${TARGET_NAME} PRIVATE algorithms/genetic.cpp)
    elseif(ALGORITHM STREQUAL "CrossEntropy")
        target_sources(${TARGET_NAME} PRIVATE algorithms/cross_entropy_method.cpp)
    elseif(ALGORITHM STREQUAL "DifferentialEvolution")
        target_sources(${TARGET_NAME} PRIVATE algorithms/differential_evolution.cpp)
    elseif(ALGORITHM STREQUAL "CmaEs")
        target_sources(${TARGET_NAME} PRIVATE algorithms/cma_es.cpp algorithms/shared/matrix_types.h)
        find_package(CUDAToolkit)
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
        target_include_directories (${TARGET_NAME} PRIVATE "${CUDA_DIR}/include")
    endif()
    if (USE_CUDA)
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_CUDA)
        target_sources(${TARGET_NAME} PRIVATE draftbot_optimization.cu)
        set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 75)
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math --extra-device-vectorization -lineinfo>)
    endif()
    if (CONSIDER_NON_BASICS)
        target_compile_definitions(${TARGET_NAME} PRIVATE CONSIDER_NON_BASICS)
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
    if(OPTIMIZE_RATINGS)
        target_compile_definitions(${TARGET_NAME} PRIVATE OPTIMIZE_RATINGS)
    endif()
    if (MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-W4 /arch:AVX2>)
        set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
        if (USE_ASAN)
            target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=address)
            target_link_options(${TARGET_NAME} PRIVATE /debug -incremental:no /wholearchive:clang_rt.asan_dynamic-x86_64.lib /wholearchive:clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
        endif()
    else()
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native -ffast-math>)
    endif()

    set_target_properties (${TARGET_NAME} PROPERTIES CXX_STANDARD 20
                                                     CUDA_STANDARD 14
                                                     CXX_STANDARD_REQUIRED ON
                                                     CXX_EXTENSIONS OFF)
endforeach()

if(USE_SYCL)
    foreach(ALGORITHM IN LISTS OPTIMIZATION_ALGORITHMS)
        set(TARGET_NAME "DraftBot${ALGORITHM}Optimization")
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_SYCL)
        target_include_directories (${TARGET_NAME} PRIVATE extern/concurrentqueue C:/opt/include)
        target_compile_options(${TARGET_NAME} PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fno-sycl-libspirv --cuda-path=C:/PROGRA~1/NVIDIA~2/CUDA/v10.1/ -Xsycl-target-backend -cuda-gpu-arch=sm_75 -v)
    endforeach()
endif()
