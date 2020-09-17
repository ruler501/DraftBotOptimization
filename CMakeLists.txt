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


set(CUDA_DIR "" CACHE PATH "Path to the root of the CUDA install.")

set(OPTIMIZATION_ALGORITHMS "Genetic;CrossEntropy;DifferentialEvolution;CmaEs" CACHE STRING "The algorithms to compile")

option(USE_SYCL "Use the SYCL implementation" OFF)
option(OPTIMIZE_RATINGS "Optimize ratings as well as weights" OFF)
option(USE_ASAN "Compile with AddressSanitizer" OFF)

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
        if (NOT CUDA_DIR)
            message(FATAL_ERROR "You must specify CUDA_DIR when compiling CmaEs or with SYCL")
        endif()
        target_sources(${TARGET_NAME} PRIVATE algorithms/cma_es.cpp algorithms/shared/matrix_types.h)
        target_link_libraries(${TARGET_NAME} PRIVATE "${CUDA_DIR}/lib/x64/cudart"
                                                     "${CUDA_DIR}/lib/x64/cublas"
                                                     "${CUDA_DIR}/lib/x64/cusolver"
                )
        target_include_directories (${TARGET_NAME} PRIVATE "${CUDA_DIR}/include")
    endif()

      target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
    if(OPTIMIZE_RATINGS)
        target_compile_definitions(${TARGET_NAME} PRIVATE OPTIMIZE_RATINGS)
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
    if (MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE -W4)
        if (USE_ASAN)
            target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=address)
            target_link_options(${TARGET_NAME} PRIVATE /debug -incremental:no /wholearchive:clang_rt.asan_dynamic-x86_64.lib /wholearchive:clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
        endif()
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -march=native)
    endif()

    set_target_properties (${TARGET_NAME} PROPERTIES CXX_STANDARD 20
                                                     CXX_STANDARD_REQUIRED ON
                                                     CXX_EXTENSIONS OFF)
endforeach()

if(USE_SYCL)
    if (NOT CUDA_DIR)
        message(FATAL_ERROR "You must specify CUDA_DIR when compiling CmaEs or with SYCL")
    endif()
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(ComputeCpp)
    foreach(ALGORITHM IN LISTS OPTIMIZATION_ALGORITHMS)
        set(TARGET_NAME "DraftBot${ALGORITHM}Optimization")
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_SYCL)
        target_include_directories (${TARGET_NAME} PRIVATE "${CUDA_DIR}/include" extern/concurrentqueue)
        add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES draftbot_optimization.cpp)
    endforeach()
endif()