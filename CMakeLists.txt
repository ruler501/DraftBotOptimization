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
cmake_minimum_required (VERSION 3.2.2)

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
add_subdirectory(extern/json)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(OPTIMIZATION_ALGORITHMS Genetic CrossEntropy DifferentialEvolution)

foreach(ALGORITHM IN LISTS OPTIMIZATION_ALGORITHMS)
    set(TARGET_NAME "DraftBot${ALGORITHM}Optimization")
    add_executable(${TARGET_NAME} draftbot_optimization.cpp parse_json.cpp algorithms/shared/util.cpp)
    if(ALGORITHM STREQUAL "Genetic")
        target_sources(${TARGET_NAME} PRIVATE algorithms/genetic.cpp)
    elseif(ALGORITHM STREQUAL "CrossEntropy")
        target_sources(${TARGET_NAME} PRIVATE algorithms/cross_entropy_method.cpp)
    elseif(ALGORITHM STREQUAL "DifferentialEvolution")
        target_sources(${TARGET_NAME} PRIVATE algorithms/differential_evolution.cpp)
    endif()
    target_include_directories(${TARGET_NAME} PRIVATE extern/concurrentqueue)

    target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json Threads::Threads)

    if (MSVC)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -O2 -g -march=native)
        target_link_options(${TARGET_NAME} PRIVATE -O2 -g)
    endif()

    set_target_properties (${TARGET_NAME} PROPERTIES CXX_STANDARD 14
                                                     CXX_STANDARD_REQUIRED ON
                                                     CXX_EXTENSIONS OFF)
endforeach()

option(USE_SYCL "Use the SYCL implementation" OFF)

if(USE_SYCL)
    # Behavioural options for the project
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(ComputeCpp)
    foreach(ALGORITHM IN LISTS OPTIMIZATION_ALGORITHMS)
        set(TARGET_NAME "DraftBot${ALGORITHM}Optimization")
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_SYCL)
        target_include_directories (${TARGET_NAME} PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include")
        add_sycl_to_target(TARGET ${TARGET_NAME}
                           SOURCES draftbot_optimization.cpp)
    endforeach()
endif()