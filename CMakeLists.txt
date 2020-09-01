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

# Behavioural options for the project
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(ComputeCpp)

add_executable(DraftBotOptimization draftbot_optimization.cpp)

target_include_directories (DraftBotOptimization PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include")

target_link_libraries(DraftBotOptimization PRIVATE nlohmann_json::nlohmann_json)

add_sycl_to_target(TARGET DraftBotOptimization
                   SOURCES draftbot_optimization.cpp)

set_target_properties (DraftBotOptimization PROPERTIES CXX_STANDARD 20
                                                       CXX_STANDARD_REQUIRED ON
                                                       CXX_EXTENSIONS OFF)