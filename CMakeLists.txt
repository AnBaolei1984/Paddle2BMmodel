cmake_minimum_required(VERSION 3.0)
project(demo CXX C)

set(TARGET p2u)
# 1. lib & include
link_directories(${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/lib ${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/lib/third_party)
include_directories(${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/include)
# 2. compile options 
add_definitions(-std=c++11 -g -O3 -DNDEBUG -pthread)
#add_definitions(-g  )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# 3.add executable output
add_executable(${TARGET} main.cc config.cc utility.cc)
target_link_libraries(${TARGET} -lpaddle_full_api_shared -lbmlib -lbmrt -lbmcompiler)
