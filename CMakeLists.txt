cmake_minimum_required(VERSION 3.0)
project(demo CXX C)

if (PADDLE_TO_UMODEL)
  set(TARGET p2u)
else()
  set(TARGET p2b)
endif()
# 1. lib & include
if (PADDLE_TO_UMODEL)
  link_directories(${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/lib/umodel ${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/lib/third_party)
else()
  link_directories(${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/lib/bmodel ${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/lib/third_party)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/inference_lite_lib/cxx/include)
# 2. compile options 
add_definitions(-std=c++11 -g -O3 -DNDEBUG -pthread)
#add_definitions(-g  )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# 3.add executable output
add_executable(${TARGET} src/main.cc src/config.cc src/utility.cc)

target_link_libraries(${TARGET} -lpaddle_full_api_shared -lbmlib -lbmrt -lbmcompiler)
