cmake_minimum_required(VERSION 3.5)
PROJECT(tensorrt_detection VERSION 1.0.0 LANGUAGES C CXX CUDA)
SET(CMAKE_BUILD_TYPE Debug)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)

file(GLOB CPPS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${SAMPLES_COMMON_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../util/*.cpp)
find_package(OpenCV REQUIRED)

include_directories(${INCLUDE_DRIS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../)
add_executable(${PROJECT_NAME} ${CPPS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES})