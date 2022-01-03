cmake_minimum_required(VERSION 3.20)
project(ppnlm)

set(CMAKE_CXX_STANDARD 23)

add_executable(ppnlm main.cpp)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppnlm ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

