CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(test)

set(CMAKE_CXX_STANDARD 14)

add_executable(test main.cpp yolo.cpp yolo.h)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(test ${OpenCV_LIBS})
