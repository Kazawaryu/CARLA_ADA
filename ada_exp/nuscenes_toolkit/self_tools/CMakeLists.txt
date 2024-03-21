cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(test_read_bin)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (test_read_bin test_read_bin.cpp)
target_link_libraries (test_read_bin ${PCL_LIBRARIES})
