cmake_minimum_required(VERSION 3.15)
project(pga_ekf)

set(CMAKE_CXX_STANDARD 17)
find_package (Eigen3 3.3 REQUIRED)

include_directories(include)

enable_testing()
add_executable(pga_ekf_test test/pga_ekf_test.cpp)
add_test(pga_ekf_test pga_ekf_test)
target_link_libraries (pga_ekf_test Eigen3::Eigen gtest gtest_main pthread)
