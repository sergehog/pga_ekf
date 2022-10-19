cmake_minimum_required(VERSION 3.15)
project(pga_ekf)

set(CMAKE_CXX_STANDARD 17)
find_package (Eigen3 3.3 REQUIRED)

include_directories(include)

option(BUILD_SAMPLES "Try building samples" ON)

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

enable_testing()
add_executable(pga_ekf_test test/pga_ekf_test.cpp test/pga_ekf_update_imu_test.cpp test/pga_ekf_predict_test.cpp  test/pga_ekf_update_enu_test.cpp)
add_test(pga_ekf_test pga_ekf_test)
target_link_libraries (pga_ekf_test Eigen3::Eigen gtest gtest_main pthread)
