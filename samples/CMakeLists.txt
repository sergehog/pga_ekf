cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

set(OPENZEN_CXX17)
set(ZEN_BLUETOOTH)
set(ZEN_USE_STATIC_LIBS)
add_definitions(-DOPENZEN_CXX17)
add_definitions(-DZEN_BLUETOOTH)
add_definitions(-DZEN_USE_STATIC_LIBS)
add_subdirectory(openzen/openzen)

add_executable(openzen_demo openzen/openzen_demo.cpp)
target_compile_definitions(openzen_demo PRIVATE ZEN_API_STATIC)
target_link_libraries (openzen_demo Eigen3::Eigen OpenZen)
target_compile_features(openzen_demo PRIVATE cxx_std_17 )

find_package(ament_cmake)
if (ament_cmake_FOUND)
    add_subdirectory(ros2_ekf_sample)
endif()