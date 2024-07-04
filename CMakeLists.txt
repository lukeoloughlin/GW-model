cmake_minimum_required(VERSION 3.15...3.29)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


project(GreensteinWinslow LANGUAGES CXX)

target_include_directories(GreensteinWinslow PRIVATE src/includes)
target_include_directories(GreensteinWinslow PRIVATE extern/eigen3)
target_include_directories(GreensteinWinslow PRIVATE extern/pybind11)

if(APPLE)
    add_compile_options(
        -Wall
        $<$<CONFIG:RELEASE>:-Ofast>
        $<$<CONFIG:RELEASE>:-mcpu=apple-m1>
        $<$<CONFIG:RELEASE>:-march=native>
        $<$<CONFIG:RELEASE>:-fopenmp>
        $<$<CONFIG:RELEASE>:-funroll-loops>
        $<$<CONFIG:RELEASE>:-flto>
        $<$<CONFIG:DEBUG>:-g>
        $<$<CONFIG:DEBUG>:-Werror>
    )
else()
    add_compile_options(
        -Wall
        $<$<CONFIG:RELEASE>:-Ofast>
        $<$<CONFIG:RELEASE>:-march=native>
        $<$<CONFIG:RELEASE>:-fopenmp>
        $<$<CONFIG:RELEASE>:-funroll-loops>
        $<$<CONFIG:RELEASE>:-flto>
        $<$<CONFIG:DEBUG>:-g>
        $<$<CONFIG:DEBUG>:-Werror>
    )
endif()
add_link_options(-fopenmp)


set(PYBIND11_FINDPYTHON ON)
if(APPLE)
    find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
else()
    find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
endif()
#find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(GreensteinWinslow MODULE pyGW.cpp)

#find_package(Eigen3 3.4 REQUIRED)
#if(Eigen3_FOUND)
#    MESSAGE(STATUS "Found Eigen3!")
#    get_target_property(inc_dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
#    message("[DEBUG] inc_dir: ${inc_dir}")
#else()
#    MESSAGE(STATUS "Failed to find Eigen3")
#endif()

#target_link_libraries(GreensteinWinslow PRIVATE Eigen3::Eigen)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(GreensteinWinslow PRIVATE OpenMP::OpenMP_CXX)
endif()


install(TARGETS GreensteinWinslow DESTINATION ..)

