cmake_minimum_required(VERSION 3.29)

# project name
project(gw-model)

# set the variable EXEC to gw.out
set(EXEC gw.out)

# set the C++ 20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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


# set the includes path
include_directories(includes)

# set the variable SOURCES to the cpp files in src
file(GLOB SOURCES src/*.cpp)

# compile files in SOURCES and generate the executable EXEC
add_executable(${EXEC} ${SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXEC} PUBLIC OpenMP::OpenMP_CXX)
endif()