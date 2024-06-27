cmake_minimum_required(VERSION 3.15...3.29)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection") # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON) # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF) # optional, keep compiler extensions off

project(GreensteinWinslow LANGUAGES CXX)

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

set(PYBIND11_FINDPYTHON ON)

find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

#set(PROJECT_INCLUDE_DIR "includes")
# Source files
#set(SOURCE_FILES pyGW.cpp)

pybind11_add_module(GreensteinWinslow MODULE pyGW.cpp)



find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXEC} OpenMP::OpenMP_CXX)
endif()


install(TARGETS GreensteinWinslow DESTINATION .)




#cmake_minimum_required(VERSION 3.29)

# project name
#project(gw-model)

# set the variable EXEC to gw.out
#set(EXEC gw.out)

# set the C++ 20 standard
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#if(APPLE)
#    add_compile_options(
#        -Wall
#        $<$<CONFIG:RELEASE>:-Ofast>
#        $<$<CONFIG:RELEASE>:-mcpu=apple-m1>
#        $<$<CONFIG:RELEASE>:-march=native>
#        $<$<CONFIG:RELEASE>:-fopenmp>
#        $<$<CONFIG:RELEASE>:-funroll-loops>
#        $<$<CONFIG:RELEASE>:-flto>
#        $<$<CONFIG:DEBUG>:-g>
#        $<$<CONFIG:DEBUG>:-Werror>
#    )
#else()
#    add_compile_options(
#        -Wall
#        $<$<CONFIG:RELEASE>:-Ofast>
#        $<$<CONFIG:RELEASE>:-march=native>
#        $<$<CONFIG:RELEASE>:-fopenmp>
#        $<$<CONFIG:RELEASE>:-funroll-loops>
#        $<$<CONFIG:RELEASE>:-flto>
#        $<$<CONFIG:DEBUG>:-g>
#        $<$<CONFIG:DEBUG>:-Werror>
#    )
#endif()


# set the includes path
#include_directories(includes)

# set the variable SOURCES to the cpp files in src
#file(GLOB SOURCES src/*.cpp)

# compile files in SOURCES and generate the executable EXEC
#add_executable(${EXEC} ${SOURCES})

#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(${EXEC} PUBLIC OpenMP::OpenMP_CXX)
#endif()