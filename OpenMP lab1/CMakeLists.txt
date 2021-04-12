cmake_minimum_required(VERSION 3.17)
project(lab2_ver1 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

add_executable(lab2_ver1 main.c)