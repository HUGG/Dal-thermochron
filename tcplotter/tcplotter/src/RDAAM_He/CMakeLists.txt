cmake_minimum_required(VERSION 3.10)

# set the project name
project(RDAAM_He)

# Configure
#set(CMAKE_OSX_ARCHITECTURES "x86_64")

# add the library
add_library(RDAAM_He SHARED
RDAAM.cpp
)
