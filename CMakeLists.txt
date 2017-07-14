cmake_minimum_required(VERSION 3.3)
project(check_gpu_device)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS program_options)
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
add_subdirectory(src)

