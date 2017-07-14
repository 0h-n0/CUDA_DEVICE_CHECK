set(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_61,code=sm_61 -Xcompiler -Wall -O0")

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})   
  CUDA_SELECT_NVCC_ARCH_FLAGS(6.1)
  add_definitions(-DBOOST_LOG_DYN_LINK)
  message(STATUS ${PROJECT_SOURCE_DIR}/spdlog/include/spdlog)
  cuda_include_directories(${PROJECT_SOURCE_DIR}/spdlog/include)
  cuda_add_executable(test_memory_allocation test_memory_allocation.cu)
  target_link_libraries(test_memory_allocation ${Boost_LIBRARIES})    
endif()

