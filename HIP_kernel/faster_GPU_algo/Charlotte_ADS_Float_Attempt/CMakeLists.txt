cmake_minimum_required(VERSION 3.24)
project(ads_float_hip LANGUAGES CXX HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()

add_executable(sssp_hip
  kernel.hip
  csr_graph.cpp
  support.cpp
)

# All translation units include HIP device declarations.
set_source_files_properties(
  kernel.hip csr_graph.cpp support.cpp
  PROPERTIES LANGUAGE HIP
)

set_target_properties(sssp_hip PROPERTIES
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED YES
  HIP_EXTENSIONS NO
)

target_compile_options(sssp_hip PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3>
  $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64>
  $<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-variable>
)
