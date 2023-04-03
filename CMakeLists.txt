cmake_minimum_required(VERSION 3.26)
project(StreamSimulator VERSION 1.0)

set(CMAKE_PREFIX_PATH "C:/Qt/6.4.2/mingw_64/lib/cmake")
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

find_package(CUDA REQUIRED)


file(GLOB UI_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/ui_module/*.hpp")
file(GLOB UI_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/ui_module/*.cpp")

file(GLOB LBM_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/lbm_module/*.hpp")
file(GLOB LBM_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/lbm_module/*.cpp")
file(GLOB LBM_KERNEL_FILES "${PROJECT_SOURCE_DIR}/src/lbm_module/*.cu")

add_library(stream_ui STATIC ${UI_HEADER_FILES} ${UI_SOURCE_FILES})
add_library(stream_lbm STATIC ${LBM_HEADER_FILES} ${LBM_SOURCE_FILES} ${LBM_KERNEL_FILES})

target_link_libraries(stream_ui PRIVATE Qt6::Core PRIVATE Qt6::Widgets)	
target_link_libraries(stream_lbm ${CUDA_LIBRARIES})												
							
target_include_directories(stream_ui PUBLIC "include/ui_module" ${Qt6_INCLUDE_DIRS})
target_include_directories(stream_lbm PUBLIC "include/lbm_module" ${CUDA_INCLUDE_DIRS})

add_executable(StreamSimulator src/main.cpp)
target_link_libraries(StreamSimulator PUBLIC stream_ui
									  PUBLIC stream_lbm)
