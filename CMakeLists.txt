cmake_minimum_required(VERSION 3.26)
project(StreamSimulator VERSION 1.0 LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CONFIGURATION_TYPES "Release;RelWithDebInfo;Debug" CACHE STRING "" FORCE)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH $ENV{QTDIR})
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
find_package(CUDA REQUIRED)			 


file(GLOB UI_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/ui_module/*.hpp" 
							"${PROJECT_SOURCE_DIR}/include/ui_module/*.h")
file(GLOB UI_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/ui_module/*.cpp")
file(GLOB UI_GUI_FILES "${PROJECT_SOURCE_DIR}/src/ui_module/*.ui")

file(GLOB LBM_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/lbm_module/*.hpp" 
							"${PROJECT_SOURCE_DIR}/include/lbm_module/*.h")
file(GLOB LBM_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/lbm_module/*.cpp")
file(GLOB LBM_KERNEL_FILES "${PROJECT_SOURCE_DIR}/src/lbm_module/*.cu")

add_library(stream_ui STATIC ${UI_HEADER_FILES} ${UI_SOURCE_FILES} ${UI_GUI_FILES})
add_library(stream_lbm STATIC ${LBM_HEADER_FILES} ${LBM_SOURCE_FILES} ${LBM_KERNEL_FILES})

target_include_directories(stream_ui 	PUBLIC "include/ui_module")
target_include_directories(stream_lbm 	PUBLIC "include/lbm_module" 
										PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(stream_ui Qt6::Core Qt6::Widgets)	
target_link_libraries(stream_lbm ${CUDA_LIBRARIES})												
							


add_executable(StreamSimulator src/main.cpp)

target_include_directories(StreamSimulator PRIVATE include)

target_link_libraries(StreamSimulator PUBLIC stream_ui
									  PUBLIC stream_lbm)

