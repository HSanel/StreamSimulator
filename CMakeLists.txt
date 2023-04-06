cmake_minimum_required(VERSION 3.26)
project(StreamSimulator VERSION 1.0 LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CONFIGURATION_TYPES "Release;RelWithDebInfo;Debug" CACHE STRING "" FORCE)

#==================================== Inclusion of Qt ============================#
set(CMAKE_PREFIX_PATH $ENV{QTDIR})
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL Gui)

#==================================== Inclusion of CUDA ==========================#
find_package(CUDA REQUIRED)			 

#==================================== Inclusion of VTK ===========================#
set(VTK_DIR $ENV{VTK_DIR})
find_package(VTK REQUIRED)


#=================================== Inclusion of own code =======================#
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include")

file(GLOB UI_HEADER_FILES 	"${INC_DIR}/ui_module/*.hpp" 
							"${INC_DIR}/ui_module/*.h")
file(GLOB UI_SOURCE_FILES 	"${SRC_DIR}/ui_module/*.cpp")
file(GLOB UI_GUI_FILES 		"${SRC_DIR}/ui_module/*.ui")

file(GLOB LBM_HEADER_FILES "${INC_DIR}/lbm_module/*.hpp" 
						   "${INC_DIR}/lbm_module/*.h")
file(GLOB LBM_SOURCE_FILES "${SRC_DIR}/lbm_module/*.cpp")
file(GLOB LBM_KERNEL_FILES "${SRC_DIR}/lbm_module/*.cu")

file(GLOB RENDERER_HEADER_FILES "${INC_DIR}/renderer_module/*.hpp" 
								"${INC_DIR}/renderer_module/*.h")
file(GLOB RENDERER_SOURCE_FILES "${SRC_DIR}/renderer_module/*.cpp")

add_library(stream_ui STATIC ${UI_HEADER_FILES} ${UI_SOURCE_FILES} ${UI_GUI_FILES})
add_library(stream_lbm STATIC ${LBM_HEADER_FILES} ${LBM_SOURCE_FILES} ${LBM_KERNEL_FILES})
add_library(stream_renderer STATIC ${RENDERER_HEADER_FILES} ${RENDERER_SOURCE_FILES})

target_include_directories(stream_ui 	PUBLIC "${INC_DIR}/ui_module")
target_include_directories(stream_lbm 	PUBLIC "${INC_DIR}/lbm_module" 
										PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(stream_renderer 	PUBLIC "${INC_DIR}/renderer_module")

target_link_libraries(stream_ui Qt6::Core Qt6::Widgets Qt6::OpenGL Qt6::Gui)	
target_link_libraries(stream_lbm ${CUDA_LIBRARIES})												
target_link_libraries(stream_renderer ${VTK_LIBRARIES})							


add_executable(StreamSimulator src/main.cpp)

target_include_directories(StreamSimulator PRIVATE include)

target_link_libraries(StreamSimulator PUBLIC stream_ui
									  PUBLIC stream_lbm)

