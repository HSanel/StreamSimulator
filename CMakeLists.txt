cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
project(StreamSimulator LANGUAGES CUDA CXX)

enable
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CONFIGURATION_TYPES "Release;RelWithDebInfo;Debug" CACHE STRING "" FORCE)



#==================================== Inclusion of Qt ============================#
set(CMAKE_PREFIX_PATH "$ENV{QTDIR}")
set(VTK_DIR $ENV{VTK_DIR})

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL Gui)

# Inclusion CUDA

#find_package(CUDAToolkit REQUIRED)
#==================================== Inclusion of VTK ===========================#

find_package(VTK REQUIRED)

#=================================== Inclusion of own code =======================#
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include")

file(GLOB UI_HEADER_FILES 	"${INC_DIR}/ui_module/*.hpp" 
							"${INC_DIR}/ui_module/*.h")
file(GLOB UI_SOURCE_FILES 	"${SRC_DIR}/ui_module/*.cpp")
file(GLOB UI_GUI_FILES 		"${SRC_DIR}/ui_module/*.ui")

file(GLOB LBM_HEADER_FILES	"${INC_DIR}/lbm_module/*.hpp" 
							"${INC_DIR}/lbm_module/*.h")
file(GLOB LBM_KERNEL_FILES	"${INC_DIR}/lbm_module/*.cu")

file(GLOB LBM_SOURCE_FILES	"${SRC_DIR}/lbm_module/*.cpp")



add_library(stream_ui  STATIC ${UI_HEADER_FILES} ${UI_SOURCE_FILES} ${UI_GUI_FILES})
add_library(stream_lbm STATIC ${LBM_HEADER_FILES}  ${LBM_SOURCE_FILES} )
add_library(stream_lbm_gpu ${LBM_KERNEL_FILES})

#enable_language(CUDA)
target_link_libraries(stream_lbm ${stream_lbm_gpu})
set_property(TARGET stream_lbm PROPERTY CUDA_SEPARABLE_COMPILATION ON)


target_include_directories(stream_ui 	PUBLIC "${INC_DIR}/ui_module")
target_include_directories(stream_lbm 	PUBLIC "${INC_DIR}/lbm_module" 
										PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(stream_ui Qt6::Core Qt6::Widgets Qt6::OpenGL Qt6::Gui ${VTK_LIBRARIES})	
target_link_libraries(stream_lbm PUBLIC CUDA::cudart 
								 PUBLIC CUDA::cuda_driver)
											


add_executable(StreamSimulator src/main.cpp)
target_include_directories(StreamSimulator PRIVATE "${INC_DIR}")

target_link_libraries(StreamSimulator PUBLIC stream_lbm PUBLIC stream_ui)

