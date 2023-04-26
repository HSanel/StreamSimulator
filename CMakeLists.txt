cmake_minimum_required(VERSION 3.9)
project(StreamSimulator LANGUAGES CUDA CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CONFIGURATION_TYPES "Release;RelWithDebInfo;Debug" CACHE STRING "" FORCE)

set(CMAKE_CUDA_ARCHITECTURES 50)

#==================================== Inclusion of Qt ============================#
set(CMAKE_PREFIX_PATH "$ENV{QTDIR}")
set(VTK_DIR $ENV{VTK_DIR})

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL Gui)

#==================================== Inclusion of VTK ===========================#

find_package(VTK REQUIRED)

#=================================== Inclusion of own code =======================#
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include")

set(UI_HEADER 	"${INC_DIR}/ui_module/MainWindow.h" 
						"${INC_DIR}/ui_module/SimRenderer.h")

set(UI_SOURCE	"${SRC_DIR}/ui_module/MainWindow.cpp"
						"${SRC_DIR}/ui_module/SimRenderer.cpp")

set(UI_GUI_FILES 		"${SRC_DIR}/ui_module/MainWindow.ui")


set(LBM_HEADER	"${INC_DIR}/lbm_module/CudaErrorHandle.h" 
						"${INC_DIR}/lbm_module/LBMKernels.h"
						"${INC_DIR}/lbm_module/LBMSolver.h"
						"${INC_DIR}/lbm_module/SimDomain.h"
						"${INC_DIR}/lbm_module/SimState.h")

set(LBM_SOURCE	"${SRC_DIR}/lbm_module/LBMSolver.cpp"
						"${SRC_DIR}/lbm_module/SimDomain.cpp"
						"${SRC_DIR}/lbm_module/SimState.cpp")

set(LBM_KERNEL	"${SRC_DIR}/lbm_module/LBMKernels.cu")



add_library(stream_ui  STATIC ${UI_HEADER} ${UI_SOURCE} ${UI_GUI_FILES})
add_library(stream_lbm STATIC ${LBM_HEADER}  ${LBM_SOURCE} ${LBM_KERNEL})


target_include_directories(stream_ui 	PUBLIC "${INC_DIR}/ui_module")
target_include_directories(stream_lbm 	PUBLIC "${INC_DIR}/lbm_module")

target_link_libraries(stream_ui Qt6::Core Qt6::Widgets Qt6::OpenGL Qt6::Gui ${VTK_LIBRARIES})	


add_executable(StreamSimulator ${SRC_DIR}/main.cpp)
target_include_directories(StreamSimulator PRIVATE "${INC_DIR}")

target_link_libraries(StreamSimulator PRIVATE stream_lbm
									  PRIVATE stream_ui)

