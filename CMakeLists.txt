cmake_minimum_required(VERSION 3.26)
project(StreamSimulator VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_PREFIX_PATH "C:/Qt/6.4.2/mingw_64/lib/cmake")
find_package(Qt6 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()

enable_language(CUDA)

add_library(stream_ui STATIC 
							src/main_window.cpp
							include/main_window.hpp)
							
							
							
target_include_directories(stream_ui PUBLIC include)

add_executable(StreamSimulator src/main.cpp)
target_link_libraries(StreamSimulator PUBLIC stream_ui
									  PRIVATE Qt6::Widgets)
