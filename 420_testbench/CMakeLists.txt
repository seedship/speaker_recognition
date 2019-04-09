#This is an example CMakeFile
cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)

project(420_testbench)


find_package( OpenCV 3.4.3 REQUIRED ) #fixing on OpenCV 3.4.3

include_directories(${CMAKE_SOURCE_DIR})
include_directories( ${OpenCV_INCLUDE_DIRS})
message(WARNING ${OpenCV_INCLUDE_DIRS})

#specify your source files here, add more if needed
add_executable(
	${PROJECT_NAME} "main.cpp"
	"ece420_lib.cpp"
	"kiss_fft.c")


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
