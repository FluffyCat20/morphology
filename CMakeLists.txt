cmake_minimum_required(VERSION 2.8.12)

project(morphology1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(HEADERS "ccl_algorithm.h" "vec_methods.h" "frame_rectangle.h" "niblack_algorithm.h" "morph.h")
set(SOURCES "main.cpp" "niblack_algorithm.cpp" "ccl_algorithm.cpp" "vec_methods.cpp" "frame_rectangle.cpp" "morph.cpp")

find_package(Qt5Core)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} "main.cpp")

target_link_libraries(${PROJECT_NAME} Qt5::Core)
target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} )
