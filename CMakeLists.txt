cmake_minimum_required(VERSION 3.16)
project(main)


# Try to find OpenCV
# set(OpenCV_DIR ....)
find_package(OpenCV 4 REQUIRED)


# include has 2 subdirectories: tensorflow and flatbuffers
# get the folder form https://github.com/muhammedakyuzlu/tensorflow_lite_libs_cpp.git
INCLUDE_DIRECTORIES("[path/tensorflow_lite_libs_cpp/include/]")

# lib has 1 file: libtensorflowlite.so
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION path/to/tensorflow_lite_libs_cpp/libs/libtensorflowlite.so)

file(GLOB SOURCE_FILES src/*.cpp)
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(
    ${CMAKE_PROJECT_NAME}
    ${OpenCV_LIBS}
    tensorflowlite
    pthread
)