# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)


set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror")



include_directories(/home/seedship/ECE/420/OpenCV-android-sdk/sdk/native/jni/include)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library(echo SHARED
        audio_common.cpp
        audio_main.cpp
        audio_player.cpp
        audio_recorder.cpp
        debug_utils.cpp
        kiss_fft/kiss_fft.c
        ece420_lib.cpp
        ece420_main.cpp)


add_library(lib_opencv SHARED IMPORTED)

# include libraries needed for hello-jni lib
target_link_libraries(echo
                      android
                      atomic
                      log
                      OpenSLES
        ${log-lib}
        lib_opencv)

# Android Studio 2.2.0 with CMake support does not pack stl shared libraries,
# so app needs to pack the right shared lib into APK. This sample uses solution
# from https://github.com/jomof/ndk-stl to find the right stl shared lib to use
# and copy it to the right place for Android Studio to pack
# Usage: download ndk-stl-config.cmake into app's directory hosting CMakeLists.txt
#        and just use it with the following line
include(ndk-stl-config.cmake)


message(WARNING ${CMAKE_SOURCE_DIR})

set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION /home/seedship/ECE/420/MyApplication/app/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.
