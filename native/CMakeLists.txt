cmake_minimum_required(VERSION 3.16)

# Unified build for opencv-kmp's native layer.
#
#   CVK_BUILD_JNI=OFF (default, used by the Kotlin/Native targets)
#     Compiles OpenCV statically plus the cvk_ C shim and merges everything
#     into ${CVK_OUTPUT_DIR}/libopencv_kmp.a for cinterop klib embedding.
#
#   CVK_BUILD_JNI=ON (used by the :jni-jvm-* modules)
#     Additionally compiles jni_bridge.cpp into a shared libopencv_jni
#     (${CVK_OUTPUT_DIR}/libopencv_jni.<ext>), statically linking OpenCV and
#     the shim. The merged archive is skipped in this mode.
#
# Cross builds pass the regular CMake toolchain variables
# (-DCMAKE_SYSTEM_NAME / -DCMAKE_C(XX)_COMPILER / NDK toolchain file /
# -DCMAKE_OSX_ARCHITECTURES); nothing here is host-specific.

project(opencv_kmp_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CVK_BUILD_JNI OFF CACHE BOOL "Also build the JNI shared library")
set(CVK_OUTPUT_DIR "${CMAKE_BINARY_DIR}/out" CACHE PATH "Where final artifacts are placed")
set(CVK_JNI_BRIDGE "" CACHE FILEPATH "jni_bridge.cpp source (JNI mode)")
set(CVK_JNI_INCLUDE "" CACHE PATH "JNI include dir (jdk include root or NDK sysroot)")
set(CVK_JNI_INCLUDE_PLATFORM "" CACHE PATH "JNI platform include dir")

if(NOT DEFINED OPENCV_SOURCE_DIR OR OPENCV_SOURCE_DIR STREQUAL "")
    get_filename_component(OPENCV_SOURCE_DIR
            "${CMAKE_CURRENT_LIST_DIR}/../opencv" ABSOLUTE)
endif()
if(NOT EXISTS "${OPENCV_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "OpenCV sources not found at ${OPENCV_SOURCE_DIR} \
        (pass -DOPENCV_SOURCE_DIR=<path>)")
endif()

file(MAKE_DIRECTORY "${CVK_OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# OpenCV: static, minimal, offline. BUILD_LIST keeps core/imgproc/imgcodecs;
# bundled zlib/libpng/libjpeg-turbo satisfy imgcodecs so no system headers or
# network downloads are involved. Everything else that could pull extra
# dependencies is switched off; ENABLE_PIC stays ON because the static objects
# end up inside shared libraries (JNI) and PIE executables (Kotlin/Native).
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# geometry (contours, affine/perspective helpers) and features (corner
# detectors) are OpenCV 5's new homes for classic imgproc APIs the binding
# exposes; imgproc already pulls them in, features is added explicitly.
set(BUILD_LIST "core,geometry,imgproc,imgcodecs,features" CACHE STRING "" FORCE)

foreach(_opt
        BUILD_TESTS
        BUILD_PERF_TESTS
        BUILD_EXAMPLES
        BUILD_opencv_apps
        BUILD_JAVA
        BUILD_opencv_java
        BUILD_opencv_js
        BUILD_ANDROID_PROJECTS
        BUILD_ANDROID_EXAMPLES
        BUILD_opencv_python2
        BUILD_opencv_python3
        WITH_IPP
        WITH_ITT
        WITH_TBB
        WITH_OPENCL
        WITH_OPENCLAMDFFT
        WITH_OPENCLAMDBLAS
        WITH_VA
        WITH_VA_INTEL
        WITH_1394
        WITH_V4L
        WITH_GTK
        WITH_GTK_2_X
        WITH_QT
        WITH_WIN32UI
        WITH_FFMPEG
        WITH_GSTREAMER
        WITH_MSMF
        WITH_OPENEXR
        WITH_TIFF
        WITH_WEBP
        WITH_JASPER
        WITH_OPENJPEG
        WITH_LAPACK
        WITH_EIGEN
        WITH_CUDA
        WITH_CUFFT
        WITH_CUBLAS
        WITH_MATLAB
        WITH_VTK
        WITH_GDCM
        WITH_ACCELERATE
        # System codec libraries (libavif, libjxl, ...) would break the
        # "self-contained artifact" guarantee; keep only the bundled ones.
        WITH_AVIF
        WITH_JXL
        WITH_GDAL
        WITH_GDCM
        # OpenCV 5 shapes text through HarfBuzz; the binding never renders
        # text and a system HarfBuzz would leak into every artifact.
        WITH_HARFBUZZ)
    set(${_opt} OFF CACHE BOOL "" FORCE)
endforeach()

set(BUILD_ZLIB ON CACHE BOOL "" FORCE)
set(BUILD_PNG ON CACHE BOOL "" FORCE)
set(BUILD_JPEG ON CACHE BOOL "" FORCE)
set(WITH_PNG ON CACHE BOOL "" FORCE)
set(WITH_JPEG ON CACHE BOOL "" FORCE)

add_subdirectory("${OPENCV_SOURCE_DIR}" opencv-build EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# cvk shim
# ---------------------------------------------------------------------------
add_library(cvk_shim STATIC opencv_shim.cpp glibc_compat.c stdcpp_compat.cpp)
target_include_directories(cvk_shim PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/include"
        # Public umbrella headers and per-module headers straight from the
        # source tree (no install step involved).
        "${OPENCV_SOURCE_DIR}/include"
        "${OPENCV_SOURCE_DIR}/modules/core/include"
        "${OPENCV_SOURCE_DIR}/modules/geometry/include"
        "${OPENCV_SOURCE_DIR}/modules/imgproc/include"
        "${OPENCV_SOURCE_DIR}/modules/flann/include"
        "${OPENCV_SOURCE_DIR}/modules/imgcodecs/include"
        "${OPENCV_SOURCE_DIR}/modules/features/include"
        # Configure-time generated headers (opencv_modules.hpp, cvconfig.h).
        "${CMAKE_BINARY_DIR}")
# The shim only ever sees plain C ABI calls from consumers, but it links
# against C++ OpenCV, hence C++ runtime linkage via the opencv targets.
target_link_libraries(cvk_shim PUBLIC opencv_core opencv_geometry opencv_imgproc
        opencv_features opencv_imgcodecs)

if(CVK_BUILD_JNI)
    if(CVK_JNI_BRIDGE STREQUAL "")
        message(FATAL_ERROR "CVK_BUILD_JNI=ON requires -DCVK_JNI_BRIDGE=<jni_bridge.cpp>")
    endif()
    add_library(opencv_jni SHARED "${CVK_JNI_BRIDGE}")
    target_include_directories(opencv_jni PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/include"
            "${CVK_JNI_INCLUDE}"
            "${CVK_JNI_INCLUDE_PLATFORM}")
    # cvk_shim propagates the OpenCV static targets (and their LINK_ONLY
    # system-library dependencies) into this shared library.
    target_link_libraries(opencv_jni PRIVATE cvk_shim)
    set_target_properties(opencv_jni PROPERTIES
            OUTPUT_NAME "opencv_jni"
            LIBRARY_OUTPUT_DIRECTORY "${CVK_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${CVK_OUTPUT_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY "${CVK_OUTPUT_DIR}")
    if(WIN32)
        # Windows DLLs carry no lib prefix (opencv_jni.dll).
        set_target_properties(opencv_jni PROPERTIES PREFIX "")
    endif()
    return()
endif()

# ---------------------------------------------------------------------------
# Artifacts for cinterop embedding: instead of merging archives (libjpeg-turbo
# ships duplicate member names for its 8/12/16-bit variants and every ar
# flavor handles nested/duplicate members differently), copy each static
# library next to the shim archive; build.gradle.kts embeds them ALL as
# separate -staticLibrary entries, ordered dependents-first so a single
# linker pass resolves everything.
# ---------------------------------------------------------------------------
# Multi-ABI Android builds place outputs in a per-ABI subdirectory.
if(ANDROID)
    set(_abi_sub "/${ANDROID_ABI}")
else()
    set(_abi_sub "")
endif()

set(CVK_COPIED_LIBS "")

foreach(_t cvk_shim opencv_imgcodecs opencv_features opencv_imgproc opencv_flann
        opencv_geometry opencv_core)
    set(_out "${CVK_OUTPUT_DIR}/lib${_t}.a")
    add_custom_command(OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:${_t}>" "${_out}"
            DEPENDS ${_t}
            VERBATIM)
    list(APPEND CVK_COPIED_LIBS "${_out}")
endforeach()

# ARM acceleration libraries (KleidiCV HAL + carotene/tegra). They exist only
# on ARM configurations, so each copy is guarded by target existence; the
# Gradle embedding lists them only for ARM targets.
foreach(_t kleidicv kleidicv_hal kleidicv_thread tegra_hal)
    if(TARGET ${_t})
        set(_out "${CVK_OUTPUT_DIR}/lib${_t}.a")
        add_custom_command(OUTPUT "${_out}"
                COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:${_t}>" "${_out}"
                DEPENDS ${_t}
                VERBATIM)
        list(APPEND CVK_COPIED_LIBS "${_out}")
    endif()
endforeach()

# Bundled codecs zlib/libpng/libjpeg-turbo; building opencv_core guarantees
# the whole 3rdparty set exists.
foreach(_n libjpeg-turbo libpng zlib)
    set(_out "${CVK_OUTPUT_DIR}/lib${_n}.a")
    add_custom_command(OUTPUT "${_out}"
            COMMAND test -f "${CMAKE_BINARY_DIR}/opencv-build/3rdparty/lib${_abi_sub}/lib${_n}.a"
            COMMAND "${CMAKE_COMMAND}" -E copy
                    "${CMAKE_BINARY_DIR}/opencv-build/3rdparty/lib${_abi_sub}/lib${_n}.a" "${_out}"
            DEPENDS opencv_core
            VERBATIM)
    list(APPEND CVK_COPIED_LIBS "${_out}")
endforeach()

add_custom_target(opencv_kmp ALL DEPENDS ${CVK_COPIED_LIBS})
