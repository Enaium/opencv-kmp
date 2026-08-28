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

project(opencv_kmp_native LANGUAGES C CXX ASM)

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

# The OpenCV checkout lives next to this project as a submodule; the patches
# are applied against that working tree (see the block below).
get_filename_component(CVK_REPO_ROOT "${OPENCV_SOURCE_DIR}/.." ABSOLUTE)
set(CVK_OPENCV_PATCHES
    "${CMAKE_CURRENT_LIST_DIR}/cmake/mlas-subproject.patch"
    "${CMAKE_CURRENT_LIST_DIR}/cmake/imgcodecs-appletv.patch"
)

file(MAKE_DIRECTORY "${CVK_OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# OpenCV: static, minimal, offline. BUILD_LIST keeps every module the binding
# exposes; bundled zlib/libpng/libjpeg-turbo satisfy imgcodecs and bundled
# protobuf satisfies dnn, so no system headers or network downloads are
# involved. Everything else that could pull extra dependencies is switched
# off; ENABLE_PIC stays ON because the static objects end up inside shared
# libraries (JNI) and PIE executables (Kotlin/Native).
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# CVK_VERBOSE=1 in the environment prints every compile/link command
# (CMAKE_VERBOSE_MAKEFILE); CI sets it to make native failures
# reproducible from logs, local builds stay quiet by default.
if(DEFINED ENV{CVK_VERBOSE})
    set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "" FORCE)
endif()
# Always use the bundled codecs. zlib: recent NDK sysroots ship libz.so
# and CI runners ship libjpeg/libpng dev packages, so OpenCV's finders
# pick the system libraries and the 3rdparty archives the merge step
# copies (libzlib.a / liblibjpeg-turbo.a / libpng.a) are never produced.
# The bundled archives are what the static libs link, and forcing them
# keeps every host deterministic. (png/jpeg must stay enabled - WITH_*
# OFF disables the codec entirely - so instead the system results are
# masked and BUILD_* stays ON for the 3rdparty build.)
set(WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(BUILD_ZLIB ON CACHE BOOL "" FORCE)
# Mask the system results before OpenCV's find modules run.
set(JPEG_FOUND FALSE CACHE BOOL "" FORCE)
set(JPEG_LIBRARY "JPEG_LIBRARY-NOTFOUND" CACHE FILEPATH "" FORCE)
set(JPEG_INCLUDE_DIR "JPEG_INCLUDE_DIR-NOTFOUND" CACHE FILEPATH "" FORCE)
set(PNG_FOUND FALSE CACHE BOOL "" FORCE)
set(PNG_LIBRARY "PNG_LIBRARY-NOTFOUND" CACHE FILEPATH "" FORCE)
set(PNG_PNG_INCLUDE_DIR "PNG_PNG_INCLUDE_DIR-NOTFOUND" CACHE FILEPATH "" FORCE)
# geometry (contours, affine/perspective helpers), features (corner
# detectors) and calib are OpenCV 5's new homes for classic imgproc APIs the
# binding exposes. ptcloud requires highgui, so every target compiles it: a
# real window backend (Win32UI/Cocoa/GTK3) on desktop, the built-in "NONE"
# stub on Android and Apple mobile where the shim reports unavailability
# through no-op stubs. GTK3 headers/libraries must be installed on Linux
# build hosts (CI installs libgtk-3-dev).
set(BUILD_LIST
        "core,geometry,imgproc,imgcodecs,features,photo,video,videoio,objdetect,dnn,stereo,ptcloud,calib,highgui"
        CACHE STRING "" FORCE)

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

# Apple mobile platforms have no AVFoundation-based videoio in this binding
# (videoio keeps its API but with no backend there, same as Linux/Windows;
# macOS keeps AVFoundation so VideoCapture/VideoWriter work for real).
# CMake only defines IOS (not TVOS/WATCHOS), so branch on the system name
# directly for the Apple-mobile targets where videoio's AVFoundation backend
# is unavailable (its macOS implementation would otherwise compile).
if(CMAKE_SYSTEM_NAME MATCHES "iOS|tvOS|watchOS")
    set(WITH_AVFOUNDATION OFF CACHE BOOL "" FORCE)
endif()

# Desktop window backends for highgui. Win32UI ships with every MinGW/MSVC
# toolchain, Cocoa with Xcode, and GTK3 is the Linux backend (CI installs
# libgtk-3-dev). Android and Apple mobile keep the "NONE" stub backend.
if(WIN32 AND NOT ANDROID)
    set(WITH_WIN32UI ON CACHE BOOL "" FORCE)
endif()
if(APPLE AND NOT ANDROID AND NOT IOS AND NOT TVOS AND NOT WATCHOS)
    set(WITH_COCOA ON CACHE BOOL "" FORCE)
endif()
# Optional AddressSanitizer instrumentation for local debugging
# (CVK_SANITIZE=address in the environment): flags the opencv static
# libraries AND the shim, so use-as-after-free inside any call is caught
# by the test JVM's preloaded ASan runtime.
if(DEFINED ENV{CVK_SANITIZE} AND NOT CVK_SANITIZE_FLAGS)
    set(CVK_SANITIZE_FLAGS "-fsanitize=$ENV{CVK_SANITIZE} -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CVK_SANITIZE_FLAGS}" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CVK_SANITIZE_FLAGS}" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=$ENV{CVK_SANITIZE}" CACHE STRING "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=$ENV{CVK_SANITIZE}" CACHE STRING "" FORCE)
    set(CVK_SANITIZE_FLAGS "${CVK_SANITIZE_FLAGS}" CACHE INTERNAL "")
endif()

# Patches against the vendored OpenCV tree (the submodule is never forked):
#   mlas-subproject.patch  - 3rdparty/mlas path fixes for add_subdirectory
#     use (without it: "No SOURCES given to target: opencv_dnn_mlas") plus
#     Apple architecture detection from CMAKE_OSX_ARCHITECTURES (the host
#     probe reports arm64 even when targeting x86_64).
#   imgcodecs-appletv.patch - exclude macOS-only sources on tvOS/watchOS
#     (upstream guards only IOS/XROS; cmake defines no TVOS/WATCHOS flags).
# Applied idempotently before the subdir is configured; a dirty working tree
# that already carries them is left untouched.
foreach(CVK_PATCH IN LISTS CVK_OPENCV_PATCHES)
    execute_process(
        COMMAND git apply --reverse --check "${CVK_PATCH}"
        WORKING_DIRECTORY "${CVK_REPO_ROOT}"
        RESULT_VARIABLE CVK_PATCH_REV
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT CVK_PATCH_REV EQUAL 0)
        execute_process(
            COMMAND git apply "${CVK_PATCH}"
            WORKING_DIRECTORY "${CVK_REPO_ROOT}"
            RESULT_VARIABLE CVK_PATCH_RES
            OUTPUT_QUIET ERROR_QUIET
        )
        if(NOT CVK_PATCH_RES EQUAL 0)
            # Concurrent configure tasks (different targets in one Gradle
            # invocation) can race on the shared working tree: re-check
            # whether the other process applied the patch in between.
            execute_process(
                COMMAND git apply --reverse --check "${CVK_PATCH}"
                WORKING_DIRECTORY "${CVK_REPO_ROOT}"
                RESULT_VARIABLE CVK_PATCH_REV2
                OUTPUT_QUIET ERROR_QUIET
            )
            if(NOT CVK_PATCH_REV2 EQUAL 0)
                message(FATAL_ERROR
                    "Failed to apply ${CVK_PATCH} in ${CVK_REPO_ROOT}; "
                    "reset the opencv submodule to a pristine checkout")
            endif()
        endif()
        message(STATUS "cvk: applied ${CVK_PATCH}")
    else()
        message(STATUS "cvk: ${CVK_PATCH} already applied")
    endif()
endforeach()

# CMake defines IOS but not TVOS/WATCHOS for those CMAKE_SYSTEM_NAME values;
# OpenCV guards (imgcodecs excluding macOS-only sources, videoio backends)
# test the matching platform variable, so define them for the whole pass.
if(CMAKE_SYSTEM_NAME STREQUAL "tvOS")
    set(TVOS 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "watchOS")
    set(WATCHOS 1)
endif()

add_subdirectory("${OPENCV_SOURCE_DIR}" opencv-build EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# cvk shim
# ---------------------------------------------------------------------------
set(CVK_SKIP_STDCPP_SHIM OFF CACHE BOOL
        "Drop the libstdc++ compat shims (mingwX64 embeds the full GNU runtime)")

# Per-module shim translations (shim_<module>.cpp) are globbed so parallel
# module work can add files without touching this file.
file(GLOB CVK_SHIM_MODULE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/shim_*.cpp")
add_library(cvk_shim STATIC opencv_shim.cpp glibc_compat.c stdcpp_compat.cpp
        ${CVK_SHIM_MODULE_SOURCES})
if(CVK_SKIP_STDCPP_SHIM)
    target_compile_definitions(cvk_shim PRIVATE CVK_SKIP_STDCPP_SHIM)
endif()
# highgui is a no-op stub on Android and Apple mobile (the built-in "NONE"
# backend); the shim reports unavailability there instead of calling it.
if(ANDROID OR IOS OR TVOS OR WATCHOS)
    target_compile_definitions(cvk_shim PRIVATE CVK_NO_HIGHGUI)
endif()
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
        "${OPENCV_SOURCE_DIR}/modules/calib/include"
        "${OPENCV_SOURCE_DIR}/modules/photo/include"
        "${OPENCV_SOURCE_DIR}/modules/video/include"
        "${OPENCV_SOURCE_DIR}/modules/videoio/include"
        "${OPENCV_SOURCE_DIR}/modules/objdetect/include"
        "${OPENCV_SOURCE_DIR}/modules/stereo/include"
        "${OPENCV_SOURCE_DIR}/modules/ptcloud/include"
        "${OPENCV_SOURCE_DIR}/modules/dnn/include"
        "${OPENCV_SOURCE_DIR}/modules/highgui/include"
        # Configure-time generated headers (opencv_modules.hpp, cvconfig.h).
        "${CMAKE_BINARY_DIR}")
# The shim only ever sees plain C ABI calls from consumers, but it links
# against C++ OpenCV, hence C++ runtime linkage via the opencv targets.
target_link_libraries(cvk_shim PUBLIC
        opencv_core opencv_geometry opencv_imgproc opencv_features opencv_flann
        opencv_imgcodecs opencv_highgui opencv_calib opencv_photo opencv_video
        opencv_videoio opencv_objdetect opencv_stereo opencv_ptcloud opencv_dnn)

if(CVK_BUILD_JNI)
    if(CVK_JNI_BRIDGE STREQUAL "")
        message(FATAL_ERROR "CVK_BUILD_JNI=ON requires -DCVK_JNI_BRIDGE=<jni_bridge.cpp>")
    endif()
    # Per-module JNI bridges (jni/jni_<module>.cpp) are globbed in as well;
    # each implements Java_cn_enaium_opencv_Jni<Module>_* on top of the
    # cvk_ ABI.
    file(GLOB CVK_JNI_MODULE_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_LIST_DIR}/../jni/jni_*.cpp")
    list(FILTER CVK_JNI_MODULE_SOURCES EXCLUDE REGEX "jni_bridge\.cpp$")
    add_library(opencv_jni SHARED "${CVK_JNI_BRIDGE}" ${CVK_JNI_MODULE_SOURCES})
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

set(CVK_COPY_MODULES cvk_shim opencv_ptcloud opencv_calib opencv_objdetect
        opencv_dnn opencv_video opencv_stereo opencv_photo opencv_highgui
        opencv_videoio opencv_imgcodecs opencv_features opencv_imgproc
        opencv_flann opencv_geometry opencv_core)

foreach(_t ${CVK_COPY_MODULES})
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

# Bundled codecs zlib/libpng/libjpeg-turbo; the 3rdparty targets must
# finish before the copies run (parallel make can otherwise race the
# archive creation), so each copy depends on its own target.
foreach(_n libjpeg-turbo libpng zlib)
    if(NOT TARGET ${_n})
        continue()
    endif()
    set(_out "${CVK_OUTPUT_DIR}/lib${_n}.a")
    add_custom_command(OUTPUT "${_out}"
            COMMAND test -f "${CMAKE_BINARY_DIR}/opencv-build/3rdparty/lib${_abi_sub}/lib${_n}.a"
            COMMAND "${CMAKE_COMMAND}" -E copy
                    "${CMAKE_BINARY_DIR}/opencv-build/3rdparty/lib${_abi_sub}/lib${_n}.a" "${_out}"
            DEPENDS opencv_core ${_n}
            VERBATIM)
    list(APPEND CVK_COPIED_LIBS "${_out}")
endforeach()
# dnn's bundled protobuf (separate 3rdparty archive referenced by
# opencv_dnn; absent when dnn is not in the build).
if(TARGET libprotobuf)
    # build.gradle.kts embeds archives as -staticLibrary lib$it.a, so the
    # copy must carry the same lib prefix as the module archives.
    set(_out "${CVK_OUTPUT_DIR}/liblibprotobuf.a")
    add_custom_command(OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:libprotobuf>" "${_out}"
            DEPENDS libprotobuf
            VERBATIM)
    list(APPEND CVK_COPIED_LIBS "${_out}")
endif()

# MinGW klibs: Kotlin/Native's bundled libstdc++ for gnu_x86_64_windows
# predates the host cross-GCC, so the real static runtime travels with the
# klib (ios_base_library_init, __gthr_win32_* helpers, stream symbols).
if(WIN32 OR EXISTS "/usr/x86_64-w64-mingw32")
    set(CVK_STDCPP_MINGW_SRC "")
    if(EXISTS "/usr/x86_64-w64-mingw32/lib/libstdc++.a")
        set(CVK_STDCPP_MINGW_SRC "/usr/x86_64-w64-mingw32/lib/libstdc++.a")
    else()
        file(GLOB_RECURSE _mingw_stdcpp
                "/usr/x86_64-w64-mingw32/*.a"
                "/usr/lib/gcc/x86_64-w64-mingw32/*.a")
        list(FILTER _mingw_stdcpp INCLUDE REGEX "libstdc\\+\\+\\.a$")
        list(GET _mingw_stdcpp 0 CVK_STDCPP_MINGW_SRC)
    endif()
    message(STATUS "CVK mingw libstdc++: ${CVK_STDCPP_MINGW_SRC}")
    set(_out "${CVK_OUTPUT_DIR}/libstdc++_mingw_x64.a")
    add_custom_command(OUTPUT "${_out}"
            COMMAND "${CMAKE_COMMAND}" -E copy "${CVK_STDCPP_MINGW_SRC}" "${_out}"
            DEPENDS "${CVK_STDCPP_MINGW_SRC}"
            VERBATIM)
    list(APPEND CVK_COPIED_LIBS "${_out}")

    # OpenCV's system.cpp references __gthr_win32_*, which only the
    # win32-thread-model libgcc (gcc-mingw-w64-x86-64-win32) defines; the
    # posix variant's libgcc does not, so prefer the *-win32 archive.
    file(GLOB_RECURSE _mingw_libgcc_win32 "/usr/lib/gcc/x86_64-w64-mingw32/*win32*/libgcc.a")
    file(GLOB_RECURSE _mingw_libgcc_any "/usr/lib/gcc/x86_64-w64-mingw32/*/libgcc.a")
    if(_mingw_libgcc_win32)
        list(GET _mingw_libgcc_win32 0 CVK_LIBGCC_MINGW_SRC)
    elseif(_mingw_libgcc_any)
        list(GET _mingw_libgcc_any 0 CVK_LIBGCC_MINGW_SRC)
    endif()
    if(CVK_LIBGCC_MINGW_SRC)
        message(STATUS "CVK mingw libgcc: ${CVK_LIBGCC_MINGW_SRC}")
        set(_out "${CVK_OUTPUT_DIR}/libgcc_mingw_x64.a")
        add_custom_command(OUTPUT "${_out}"
                COMMAND "${CMAKE_COMMAND}" -E copy "${CVK_LIBGCC_MINGW_SRC}" "${_out}"
                DEPENDS "${CVK_LIBGCC_MINGW_SRC}"
                VERBATIM)
        list(APPEND CVK_COPIED_LIBS "${_out}")
    endif()
endif()


add_custom_target(opencv_kmp ALL DEPENDS ${CVK_COPIED_LIBS})
