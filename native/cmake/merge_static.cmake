# Merges the cvk shim and all OpenCV/3rdparty static archives into one
# archive for cinterop embedding.
#
# 'ar' implementations disagree on nested archives: GNU ar flattens members
# while LLVM/BSD ar store the whole input archive as one member. To be
# implementation-independent this script extracts every input into its own
# directory, copies the objects back under a "<index>_<member>" name that
# preserves the dependency order, and finally combines the flat objects.
#
# Inputs arrive as cache variables from native/CMakeLists.txt. Producing
# nothing is a hard error.

set(_libs "${CVK_SHIM}")

file(GLOB _modules "${CVK_OPENCV_LIB_DIR}/*.a")
list(SORT _modules)

# Move libopencv_core.a (if present) to the very end of the module list so
# earlier modules' references resolve in a single pass.
list(FIND _modules "${CVK_OPENCV_LIB_DIR}/libopencv_core.a" _core_index)
if(NOT _core_index EQUAL -1)
    list(REMOVE_AT _modules ${_core_index})
    list(APPEND _modules "${CVK_OPENCV_LIB_DIR}/libopencv_core.a")
endif()

list(APPEND _libs ${_modules})

if(EXISTS "${CVK_OPENCV_3RDPARTY_LIB_DIR}")
    file(GLOB _thirdparty "${CVK_OPENCV_3RDPARTY_LIB_DIR}/*.a")
    list(SORT _thirdparty)
    list(APPEND _libs ${_thirdparty})
endif()

list(REMOVE_DUPLICATES _libs)
list(LENGTH _libs _count)
if(_count LESS 2)
    message(FATAL_ERROR "merge_static: nothing to merge (only ${_libs})")
endif()

message(STATUS "Merging ${_count} archives into ${CVK_OUT}")

set(_work "${CVK_OUT}.merge")
file(REMOVE_RECURSE "${_work}")
file(MAKE_DIRECTORY "${_work}")
file(REMOVE "${CVK_OUT}")

set(_objects "")
set(_index 0)

foreach(_archive ${_libs})
    math(EXPR _index "${_index} + 1")
    get_filename_component(_archive_name "${_archive}" NAME_WE)
    set(_extract_dir "${_work}/x${_index}")
    file(MAKE_DIRECTORY "${_extract_dir}")
    execute_process(COMMAND "${CVK_AR}" x "${_archive}"
            WORKING_DIRECTORY "${_extract_dir}"
            COMMAND_ERROR_IS_FATAL ANY)
    file(GLOB _members LIST_DIRECTORIES false "${_extract_dir}/*")
    foreach(_member ${_members})
        get_filename_component(_member_name "${_member}" NAME)
        set(_flat_name "${_index}_${_archive_name}_${_member_name}")
        file(RENAME "${_member}" "${_work}/${_flat_name}")
        list(APPEND _objects "${_work}/${_flat_name}")
    endforeach()
endforeach()

# Sort lexicographically: the numeric index prefix keeps the dependency
# order inside every group, and same-index objects group by original name.
list(SORT _objects)

execute_process(COMMAND "${CVK_AR}" qc "${CVK_OUT}" ${_objects}
        WORKING_DIRECTORY "${_work}"
        COMMAND_ERROR_IS_FATAL ANY)

# Some ar frontends skip the symbol index for 'q' merges; make it explicit.
execute_process(COMMAND "${CVK_RANLIB}" "${CVK_OUT}"
        RESULT_VARIABLE _ranlib_result
        ERROR_QUIET)

file(REMOVE_RECURSE "${_work}")

if(NOT _ranlib_result EQUAL 0)
    message(WARNING "merge_static: ranlib exited with ${_ranlib_result}; continuing")
endif()
