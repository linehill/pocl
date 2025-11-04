#=============================================================================
#   CMake build system files - add_test_pocl() etc. test wrappers
#
#   Copyright (c) 2014-2017 pocl developers
#                 2024-2025 Pekka Jääskeläinen / Intel Finland Oy
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

include(CMakeParseArguments)

# This is a wrapper around add_test
# Solves several problems:
# 1) allows expected outputs (optionally sorted)
# 2) handles the exit status problem (test properties WILL_FAIL does not work if
#    the test exits with !0 exit status)
#
# If LLVM_FILECHECKS list, containing FileCheck files, is set,
# additional tests will be added that runs the test with the LLVM IR
# tester script using the loopvec method. Only one successfull file-check
# on any FileCheck file is needed to pass the IR check.
#
# If ONLY_FILECHECKS is set to 1, the test is only added as an LLVM IR
# filecheck which runs the program and validates the parallel.bc
# IR. Otherwise, if LLVM_FILECHECKS is given, the execution test is
# added also separately.
#
# LABELS can be used to add labels as a semicolon separated list.
# By default no labels are added and the test is expected to pass with host CPUs.
# Use tags such as cpu_fail, mingw_fail, win_fail to mark tests that are expected
# to fail on targets/platforms where they are expected to pass by default.
#
# WORKITEM_HANDLER can be set to a list of WG handlers to test with. Otherwise,
# "loopvec" and "cbs" are tested.
#
# EXPECTED_OUTPUT: Path to a file. If the path is not absolute, the
# file will be searched relatively to the CMAKE_CURRENT_SOURCE_DIR.
# When set, the test will check that the output of the test command matches
# to the contents of the given file.
#
# UNORDERED_OUTPUT_DIFF: Meaningful if EXPECTED_OUTPUT is set. This option
# changes the output checking so that the check passes if the output lines
# of the test command appear in the EXPECTED_OUTPUT file (including the
# duplicate lines).
function(add_test_pocl)

  set(options SORT_OUTPUT UNORDERED_OUTPUT_DIFF)
  set(oneValueArgs EXPECTED_OUTPUT NAME WORKING_DIRECTORY
    ONLY_FILECHECKS ENVIRONMENT ONLY_FILECHECK LLVM_FILECHECK)
  set(multiValueArgs COMMAND WORKITEM_HANDLER LABELS LLVM_FILECHECKS)
  cmake_parse_arguments(POCL_TEST "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  if(POCL_TEST_WORKITEM_HANDLER)
    set(VARIANTS ${POCL_TEST_WORKITEM_HANDLER})
  else()
    set(VARIANTS "loopvec" "fiber")
  endif()

  list(LENGTH VARIANTS VARIANTS_COUNT)

  # For an (unspeficied) transition period, catch mistakes of using the old
  # parameters.
  if(POCL_TEST_LLVM_FILECHECK)
   message(FATAL_ERROR "LLVM_FILECHECK has been renamed to LLVM_FILECHECKS")
  endif()
  if(POCL_TEST_ONLY_FILECHECK)
   message(FATAL_ERROR "ONLY_FILECHECK has been renamed to ONLY_FILECHECKS")
  endif()

  foreach(VARIANT ${VARIANTS})
    if(${VARIANTS_COUNT} GREATER 1)
      set(POCL_VARIANT_TEST_NAME ${POCL_TEST_NAME}_${VARIANT})
    else()
      set(POCL_VARIANT_TEST_NAME ${POCL_TEST_NAME})
    endif()
    unset(RUN_CMD)

    set(POCL_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(POCLBIN_DIR "${CMAKE_BINARY_DIR}/bin")
    get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(is_multi_config)
      set(POCL_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
      set(POCLBIN_DIR "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
    endif()

    foreach(LOOPVAR ${POCL_TEST_COMMAND})
      if(NOT RUN_CMD)
        # Special command name expansion.
        if(${LOOPVAR} STREQUAL "poclcc")
          set(RUN_CMD "${POCLBIN_DIR}/poclcc")
        else()
          set(RUN_CMD "${POCL_TEST_DIR}/${LOOPVAR}")
        endif()
      else()
        set(RUN_CMD "${RUN_CMD}####${LOOPVAR}")
      endif()
    endforeach()

    set(POCL_TEST_ARGLIST "NAME" "${POCL_VARIANT_TEST_NAME}")
    if(POCL_TEST_WORKING_DIRECTORY)
      list(APPEND POCL_TEST_ARGLIST "WORKING_DIRECTORY")
      list(APPEND POCL_TEST_ARGLIST "${POCL_TEST_WORKING_DIRECTORY}")
    endif()

    list(APPEND POCL_TEST_ARGLIST "COMMAND" "${CMAKE_COMMAND}" "-Dtest_cmd=${RUN_CMD}")
    if(INTEL_SDE_AVX512)
      list(APPEND POCL_TEST_ARGLIST "-DSDE=${INTEL_SDE_AVX512}")
    endif()

    if(POCL_TEST_EXPECTED_OUTPUT)
      if (NOT IS_ABSOLUTE "${POCL_TEST_EXPECTED_OUTPUT}")
        set(POCL_TEST_EXPECTED_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/${POCL_TEST_EXPECTED_OUTPUT}")
      endif()
      list(APPEND POCL_TEST_ARGLIST
        "-Doutput_blessed=${POCL_TEST_EXPECTED_OUTPUT}")
    endif()
    if(POCL_TEST_SORT_OUTPUT)
      list(APPEND POCL_TEST_ARGLIST "-Dsort_output=1")
    endif()
    if(POCL_TEST_UNORDERED_OUTPUT_DIFF)
      list(APPEND POCL_TEST_ARGLIST "-Dunordered_diff=1")
    endif()

    list(APPEND POCL_TEST_ARGLIST "-P" "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake")

    if(NOT POCL_TEST_ONLY_FILECHECKS)
      add_test(${POCL_TEST_ARGLIST})

      if(NOT ENABLE_ANYSAN)
        set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
          PASS_REGULAR_EXPRESSION "OK"
          FAIL_REGULAR_EXPRESSION "FAIL")
      endif()
      set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
        SKIP_RETURN_CODE 77)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16)
        set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
          SKIP_REGULAR_EXPRESSION "SKIP")
      endif()

      set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
        ENVIRONMENT "POCL_WORK_GROUP_METHOD=${VARIANT};${POCL_TEST_ENVIRONMENT}")

      set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
        LABELS "${POCL_TEST_LABELS}")
    endif()

    if(ENABLE_LLVM_FILECHECKS AND POCL_TEST_LLVM_FILECHECKS)
      set(FC_RUN_CMD "${CMAKE_SOURCE_DIR}/tools/scripts/run-and-check-llvm-ir####${TARGET_LLVM_FILECHECK}####${TARGET_LLVM_DIS}####${CMAKE_CURRENT_SOURCE_DIR}/${POCL_TEST_LLVM_FILECHECK}####${RUN_CMD}")
      foreach(FC IN LISTS POCL_TEST_LLVM_FILECHECKS)
        set(FC_RUN_CMD "${FC_RUN_CMD}####${CMAKE_CURRENT_SOURCE_DIR}/${FC}")
      endforeach()
      set(RUN_CMD "${FC_RUN_CMD}####--####${RUN_CMD}")

      set(POCL_TEST_IR_CHECK_NAME "${POCL_VARIANT_TEST_NAME}_llvm-ir-checks")
      set(POCL_TEST_ARGLIST "NAME" ${POCL_TEST_IR_CHECK_NAME})
      if(POCL_TEST_WORKING_DIRECTORY)
        list(APPEND POCL_TEST_ARGLIST "WORKING_DIRECTORY")
        list(APPEND POCL_TEST_ARGLIST "${POCL_TEST_WORKING_DIRECTORY}")
      endif()
      list(APPEND POCL_TEST_ARGLIST "COMMAND" "${CMAKE_COMMAND}" "-Dtest_cmd=${RUN_CMD}")
      list(APPEND POCL_TEST_ARGLIST "-DSKIP_RETURN_CODE=77")
      list(APPEND POCL_TEST_ARGLIST "-P" "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake")

      add_test(${POCL_TEST_ARGLIST})

      set_tests_properties(${POCL_TEST_IR_CHECK_NAME} PROPERTIES
                          PASS_REGULAR_EXPRESSION "OK"
                          FAIL_REGULAR_EXPRESSION "FAIL"
                          SKIP_RETURN_CODE 77
                          ENVIRONMENT "POCL_WORK_GROUP_METHOD=${VARIANT};${POCL_TEST_ENVIRONMENT}"
                          LABELS "${POCL_TEST_LABELS}"
                          DEPENDS "pocl_version_check")
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16)
        set_tests_properties("${POCL_TEST_IR_CHECK_NAME}" PROPERTIES
          SKIP_REGULAR_EXPRESSION "SKIP")
      endif()

    endif()

  endforeach()

endfunction()
