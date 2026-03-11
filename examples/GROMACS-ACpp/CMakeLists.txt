#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2026 pocl developers
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

set(TS_NAME "GROMACS-ACpp")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SAMPLEDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}/samples")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

if(NOT Python3_EXECUTABLE)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find suitable python3")
  return()
else()
  set(PYTHON_INTERP "${Python3_EXECUTABLE}")
endif()

if(NOT HOST_LLVM_SPIRV)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires llvm-spirv")
  return()
endif()

find_program(ACPP_EXECUTABLE acpp)

if (NOT ACPP_EXECUTABLE)
  message(STATUS "Disabling testsuite ${TS_NAME}, AdaptiveCpp not found")
  return()
else()
  message(STATUS "Found AdaptiveCpp")
endif()


message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://gitlab.com/tneval/gromacs-tn.git"
  GIT_TAG "PoCL-ACpp"
  ${GIT_OPTIONS}
  CMAKE_ARGS
    -DCMAKE_C_COMPILER=${HOST_CLANG}
    -DCMAKE_CXX_COMPILER=acpp
    -DREGRESSIONTEST_DOWNLOAD=OFF
    -DGMX_GPU=SYCL
    -DGMX_SYCL=ACPP
    -DACPP_TARGETS="generic"
    -DGMX_BUILD_OWN_FFTW=ON
    -DGMX_GPU_FFT_LIBRARY=none
    -DGMX_GPU_NB_CLUSTER_SIZE=8
    -DGMX_OPENMP=OFF
  INSTALL_COMMAND ""
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})


set(TEST_LAUNCHER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/GROMACS_runner.sh")

add_test(NAME GROMACS-SYCL-OFFLOAD-NONBONDED-FORCES COMMAND ${TEST_LAUNCHER_SCRIPT} "${TS_SAMPLEDIR}" "${TS_BUILDDIR}" "${CMAKE_CURRENT_SOURCE_DIR}" "grappa-0024-rf")

set_tests_properties(
  GROMACS-SYCL-OFFLOAD-NONBONDED-FORCES
  PROPERTIES
  # RUN_SERIAL to avoid running anything else at the same time
  RUN_SERIAL ON
  LABELS "GROMACS-ACpp")