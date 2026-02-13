#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2024 pocl developers
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


add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
include_directories("${CMAKE_SOURCE_DIR}/lib/CL")
add_compile_options(${OPENCL_CFLAGS})

add_executable("run_kernel" "run_kernel.c")
target_link_libraries("run_kernel" ${POCLU_LINK_OPTIONS})

if((LLVM_VERSION VERSION_GREATER_EQUAL 21.0) AND
    (${TESTED_CPU_UARCH} MATCHES "(broadwell|znver3)"))
  # Check the kernel emits a vector load instead of a (masked) gather.
  list(APPEND SG_SHUFFLE_SMALL_FC_LIST sg_shuffle_small.fc)
else()
  # So far, the vector load is known to be emitted for few(?)
  # architectures by PoCL's pass pipeline.  However, dumping the
  # kernel bitcode right after the workgroup pass and passing it to
  # `clang -O2` emits code with vector load instead of gather
  # instruction (on an Alderlake machine at least). Maybe PoCL's
  # pipeline needs tweaking?
  list(APPEND SG_SHUFFLE_SMALL_FC_LIST
    sg_shuffle_small_gather.fc sg_shuffle_small.fc sg_shuffle_small_vstore.fc)
endif()

# This is expected to generate shufflevector instructions.
# only enable these tests if cl_khr_subgroup is enabled
if(HOST_DEVICE_EXTENSIONS MATCHES "cl_khr_subgroup_shuffle")
  add_test_pocl(NAME "subgroup/shuffle_small"
                EXPECTED_OUTPUT "sg_shuffle_small_1_8_1_1.stdout"
                COMMAND "run_kernel" "sg_shuffle.cl" 1 8 1 1
                LLVM_FILECHECKS "${SG_SHUFFLE_SMALL_FC_LIST}"
                WORKITEM_HANDLER "loopvec")

  add_test_pocl(NAME "subgroup/shuffle_inside_kloop"
                EXPECTED_OUTPUT "sg_shuffle_kloop_1_16_1_1.stdout"
                COMMAND "run_kernel" "sg_shuffle_kloop.cl" 1 16 1 1
                LLVM_FILECHECKS "sg_shuffle_kloop.fc"
                WORKITEM_HANDLER "loopvec")
endif()

add_test_pocl(NAME "workgroup/different_implicit_barrier_injection_scenarios"
              EXPECTED_OUTPUT "implicit_barriers_1_2_1_1.stdout"
              COMMAND "run_kernel" "implicit_barriers.cl" 1 2 1 1
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "workgroup/unbarriered_for_loops"
              EXPECTED_OUTPUT "forloops_2_2_1_1.stdout"
              COMMAND "run_kernel" "forloops.cl" 2 2 1 1)

add_test_pocl(NAME "workgroup/barriered_for_loops"
              EXPECTED_OUTPUT "loopbarriers_1_4_1_1.stdout"
              COMMAND "run_kernel" "loopbarriers.cl" 1 4 1 1)

add_test_pocl(NAME "workgroup/switch_case"
              EXPECTED_OUTPUT "switch_case_1_4_1_1.stdout"
              COMMAND "run_kernel" "switch_case.cl" 1 4 1 1)

# This is a case CBS was motivated for. Loopvec falls back to CBS currently,
# later on we add the simpler fiber-based fallback.
# TODO: Re-enable after CBS is brought back.
#add_test_pocl(NAME "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_cbs"
#              EXPECTED_OUTPUT "tricky_for_1_2_1_1.stdout"
#              COMMAND "run_kernel" "tricky_for.cl" 1 2 1 1
#              WORKITEM_HANDLER "cbs")

add_test_pocl(NAME "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_fiber"
              EXPECTED_OUTPUT "tricky_for_1_2_1_1.stdout"
              COMMAND "run_kernel" "tricky_for.cl" 1 2 1 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "workgroup/for_with_divergent_return"
              EXPECTED_OUTPUT "for_with_divergent_return_1_6_1_1.stdout"
              COMMAND "run_kernel" "for_with_divergent_return.cl" 1 6 1 1)

add_test_pocl(NAME "workgroup/cond_barriers_in_for"
              EXPECTED_OUTPUT "cond_barriers_in_for_2_4_1_1.stdout"
              COMMAND "run_kernel" "cond_barriers_in_for.cl" 2 4 1 1)

add_test_pocl(NAME "workgroup/outerlooppar_multiple_exits"
              EXPECTED_OUTPUT "outerlooppar_multiple_exits_1_4_1_1.stdout"
              COMMAND "run_kernel" "outerlooppar_multiple_exits.cl" 1 4 1 1
              LLVM_FILECHECKS "outerlooppar_multiple_exits.fc"
              WORKITEM_HANDLER "loopvec")

#add_test_pocl(NAME "workgroup/cond_barrier_in_var_for_cbs"
#              EXPECTED_OUTPUT "cond_barrier_in_var_for_2_4_1_1.stdout"
#              COMMAND "run_kernel" "cond_barrier_in_var_for.cl" 2 4 1 1
#              WORKITEM_HANDLER "cbs")

# Test cases for subgroup barriers (Not supported in conformance build).
add_test_pocl(NAME "subgroup/basic_subgroup_barrier_fiber"
              EXPECTED_OUTPUT "basic_sg_barrier_1_4_1_1_fiber.stdout"
              COMMAND "run_kernel" "basic_sg_barrier.cl" 1 4 1 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/basic_subgroup_barrier_loopvec"
              EXPECTED_OUTPUT "basic_sg_barrier_1_4_1_1_loopvec.stdout"
              COMMAND "run_kernel" "basic_sg_barrier.cl" 1 4 1 1
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "subgroup/basic_diverging_sg_barrier_fiber"
              EXPECTED_OUTPUT "basic_diverging_sg_barrier_1_4_1_1.stdout"
              COMMAND "run_kernel" "basic_diverging_sg_barrier.cl" 1 4 1 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/basic_diverging_sg_barrier_loopvec"
              EXPECTED_OUTPUT "basic_diverging_sg_barrier_1_4_1_1.stdout"
              COMMAND "run_kernel" "basic_diverging_sg_barrier.cl" 1 4 1 1
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "subgroup/diverging_sg_barriers_fiber"
              EXPECTED_OUTPUT "diverging_sg_barriers_1_128_1_1.stdout"
              COMMAND "run_kernel" "diverging_sg_barriers.cl" 1 128 1 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/mixing_barriers_fiber"
              EXPECTED_OUTPUT "mixing_barriers_1_8_1_1.stdout"
              COMMAND "run_kernel" "mixing_barriers.cl" 1 8 1 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/sg_barrier_in_for_in_if_fiber"
              EXPECTED_OUTPUT "sg_barrier_in_for_in_if_1_4_1_1_fiber.stdout"
              COMMAND "run_kernel" "sg_barrier_in_for_in_if.cl" 1 4 1 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/sg_barriers_with_sg_size_spanning_multiple_x_lanes_fiber"
              EXPECTED_OUTPUT "sg_barriers_with_sg_size_spanning_multiple_x_lanes_1_8_8_1_fiber.stdout"
              COMMAND "run_kernel" "sg_barriers_with_tricky_sg_size_1.cl" 1 8 8 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/sg_barriers_with_sg_size_spanning_multiple_x_lanes_loopvec"
              EXPECTED_OUTPUT "sg_barriers_with_sg_size_spanning_multiple_x_lanes_1_8_8_1_loopvec.stdout"
              COMMAND "run_kernel" "sg_barriers_with_tricky_sg_size_1.cl" 1 8 8 1
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "subgroup/sg_barriers_with_sg_size_spanning_multiple_z_lanes_fiber"
              EXPECTED_OUTPUT "sg_barriers_with_sg_size_spanning_multiple_z_lanes_1_4_1_32_fiber.stdout"
              COMMAND "run_kernel" "sg_barriers_with_tricky_sg_size_2.cl" 1 4 1 32
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/sg_barriers_with_sg_size_spanning_multiple_z_lanes_loopvec"
              EXPECTED_OUTPUT "sg_barriers_with_sg_size_spanning_multiple_z_lanes_1_4_1_32_loopvec.stdout"
              COMMAND "run_kernel" "sg_barriers_with_tricky_sg_size_2.cl" 1 4 1 32
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "subgroup/ids_in_sg_regions_fiber"
              EXPECTED_OUTPUT "ids_in_sg_regions_2_4_4_4_fiber.stdout"
              COMMAND "run_kernel" "ids_in_sg_regions.cl" 2 4 4 4
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "subgroup/ids_in_sg_regions_loopvec"
              EXPECTED_OUTPUT "ids_in_sg_regions_2_4_4_4_fiber.stdout"
              COMMAND "run_kernel" "ids_in_sg_regions.cl" 2 4 4 4
              WORKITEM_HANDLER "loopvec")


set(TEST_LIST
  "subgroup/basic_subgroup_barrier_fiber"
  "subgroup/basic_subgroup_barrier_loopvec"
  "subgroup/basic_diverging_sg_barrier_fiber"
  "subgroup/basic_diverging_sg_barrier_loopvec"
  "subgroup/diverging_sg_barriers_fiber"
  "subgroup/mixing_barriers_fiber"
  "subgroup/sg_barrier_in_for_in_if_fiber"
  "subgroup/sg_barriers_with_sg_size_spanning_multiple_x_lanes_fiber"
  "subgroup/sg_barriers_with_sg_size_spanning_multiple_x_lanes_loopvec"
  "subgroup/sg_barriers_with_sg_size_spanning_multiple_z_lanes_fiber"
  "subgroup/sg_barriers_with_sg_size_spanning_multiple_z_lanes_loopvec"
  "subgroup/ids_in_sg_regions_fiber"
  "subgroup/ids_in_sg_regions_loopvec"
  )
set_property(TEST ${TEST_LIST}
  APPEND PROPERTY ENVIRONMENT "POCL_MAX_COMPUTE_UNITS=1")
# Disable subgroup tests in conformance build.
if(ENABLE_CONFORMANCE)
  set_tests_properties(${TEST_LIST} PROPERTIES LABELS "cpu_fail")
endif()

add_test_pocl(NAME "workgroup/unconditional_barriers"
              EXPECTED_OUTPUT "basic_barriers_2_2_2_2.stdout"
              COMMAND "run_kernel" "basic_barriers.cl" 2 2 2 2)

add_test_pocl(NAME "workgroup/conditional_barrier_loopvec"
              EXPECTED_OUTPUT "cond_barriers_1_2_1_1_loopvec.stdout"
              COMMAND "run_kernel" "conditional_barriers.cl" 1 2 1 1
              WORKITEM_HANDLER "loopvec")

# TODO: Re-enable after CBS is brought back.
#add_test_pocl(NAME "workgroup/conditional_barrier_cbs"
#              EXPECTED_OUTPUT "cond_barriers_1_2_1_1_cbs.stdout"
#              COMMAND "run_kernel" "conditional_barriers.cl" 1 2 1 1
#              WORKITEM_HANDLER "cbs")


add_test_pocl(NAME "workgroup/conditional_barrier_dynamic"
              EXPECTED_OUTPUT "cond_barriers_dynamic_1_16_1_1_loopvec.stdout"
              COMMAND "run_kernel" "conditional_barrier_dynamic.cl" 1 16 1 1
              WORKITEM_HANDLER "loopvec"
              LABELS "cpu_fail")

add_test_pocl(NAME "workgroup/conditional_barrier_fiber"
              EXPECTED_OUTPUT "cond_barriers_1_2_1_1_fiber.stdout"
              COMMAND "run_kernel" "conditional_barriers.cl" 1 2 1 1
              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "workgroup/forcing_horizontal_parallelization_to_some_outer_loopvec"
              EXPECTED_OUTPUT "outerlooppar_2_2_1_1.stdout"
              COMMAND "run_kernel" "outerlooppar.cl" 2 2 1 1
              WORKITEM_HANDLER "loopvec")

# Bring this back when we have more intelligence on choosing the WI/kernel
# loop vectorization.
#add_test_pocl(NAME "workgroup/outerloop_intelligence_loopvec"
#              EXPECTED_OUTPUT "outerloop_intelligence_1_2_1_1.stdout"
#              COMMAND "run_kernel" "outerloop_intelligence.cl" 1 2 1 1
#              WORKITEM_HANDLER "loopvec")

#add_test_pocl(NAME "workgroup/outerloop_intelligence_fiber"
#              EXPECTED_OUTPUT "outerloop_intelligence_1_2_1_1.stdout"
#              COMMAND "run_kernel" "outerloop_intelligence.cl" 1 2 1 1
#              WORKITEM_HANDLER "fiber")

add_test_pocl(NAME "workgroup/reduce"
              EXPECTED_OUTPUT "reduce_2_512_1_1.stdout"
              COMMAND "run_kernel" "reduce.cl" 2 512 1 1
              LLVM_FILECHECKS "reduce.fc"
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "workgroup/reduce_intra_kernel"
              EXPECTED_OUTPUT "reduce_2_512_1_1.stdout"
              COMMAND "run_kernel" "reduce_intra_kernel.cl" 2 512 1 1
              LLVM_FILECHECKS "reduce_intra.fc"
              # The intra FP reduce only vectorizes with relaxed math.
              ENVIRONMENT "POCL_EXTRA_BUILD_FLAGS=-cl-fast-relaxed-math"
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "workgroup/loop_with_two_paths_to_the_latch"
              EXPECTED_OUTPUT "for_bug_1_2_1_1.stdout"
              UNORDERED_OUTPUT_DIFF
              COMMAND "run_kernel" "for_bug.cl" 1 2 1 1)

#add_test_pocl(NAME "workgroup/loop_with_two_paths_to_the_latch_cbs"
#              EXPECTED_OUTPUT "for_bug_1_2_1_1.stdout"
#              COMMAND "run_kernel" "for_bug.cl" 1 2 1 1
#              WORKITEM_HANDLER "cbs")

add_test_pocl(NAME "workgroup/b_loop_with_two_latches"
              EXPECTED_OUTPUT "multilatch_bloop_1_3_1_1.stdout"
              UNORDERED_OUTPUT_DIFF
              COMMAND "run_kernel" "multilatch_bloop.cl" 1 3 1 1)

#add_test_pocl(NAME "workgroup/b_loop_with_two_latches_cbs"
#              EXPECTED_OUTPUT "multilatch_bloop_1_3_1_1.stdout"
#              COMMAND "run_kernel" "multilatch_bloop.cl" 1 3 1 1
#              WORKITEM_HANDLER "cbs")

add_test_pocl(NAME "workgroup/workgroup_sizes_work_items_get_wrong_ids"
              SORT_OUTPUT
              EXPECTED_OUTPUT "print_all_ids_114114.txt"
              COMMAND "run_kernel" "print_all_ids.cl" 1 1 1 4)

add_test_pocl(NAME "workgroup/issue_548_convergent_propagation"
              EXPECTED_OUTPUT "issue_548_1_2_1_1.stdout"
              COMMAND "run_kernel" "issue_548.cl" "1" "2" "1" "1")

add_test_pocl(NAME "workgroup/issue_1747_wg_divergent_barrier"
              EXPECTED_OUTPUT "issue_1747.stdout"
              COMMAND "run_kernel" "issue_1747.cl" "2" "2" "1" "1")

add_test_pocl(NAME "workgroup/range_md_small_grid"
              EXPECTED_OUTPUT "range_md_small_grid.stdout"
              COMMAND "run_kernel" "range_md.cl" "1" "13" "1" "1")

add_test_pocl(NAME "workgroup/range_md_large_grid"
              EXPECTED_OUTPUT ""
              COMMAND "run_kernel" "range_md.cl" "1000" "128" "1" "1")

add_test_pocl(NAME "workgroup/ballot"
              EXPECTED_OUTPUT "ballot_1_32_1_1.stdout"
              COMMAND "run_kernel" "ballot.cl" "1" "32" "1" "1")

if(HOST_DEVICE_EXTENSIONS MATCHES "cl_khr_subgroup")
  add_test_pocl(NAME "workgroup/sg_shuffle_down_khr"
              EXPECTED_OUTPUT "sg_shuffle_down_1_8_1_1.stdout.khr"
              COMMAND "run_kernel" "simple_sg_shuffle_down.cl" "1" "8" "1" "1")

  add_test_pocl(NAME "workgroup/sg_shuffle_up_khr"
              EXPECTED_OUTPUT "sg_shuffle_up_1_8_1_1.stdout.khr"
              COMMAND "run_kernel" "simple_sg_shuffle_up.cl" "1" "8" "1" "1")
endif()
if(HOST_DEVICE_EXTENSIONS MATCHES "cl_intel_subgroups")
  add_test_pocl(NAME "workgroup/sg_shuffle_down_intel"
              EXPECTED_OUTPUT "sg_shuffle_down_1_8_1_1.stdout.intel"
              COMMAND "run_kernel" "simple_sg_shuffle_down_intel.cl" "1" "8" "1" "1")

  add_test_pocl(NAME "workgroup/sg_shuffle_up_intel"
              EXPECTED_OUTPUT "sg_shuffle_up_1_8_1_1.stdout.intel"
              COMMAND "run_kernel" "simple_sg_shuffle_up_intel.cl" "1" "8" "1" "1")
endif()


add_test_pocl(NAME "workgroup/dynamic_wiloop_bounds"
             EXPECTED_OUTPUT "dynamic_wiloop_bounds.stdout"
             COMMAND "run_kernel" "dynamic_wiloop_bounds.cl" 1 4 4 4
             WORKITEM_HANDLER "loopvec")

set(VARIANTS "loopvec;fiber")
foreach(VARIANT ${VARIANTS})
  set(TEST_LIST "workgroup/ballot_${VARIANT}")
  if(HOST_DEVICE_EXTENSIONS MATCHES "cl_intel_subgroups")
    list(APPEND TEST_LIST
      "workgroup/sg_shuffle_down_intel_${VARIANT}"
      "workgroup/sg_shuffle_up_intel_${VARIANT}")
  endif()
  if(HOST_DEVICE_EXTENSIONS MATCHES "cl_khr_subgroups")
    list(APPEND TEST_LIST
      "workgroup/sg_shuffle_down_khr_${VARIANT}"
      "workgroup/sg_shuffle_up_khr_${VARIANT}")
  endif()

  set_property(TEST ${TEST_LIST}
    APPEND PROPERTY ENVIRONMENT "POCL_MAX_COMPUTE_UNITS=1")
  # Disable subgroup & ballot tests in conformance build.
  if(ENABLE_CONFORMANCE)
    set_tests_properties(${TEST_LIST} PROPERTIES LABELS "cpu_fail")
  endif()
endforeach()


# These tests are now always ran with the basic device with a predefined
# work-group execution order. Their printout verification depends
# on it.
# # TODO: add "cbs" after CBS is brought back.
set(VARIANTS "loopvec;fiber")
foreach(VARIANT ${VARIANTS})
  set(TEST_LIST     "workgroup/unconditional_barriers_${VARIANT}"
  "workgroup/conditional_barrier_${VARIANT}"
  "workgroup/forcing_horizontal_parallelization_to_some_outer_loopvec"
  "workgroup/loop_with_two_paths_to_the_latch_${VARIANT}"
  "workgroup/b_loop_with_two_latches_${VARIANT}"
  "workgroup/workgroup_sizes_work_items_get_wrong_ids_${VARIANT}"
  "workgroup/issue_548_convergent_propagation_${VARIANT}"
  "workgroup/range_md_small_grid_${VARIANT}"
  "workgroup/range_md_large_grid_${VARIANT}"
  "workgroup/issue_1747_wg_divergent_barrier_${VARIANT}"
  "workgroup/ballot_${VARIANT}"
#  "workgroup/outerloop_intelligence_${VARIANT}"
  )
  set_tests_properties(
    ${TEST_LIST}
    PROPERTIES
      COST 2.0
      PROCESSORS 1
      DEPENDS "pocl_version_check")
  set_property(TEST ${TEST_LIST}
    APPEND PROPERTY ENVIRONMENT "POCL_MAX_COMPUTE_UNITS=1")
endforeach()

set_property(TEST
  "workgroup/issue_548_convergent_propagation_loopvec"
#  "workgroup/issue_548_convergent_propagation_cbs"
  APPEND PROPERTY LABELS "mingw_fail")

set_property(TEST
  "workgroup/unbarriered_for_loops_loopvec"
  "workgroup/barriered_for_loops_loopvec"
#  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_cbs"
  APPEND PROPERTY LABELS "proxy")

set_property(TEST
  "workgroup/unbarriered_for_loops_loopvec"
  "workgroup/barriered_for_loops_loopvec"
  "workgroup/switch_case_loopvec"
#  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_cbs"
  APPEND PROPERTY LABELS "vulkan")

# broken:  "workgroup/for_with_divergent_return"
set_property(TEST
  "workgroup/unbarriered_for_loops_loopvec"
  "workgroup/barriered_for_loops_loopvec"
  "workgroup/switch_case_loopvec"
#  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_cbs"
  "workgroup/cond_barriers_in_for_loopvec"
  APPEND PROPERTY LABELS "level0")

add_symlink_to_built_opencl_dynlib(run_kernel)
