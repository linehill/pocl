// Copyright (c) 2025 Pekka Jääskeläinen / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/**
 * Tests a long16 to double16 conversion case that caused a crash with
 * a large enough work-group size due to overflowing the cache. This was
 * due to allocas created for byval temps etc. that could not be removed
 * when wrapped in the WI loop. The fix/workaround is to run -O1 before
 * WILoop formation. This still fails when -cl-disable-opt is given as a
 * build option.
 *
 * LLVM should take care of this case with stack coloring: Now the allocas
 * inside the loop are accumulated across all WIs although their lifespan
 * is only the loop body.
 *
 * The case can be ran in the CTS with
 * test_conformance/conversions/test_conversions "-m" "-w" "double_rtz_ulong"
 */

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/opencl.hpp>
#include <iostream>

const char *SOURCE = R"RAW(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void test_convert_double16_rtz_ulong16( __global ulong16 *src, __global double16 *dest )
{
   size_t i = get_global_id(0);
//   printf ("i == %d dest[i] %p src[i] %p\n", i, &dest[i], &src[i]);
   dest[i] = convert_double16_rtz(src[i]);
}
)RAW";

int main() {
  // 2048 crashes (with -cl-opt-disable much smaller local sizes crash)
  constexpr size_t local_size_x = 128;
  constexpr size_t n_groups = 1;
  cl::Platform platform = cl::Platform::getDefault();
  cl::Device device = cl::Device::getDefault();
  try {
    cl::CommandQueue queue = cl::CommandQueue::getDefault();
    cl::Program program(SOURCE, true);
    cl::Buffer src(CL_MEM_READ_ONLY, sizeof(cl_ulong16) * local_size_x);
    cl::Buffer dest(CL_MEM_READ_ONLY, sizeof(cl_ulong16) * local_size_x);
    cl::Kernel kernel(program, "test_convert_double16_rtz_ulong16");
    kernel.setArg(0, src);
    kernel.setArg(1, dest);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(local_size_x*n_groups),
                               cl::NDRange(local_size_x));
    queue.finish();
  } catch (cl::Error &err) {
    std::cout << "FAIL with OpenCL error = " << err.err() << std::endl;
    return EXIT_FAILURE;
  }

  platform.unloadCompiler();

  std::cout << "OK" << std::endl;
  return EXIT_SUCCESS;
}
