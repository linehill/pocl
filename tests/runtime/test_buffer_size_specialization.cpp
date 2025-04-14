/* Tests for 4G buffer size specialization.

   Copyright (c) 2025 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS

#include "pocl_opencl.h"

#include "../../include/CL/cl_ext_pocl.h"
#include <CL/opencl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>

static char KernelSources[] = R"raw(
  __kernel void vecadd (const __global int *A, const __global int *B,
                        __global int *C) {
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
  }
)raw";

int main(int Argc, char *Argv[]) {

  if (Argc != 2) {
    std::cerr << "The subtest name is missing.";
    return EXIT_FAILURE;
  }

  std::vector<cl::Platform> PlatformList;

  cl::Platform::get(&PlatformList);

  cl_context_properties cprops[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)(PlatformList[0])(), 0};

  cl::Context Context(CL_DEVICE_TYPE_ALL, cprops);

  std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

  if (Devices.empty()) {
    std::cout << "No devices found." << std::endl;
    exit(EXIT_FAILURE);
  }

  cl::Device Dev = Devices[0];

  cl::Program::Sources Sources({KernelSources});
  cl::Program Program(Context, Sources);
  // disable optimizations to avoid removing the truncs.
  Program.build(Dev, "-cl-opt-disable");
  cl::Kernel VecAddKernel(Program, "vecadd");
  cl::CommandQueue Queue(Context, Dev, 0);

  std::string TestName = Argv[1];
  try {

    std::cout << TestName << ":\n";

    cl::Buffer InputBuffer;
    cl::Buffer InputBuffer2;
    cl::Buffer ResultBuffer;

    void *InputSVMBuffer = nullptr;
    void *InputSVMBuffer2 = nullptr;
    void *ResultSVMBuffer = nullptr;

    size_t NumResultData;
    if (TestName == "LargeBuffers") {
      size_t NumInputData = (size_t)UINT32_MAX + 1024;
      NumResultData = NumInputData;

      InputBuffer = cl::Buffer(Context, CL_MEM_READ_ONLY,
                               NumInputData);

      ResultBuffer =
        cl::Buffer(Context, CL_MEM_READ_WRITE,
                   NumResultData);

      VecAddKernel.setArg(0, InputBuffer);
      VecAddKernel.setArg(1, InputBuffer);
      VecAddKernel.setArg(2, ResultBuffer);
    } else if (TestName == "SmallBuffers") {
      size_t NumInputData = UINT32_MAX;
      InputBuffer = cl::Buffer(Context, CL_MEM_READ_ONLY,
                               NumInputData);

      NumResultData = NumInputData;
      ResultBuffer =
        cl::Buffer(Context, CL_MEM_READ_WRITE,
                   NumResultData);

      VecAddKernel.setArg(0, InputBuffer);
      VecAddKernel.setArg(1, InputBuffer);
      VecAddKernel.setArg(2, ResultBuffer);

    } else if (TestName == "MixedBuffers") {
      size_t NumInputData = 1024;
      InputBuffer = cl::Buffer(Context, CL_MEM_READ_ONLY,
                               NumInputData);

      NumResultData = (size_t)UINT32_MAX + 1024;
      ResultBuffer =
        cl::Buffer(Context, CL_MEM_READ_WRITE,
                   NumResultData);

      VecAddKernel.setArg(0, InputBuffer);
      VecAddKernel.setArg(1, InputBuffer);
      VecAddKernel.setArg(2, ResultBuffer);

    } else if (TestName == "LargeSVMBuffers") {

      size_t NumInputData = (size_t)UINT32_MAX + 1024;
      NumResultData = NumInputData;

      InputSVMBuffer =
        clSVMAlloc(Context.get(), CL_MEM_READ_ONLY,
                   NumInputData, 128);

      ResultSVMBuffer =
        clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
                   NumInputData, 128);

      VecAddKernel.setArg(0, InputSVMBuffer);
      VecAddKernel.setArg(1, InputSVMBuffer);
      VecAddKernel.setArg(2, ResultSVMBuffer);

    } else if (TestName == "SmallSVMBuffers") {

      size_t NumInputData = (size_t)UINT32_MAX;
      NumResultData = NumInputData;

      InputSVMBuffer =
        clSVMAlloc(Context.get(), CL_MEM_READ_ONLY,
                   NumInputData, 128);

      ResultSVMBuffer =
        clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
                   NumInputData, 128);

      VecAddKernel.setArg(0, InputSVMBuffer);
      VecAddKernel.setArg(1, InputSVMBuffer);
      VecAddKernel.setArg(2, ResultSVMBuffer);

    } else if (TestName == "MixedSVMBuffers") {

      size_t NumInputData = (size_t)UINT32_MAX + 1024;
      NumResultData = 1024;

      InputSVMBuffer =
        clSVMAlloc(Context.get(), CL_MEM_READ_ONLY,
                   NumInputData, 128);

      ResultSVMBuffer =
        clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
                   NumInputData, 128);

      VecAddKernel.setArg(0, InputSVMBuffer);
      VecAddKernel.setArg(1, InputSVMBuffer);
      VecAddKernel.setArg(2, ResultSVMBuffer);

    } else {
      std::cerr << "Unknown test: " << TestName << "\n";
      return EXIT_FAILURE;
    }

    Queue.enqueueNDRangeKernel(VecAddKernel, cl::NullRange,
                               cl::NDRange(1024*16),
                               cl::NullRange);

    Queue.finish();

    clSVMFree(Context.get(), InputSVMBuffer);
    InputBuffer = nullptr;
    clSVMFree(Context.get(), InputSVMBuffer2);
    InputBuffer2 = nullptr;
    clSVMFree(Context.get(), ResultSVMBuffer);
    ResultBuffer = nullptr;

    CHECK_CL_ERROR(clUnloadCompiler());
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return EXIT_FAILURE;
  }



  std::cout << "OK" << std::endl;
  return EXIT_SUCCESS;
}
