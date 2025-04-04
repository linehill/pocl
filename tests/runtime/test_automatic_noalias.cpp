/* Tests for corner cases of automated noalias inference.

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

  constexpr size_t NumData = 16;
  std::vector<int> HostBufA, HostBufB, HostBufC;
  for (size_t i = 0; i < NumData; ++i) {
    HostBufA.push_back(i);
    HostBufB.push_back(0);
    HostBufC.push_back(3);
  }

  cl::Buffer ABuffer =
    cl::Buffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
               sizeof(cl_int) * NumData, HostBufA.data());

  cl::Buffer BBuffer =
    cl::Buffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
               sizeof(cl_int) * NumData, HostBufB.data());

  cl::Buffer CBuffer =
    cl::Buffer(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
               sizeof(cl_int) * NumData, HostBufC.data());

  void *ASVM =
    clSVMAlloc(Context.get(), CL_MEM_READ_ONLY,
               sizeof(cl_int) * NumData, 128);

  void *BSVM =
    clSVMAlloc(Context.get(), CL_MEM_READ_ONLY,
               sizeof(cl_int) * NumData, 128);

  void *CSVM =
    clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
               sizeof(cl_int) * NumData, 128);

  cl::Program::Sources Sources({KernelSources});
  cl::Program Program(Context, Sources);
  Program.build(Dev);
  cl::Kernel VecAddKernel(Program, "vecadd");
  cl::CommandQueue Queue(Context, Dev, 0);

  std::string TestName = Argv[1];
  try {

    std::cout << TestName << ":\n";

    if (TestName == "DisjointBuffers") {
      VecAddKernel.setArg(0, ABuffer);
      VecAddKernel.setArg(1, BBuffer);
      VecAddKernel.setArg(2, CBuffer);
    } else if (TestName == "InplaceBuffers") {
      VecAddKernel.setArg(0, CBuffer);
      VecAddKernel.setArg(1, BBuffer);
      VecAddKernel.setArg(2, CBuffer);
    } else if (TestName == "DisjointCoarseSVM") {
      VecAddKernel.setArg(0, ASVM);
      VecAddKernel.setArg(1, BSVM);
      VecAddKernel.setArg(2, CSVM);
    } else if (TestName == "InplaceCoarseSVM") {
      VecAddKernel.setArg(0, CSVM);
      VecAddKernel.setArg(1, BSVM);
      VecAddKernel.setArg(2, CSVM);
    } else if (TestName == "DisjointCoarseSVMWithIndirect") {
      VecAddKernel.setArg(0, ASVM);
      VecAddKernel.setArg(1, BSVM);
      VecAddKernel.setArg(2, CSVM);

      // This signals an overlapping buffer that the kernel might
      // access and we don't analyze all the memory accesses
      // yet.
      if (::clSetKernelExecInfo(VecAddKernel.get(),
                                CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                sizeof(void *), &CSVM) == CL_INVALID_OPERATION) {
        std::cerr << "ERROR: failed setting SVM. Not supported?\n";
        return -77; // Skip test.
      }
    } else if (TestName == "DisjointCoarseSVMWithFGSystem") {
      VecAddKernel.setArg(0, ASVM);
      VecAddKernel.setArg(1, BSVM);
      VecAddKernel.setArg(2, CSVM);

      // This signals an overlapping buffer that the kernel might
      // access and we don't analyze all the memory accesses
      // yet.
      cl_bool FGS = CL_TRUE;
      if (::clSetKernelExecInfo(VecAddKernel.get(),
                                CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM,
                                sizeof(cl_bool), &FGS) == CL_INVALID_OPERATION) {
        std::cerr << "ERROR: failed setting SVM. Not supported?\n";
        return -77; // Skip test.
      }
    } else {
      std::cerr << "Unknown test: " << TestName << "\n";
      return EXIT_FAILURE;
    }

    Queue.enqueueNDRangeKernel(VecAddKernel, cl::NullRange,
                               cl::NDRange(NumData),
                               cl::NullRange);

    Queue.enqueueReadBuffer(CBuffer, CL_TRUE, 0, sizeof(cl_int) * NumData,
                            HostBufC.data());

    CHECK_CL_ERROR(clUnloadCompiler());
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return EXIT_FAILURE;
  }

  clSVMFree(Context.get(), ASVM);
  clSVMFree(Context.get(), BSVM);
  clSVMFree(Context.get(), CSVM);

  std::cout << "OK" << std::endl;
  return EXIT_SUCCESS;
}
