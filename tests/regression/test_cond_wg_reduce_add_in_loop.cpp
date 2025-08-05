/* Tests a kernel with a barrier as the last statement.

   Copyright (c) 2025 Felix Weiglhofer (fweig)

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

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>

#define GPUCA_WARPSIZE 32

cl_device_id gDevice;
cl_context gContext;

const char *kernelSrc = R"(
__kernel void decode(uchar flags)
{
  const ushort N = 1;

  for (ushort i = 0; i < N; i++) {
    if (i != N - 1 || flags) {
      work_group_reduce_add(0);
    }
  }
}
)";

static cl_kernel loadAndCompileKernel() {
  cl_int st;

  size_t srcLen = strlen(kernelSrc);
  auto program =
      clCreateProgramWithSource(gContext, 1, &kernelSrc, &srcLen, &st);
  if (st != CL_SUCCESS) {
    printf("ERROR: Failed to create program with source. Error code: %d\n", st);
    abort();
  }

  st = clBuildProgram(program, 1, &gDevice, "", nullptr, nullptr);
  if (st != CL_SUCCESS) {
    size_t ls;
    clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &ls);
    auto *log = (char *)malloc(ls);
    clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BUILD_LOG, ls, log,
                          nullptr);
    printf("Error during compiling: '%d'\nBuild log:\n%s\n", st, log);
    free(log);
    abort();
  }

  auto k = clCreateKernel(program, "decode", &st);
  if (st != CL_SUCCESS) {
    printf("Error: Failed to get kernel\n");
    abort();
  }

  clReleaseProgram(program);
  return k;
}

int main() {
  cl_int st;
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &gDevice, NULL);

  size_t nameLen;
  clGetDeviceInfo(gDevice, CL_DEVICE_NAME, 0, nullptr, &nameLen);
  auto *name = (char *)malloc(nameLen);
  clGetDeviceInfo(gDevice, CL_DEVICE_NAME, nameLen, name, nullptr);
  printf("OpenCL device: %s\n", name);
  free(name);

  gContext = clCreateContext(NULL, 1, &gDevice, NULL, NULL, NULL);
  auto kernel = loadAndCompileKernel();

  auto queue =
      clCreateCommandQueueWithProperties(gContext, gDevice, 0, nullptr);

  constexpr size_t sMem = sizeof(cl_mem);
  uint8_t flags = 0;
  int idx = 0;
  clSetKernelArg(kernel, idx++, sizeof(flags), &flags);

  size_t global_work_size[] = {GPUCA_WARPSIZE, 0, 0};
  size_t local_work_size[] = {GPUCA_WARPSIZE, 0, 0};
  st = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_work_size,
                              local_work_size, 0, nullptr, nullptr);

  clFinish(queue);

  // --- CLEANUP
  clReleaseCommandQueue(queue);
  clReleaseContext(gContext);

  return 0;
}
