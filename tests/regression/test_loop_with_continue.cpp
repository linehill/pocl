// Copyright (c) 2025 Henry Linjamäki / Tampere University

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Test for a kernel that loopvec WG method couldn't handle: crashes
// or goes into an infinite loop. Also, triggered an assertion in LLVM
// when built with assertions on.
//
// Also, notice a missed opportunity: the kernel can be transformed to
// an empty one as there are no observable behaviors.

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <iostream>

static char SourceStr[] = R"clc(
kernel void test(int val) {
  while (true) {
    val = val - 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (val == 4)
      continue;
    if (val == 0)
      break;
  }
}
)clc";

int main() try {
  std::vector<cl::Platform> PlatformList;
  cl::Platform::get(&PlatformList);
  auto PlatformName = PlatformList.at(0).getInfo<CL_PLATFORM_NAME>();
  std::cout << "Platform: " << PlatformName << std::endl;

  cl_context_properties CProps[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(PlatformList[0])(), 0};
  cl::Context Context(CL_DEVICE_TYPE_ALL, CProps);

  std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();
  auto DeviceName = Devices.at(0).getInfo<CL_DEVICE_NAME>();
  std::cout << "Device: " << DeviceName << std::endl;

  cl::Program::Sources Source({SourceStr});
  cl::Program program(Context, Source);
  program.build(Devices);
  cl::Kernel TestKernel(program, "test");
  TestKernel.setArg<int>(0, 5);
  cl::CommandQueue Queue(Context, Devices.at(0), 0);
  Queue.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(2, 2),
                             cl::NullRange);
  Queue.finish();
  return 0;
} catch (cl::Error &err) {
  std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
  return 1;
}
