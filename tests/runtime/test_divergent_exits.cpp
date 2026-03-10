// Tests for uniformize-divergent-exits transformation.
//
// Copyright (c) 2026 Henry Linjamäki / Tampere University
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

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <climits>
#include <iostream>
#include <numeric>

struct Target {
  cl::Context Ctx;
  cl::Device Dev;
  cl::CommandQueue CmdQ;
};

static std::pair<cl::Program, cl::Kernel>
buildKernel(Target &T, const std::string &ClSource,
            const std::string &KernelName) {
  cl::Program::Sources Source({ClSource});
  cl::Program Program(T.Ctx, Source);
  Program.build(T.Dev);
  cl::Kernel Kernel(Program, KernelName);
  return std::make_pair(Program, Kernel);
}

static void case0(Target &T) {
  const char SourceStr[] = R"clc(
kernel void test(global int *data, int n, int m, int k) {
  size_t tid = get_global_id(0);
  if (tid >= n)
    return;
  for (int i = 0; i < m; i++)
    data[i * k + tid] = tid * (i + 1);
}
)clc";

  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  size_t GridSize = 64;
  std::vector<int> Data(GridSize, 0.0f);

  cl::Buffer DataBuf =
      cl::Buffer(T.Ctx, Data.begin(), Data.end(), /*useHostPtr=*/false);

  TestKernel.setArg(0, DataBuf);
  TestKernel.setArg<cl_uint>(1, 8);
  TestKernel.setArg<cl_uint>(2, 4);
  TestKernel.setArg<cl_uint>(3, 16);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(GridSize),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(DataBuf, CL_TRUE, 0, GridSize * sizeof(int),
                           Data.data());

  const int Ref[] = {0, 1, 2, 3,  4,  5,  6,  7,  0, 0, 0, 0, 0, 0, 0, 0,
                     0, 2, 4, 6,  8,  10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 3, 6, 9,  12, 15, 18, 21, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0};

  for (unsigned I = 0, E = Data.size(); I < E; I++)
    if (Data[I] != Ref[I]) {
      std::cout << __FUNCTION__ << ": Failure at Data[" << I << "]: Expected '"
                << Ref[I] << "'. Got '" << Data[I] << "'" << std::endl;
      exit(1);
    }
}

static void case1(Target &T) {
  const char SourceStr[] = R"clc(
kernel void test(global int *data, int n, int m, int k) {
  int tid = get_global_id(0);
  if (tid >= n)
    return;
  for (int i = 0; i < m; i++)
    data[i * k + tid] = tid * (i + 1);
}
)clc";

  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  size_t GridSize = 64;
  std::vector<int> Data(GridSize, 0.0f);

  cl::Buffer DataBuf =
      cl::Buffer(T.Ctx, Data.begin(), Data.end(), /*useHostPtr=*/false);

  TestKernel.setArg(0, DataBuf);
  TestKernel.setArg<cl_uint>(1, 8);
  TestKernel.setArg<cl_uint>(2, 4);
  TestKernel.setArg<cl_uint>(3, 16);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(GridSize),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(DataBuf, CL_TRUE, 0, GridSize * sizeof(int),
                           Data.data());

  const int Ref[] = {0, 1, 2, 3,  4,  5,  6,  7,  0, 0, 0, 0, 0, 0, 0, 0,
                     0, 2, 4, 6,  8,  10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 3, 6, 9,  12, 15, 18, 21, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0};

  for (unsigned I = 0, E = Data.size(); I < E; I++)
    if (Data[I] != Ref[I]) {
      std::cout << __FUNCTION__ << ": Failure at Data[" << I << "]: Expected '"
                << Ref[I] << "'. Got '" << Data[I] << "'" << std::endl;
      exit(1);
    }
}

static void case2(Target &T) {
  const char SourceStr[] = R"clc(
kernel void test(global int *data, uint n, int m, int k) {
  uint tid = get_global_id(0); // [1]
  if (tid >= n)
    return;
  for (int i = 0; i < m; i++)
    data[i * k + tid] = tid * (i + 1);
}
)clc";

  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  size_t GridSize = 64;
  std::vector<int> Data(GridSize, 0.0f);

  cl::Buffer DataBuf =
      cl::Buffer(T.Ctx, Data.begin(), Data.end(), /*useHostPtr=*/false);

  TestKernel.setArg(0, DataBuf);
  TestKernel.setArg<cl_uint>(1, 8);
  TestKernel.setArg<cl_uint>(2, 4);
  TestKernel.setArg<cl_uint>(3, 16);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(GridSize),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(DataBuf, CL_TRUE, 0, GridSize * sizeof(int),
                           Data.data());

  const int Ref[] = {0, 1, 2, 3,  4,  5,  6,  7,  0, 0, 0, 0, 0, 0, 0, 0,
                     0, 2, 4, 6,  8,  10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 3, 6, 9,  12, 15, 18, 21, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0};

  for (unsigned I = 0, E = Data.size(); I < E; I++)
    if (Data[I] != Ref[I]) {
      std::cout << __FUNCTION__ << ": Failure at Data[" << I << "]: Expected '"
                << Ref[I] << "'. Got '" << Data[I] << "'" << std::endl;
      exit(1);
    }
}

static void case3(Target &T) {
  // Test a overflow case. Here work-item 'INT_MAX + 1' should not
  // exit the kernel. Note: size_t->int conversion is
  // implementation-defined (in C99) - assuming here the extra MSB
  // bits are thrown away and left-over bits are reinterpreted as int.
  const char SourceStr[] = R"clc(
kernel void test(global uint *count, int n) {
  int tid = get_global_id(0); // (A)
  if (tid >= n)
    return;
  atomic_fetch_add((global atomic_uint *)count, (uint)1);
}
)clc";
  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  size_t GridSize = static_cast<size_t>(INT_MAX) + 2;
  cl_uint Count = 0;

  cl::Buffer CountBuf =
      cl::Buffer(T.Ctx, &Count, &Count + 1, /*useHostPtr=*/false);

  TestKernel.setArg(0, CountBuf);
  TestKernel.setArg<cl_int>(1, 1);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(GridSize),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(CountBuf, CL_TRUE, 0, sizeof(cl_uint), &Count);

  if (Count != 2) {
    std::cout << __FUNCTION__ << ": Failure. Expected '2'. Got '" << Count
              << "'." << std::endl;
    exit(1);
  }
}

static void case4(Target &T) {
  const char SourceStr[] = R"clc(
kernel void test(global uint *count, int n) {
  int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (tid >= n)
    return;
  atomic_fetch_add((global atomic_uint *)count, (uint)1);
}
)clc";
  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  size_t GridSize = 128u;
  cl_uint Count = 0;

  cl::Buffer CountBuf =
      cl::Buffer(T.Ctx, &Count, &Count + 1, /*useHostPtr=*/false);

  TestKernel.setArg(0, CountBuf);
  TestKernel.setArg<cl_int>(1, 17);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(GridSize),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(CountBuf, CL_TRUE, 0, sizeof(cl_uint), &Count);

  if (Count != 17) {
    std::cout << __FUNCTION__ << ": Failure. Expected '17'. Got '" << Count
              << "'." << std::endl;
    exit(1);
  }
}

static void case5(Target &T) {
  // Mimics an early-exit pattern commonly seen in HIP (at least in HeCBench).
  //
  //   int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //   if(tid >= n) return;
  //
  // The type of HIP built-in variables (block* and threadIdx) is uint3.

  const char SourceStr[] = R"clc(
kernel void test(global uint *count, int n) {
  int tid = (uint)get_group_id(0) * (uint)get_local_size(0) + (uint)get_local_id(0);
  if (tid >= n)
    return;
  atomic_fetch_add((global atomic_uint *)count, (uint)1);
}
)clc";
  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  size_t GridSize = 128u;
  cl_uint Count = 0;

  cl::Buffer CountBuf =
      cl::Buffer(T.Ctx, &Count, &Count + 1, /*useHostPtr=*/false);

  TestKernel.setArg(0, CountBuf);
  TestKernel.setArg<cl_int>(1, 17);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(GridSize),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(CountBuf, CL_TRUE, 0, sizeof(cl_uint), &Count);

  if (Count != 17) {
    std::cout << __FUNCTION__ << ": Failure. Expected '17'. Got '" << Count
              << "'." << std::endl;
    exit(1);
  }
}

static void case6(Target &T) {
  // Mimics an early-exit pattern in a HIP case (bilateral-hip):
  //
  //  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  //  const int idy = blockIdx.y*blockDim.y + threadIdx.y;
  //  if(idx >= w || idy >= h) return;

  const char SourceStr[] = R"clc(
kernel void test(global uint *count, int w, int h) {
  int idx = (uint)get_group_id(0) * (uint)get_local_size(0) +
            (uint)get_local_id(0);
  int idy = (uint)get_group_id(1) * (uint)get_local_size(1) +
            (uint)get_local_id(1);
  if (idx >= w || idy >= h)
    return;
  atomic_fetch_add((global atomic_uint *)count, (uint)1);
}
)clc";
  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  cl_uint Count = 0;

  cl::Buffer CountBuf =
      cl::Buffer(T.Ctx, &Count, &Count + 1, /*useHostPtr=*/false);

  TestKernel.setArg(0, CountBuf);
  TestKernel.setArg<cl_int>(1, 3);
  TestKernel.setArg<cl_int>(2, 4);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(8, 8),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(CountBuf, CL_TRUE, 0, sizeof(cl_uint), &Count);

  if (Count != 12) {
    std::cout << __FUNCTION__ << ": Failure. Expected '12'. Got '" << Count
              << "'." << std::endl;
    exit(1);
  }
}

static void case7(Target &T) {
  // A regression test derived from a failing OpenVX case. The failure
  // was caused by mixing up elements in context saves and restores.

  const char SourceStr[] = R"clc(
kernel void test(global int *data, uint stride, int w, int h) {
  int idx = (uint)get_group_id(0) * (uint)get_local_size(0) +
            (uint)get_local_id(0);
  int idy = (uint)get_group_id(1) * (uint)get_local_size(1) +
            (uint)get_local_id(1);
  if (idx < w && idy < h) {
    unsigned pos = idy * stride + idx;
    int tmp = data[pos];
    data[pos] = tmp < 0 ? -tmp : tmp;
  }
  return;
}
)clc";
  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  const int H = 6, W = 7;
  const int NumElts = H * W;
  std::vector<int> Data(NumElts);
  for (int I = 0; I < Data.size(); I++)
    Data[I] = (I % 2) ? I : -I;

  cl::Buffer DataBuf = cl::Buffer(T.Ctx, Data.data(), Data.data() + NumElts,
                                  /*useHostPtr=*/false);

  const int SubW = 3, SubH = 4;
  TestKernel.setArg(0, DataBuf);
  TestKernel.setArg<cl_uint>(1, W);
  TestKernel.setArg<cl_int>(2, SubW);
  TestKernel.setArg<cl_int>(3, SubH);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(8, 8),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(DataBuf, CL_TRUE, 0, sizeof(cl_int) * NumElts,
                           Data.data());

  auto GetRef = [&](int X, int Y) -> int {
    int Res = Y * W + X;
    Res = (Res % 2) ? Res : -Res;
    if (X < SubW && Y < SubH)
      Res = Res < 0 ? -Res : Res;
    return Res;
  };

  for (int Y = 0; Y < H; Y++) {
    for (int X = 0; X < W; X++) {
      int Elt = Data[Y * W + X];
      int Ref = GetRef(X, Y);
      if (Elt != Ref) {
        std::cout << __FUNCTION__ << ": Failure at Data[" << X << ", " << Y
                  << "]: Expected '" << Ref << "'. Got '" << Elt << "'"
                  << std::endl;
        exit(1);
      }
    }
  }
}

static void case8(Target &T) {
  // A corner case constructed from an option in the
  // uniformize-divergent-exit source which effectively removes the
  // return instruction at (1) but doing so caused thee following
  // kernel enter into an infinite loop.

  const char SourceStr[] = R"clc(
kernel void test(global int *count, ulong n) {
  if (get_global_id(0) >= n) // n is intended to be zero.
    return; // (1)

  // Control should not reach here.
  while(1)
    atomic_fetch_add((global atomic_int *)count, 1);
}
)clc";
  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  int Count = 0;
  cl::Buffer DataBuf = cl::Buffer(T.Ctx, &Count, &Count + 1,
                                  /*useHostPtr=*/false);

  TestKernel.setArg(0, DataBuf);
  TestKernel.setArg<cl_ulong>(1, 0);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(8),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(DataBuf, CL_TRUE, 0, sizeof(cl_int), &Count);

  if (Count) {
    std::cout << __FUNCTION__ << ": Failure: expected 'Count' to be zero!"
              << std::endl;
    exit(1);
  }
}

static void case9(Target &T) {
  const char SourceStr[] = R"clc(
kernel void test(global int *data, ulong n) {
  size_t tid = get_global_id(0);
  if (get_global_id(0) < n)
    return;
  data[tid] = tid;
}
)clc";
  auto [Program, TestKernel] = buildKernel(T, SourceStr, "test");
  size_t GridSize = 8;
  std::vector<int> Data(GridSize, 0);

  cl::Buffer DataBuf =
      cl::Buffer(T.Ctx, Data.begin(), Data.end(), /*useHostPtr=*/false);

  TestKernel.setArg(0, DataBuf);
  TestKernel.setArg<cl_ulong>(1, 4);
  T.CmdQ.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(GridSize),
                              cl::NullRange);
  T.CmdQ.enqueueReadBuffer(DataBuf, CL_TRUE, 0, GridSize * sizeof(int),
                           Data.data());

  for (unsigned I = 0, E = Data.size(); I < E; I++) {
    int Ref = (I < 4) ? 0 : I;

    if (Data[I] != Ref) {
      std::cout << __FUNCTION__ << ": Failure at Data[" << I << "]: Expected '"
                << Ref << "'. Got '" << Data[I] << "'" << std::endl;
      exit(1);
    }
  }
}

std::vector<std::function<void(Target &)>> Cases = {
    case0, case1, case2, case3, case4, case5, case6, case7, case8, case9};

int main(int ArgC, char *ArgV[]) try {
  // uniformize-divergent-exits is default off (for now).
  setenv("POCL_UNIFORMIZE_DIVERGENT_EXITS", "1", 1);

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

  cl::CommandQueue Queue(Context, Devices.at(0), 0);

  Target T{Context, Devices.at(0), Queue};

  switch (ArgC) {
  default:
    return 2;
  case 1:
    for (auto Case : Cases)
      Case(T);
    break;
  case 2:
    Cases.at(std::atoi(ArgV[1]))(T);
    break;
  }

  std::cout << "OK\n";
  return 0;
} catch (cl::Error &err) {
  std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
  return 1;
}
