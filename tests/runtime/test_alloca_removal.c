/* Test alloca removal and loop vectorization after function inlining

   Copyright (c) 2025 Raúl Peñacoba / Barcelona Supercomputing Center (BSC)

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
#include "pocl_opencl.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int starting;
    int no_of_edges;
} Node;

const char *kernelSource =
"typedef struct { \n"
"    int starting; \n"
"    int no_of_edges; \n"
"} Node; \n"
"\n"
"__kernel void __attribute((always_inline)) BFS_step_impl( const __global Node* __restrict__ g_graph_nodes, \n"
"                             const __global int* __restrict__ g_graph_edges, \n"
"                             __global char* __restrict__ g_graph_mask, \n"
"                             __global char* __restrict__ g_updating_graph_mask, \n"
"                             __global char* __restrict__ g_graph_visited, \n"
"                             __global int* __restrict__ g_cost, \n"
"                             const int no_of_nodes) { \n"
"    int tid = get_global_id(0); \n"
"    if (tid < no_of_nodes && g_graph_mask[tid]) { \n"
"        g_graph_mask[tid] = 0; \n"
"        int start = g_graph_nodes[tid].starting; \n"
"        int end   = start + g_graph_nodes[tid].no_of_edges; \n"
"        int cache_cost = g_cost[tid]; \n"
"        for (int i = start; i < end; i++) { \n"
"            int nid = g_graph_edges[i]; \n"
"            if (!g_graph_visited[nid]) { \n"
"                g_cost[nid] = cache_cost + 1; \n"
"                g_updating_graph_mask[nid] = 1; \n"
"            } \n"
"        } \n"
"    } \n"
"} \n"
"\n"
"__kernel void BFS_step( const __global Node* __restrict__ g_graph_nodes, \n"
"                        const __global int* __restrict__ g_graph_edges, \n"
"                        __global char* __restrict__ g_graph_mask, \n"
"                        __global char* __restrict__ g_updating_graph_mask, \n"
"                        __global char* __restrict__ g_graph_visited, \n"
"                        __global int* __restrict__ g_cost, \n"
"                        const int no_of_nodes) { \n"
"    if (no_of_nodes == 0) return; \n"
"    BFS_step_impl(g_graph_nodes, \n"
"                  g_graph_edges, \n"
"                  g_graph_mask, \n"
"                  g_updating_graph_mask, \n"
"                  g_graph_visited, \n"
"                  g_cost, \n"
"                  no_of_nodes); \n"
"} \n";

int main() {
    // OpenCL boilerplate
    cl_platform_id platform; cl_device_id device;
    cl_context context; cl_command_queue queue;
    cl_program program; cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");

    // Buffers
    cl_mem d_nodes = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_edges = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_mask = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_updating = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_visited = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_cost = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");

    // Compile kernel
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clBuildProgram");
    kernel = clCreateKernel(program, "BFS_step", &err);
    TEST_ASSERT(kernel);
    CHECK_OPENCL_ERROR_IN("clCreateKernel");

    // Set args
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_nodes);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_edges);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_mask);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_updating);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_visited);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_cost);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");
    // Early kernel exit. We only want to save the LLVM IR for
    // FileCheck
    int no_of_nodes = 0;
    err = clSetKernelArg(kernel, 6, sizeof(int), &no_of_nodes);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");

    size_t globalSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
    clFinish(queue);

    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    return 0;
}
