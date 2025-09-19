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
"__kernel void BFS_step_impl( const __global Node* __restrict__ g_graph_nodes, \n"
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
"    BFS_step_impl(g_graph_nodes, \n"
"                  g_graph_edges, \n"
"                  g_graph_mask, \n"
"                  g_updating_graph_mask, \n"
"                  g_graph_visited, \n"
"                  g_cost, \n"
"                  no_of_nodes); \n"
"} \n";

int main() {
    // Grafo: 0->1,2 ; 1->3 ; 2->3
    int no_of_nodes = 4;
    Node graph_nodes[4];
    int edges[] = {1,2, 3, 3}; // 0->1,2; 1->3; 2->3

    // offsets
    graph_nodes[0].starting = 0; graph_nodes[0].no_of_edges = 2;
    graph_nodes[1].starting = 2; graph_nodes[1].no_of_edges = 1;
    graph_nodes[2].starting = 3; graph_nodes[2].no_of_edges = 1;
    graph_nodes[3].starting = 4; graph_nodes[3].no_of_edges = 0;

    // BFS masks y visited
    char graph_mask[4] = {1,0,0,0};   // nivel inicial = {0}
    char updating_mask[4] = {0};
    char visited[4] = {1,0,0,0};
    int cost[4] = {0,-1,-1,-1};

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
    cl_mem d_nodes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(graph_nodes), graph_nodes, NULL);
    cl_mem d_edges = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(edges), edges, NULL);
    cl_mem d_mask = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(graph_mask), graph_mask, NULL);
    cl_mem d_updating = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(updating_mask), updating_mask, NULL);
    cl_mem d_visited = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(visited), visited, NULL);
    cl_mem d_cost = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cost), cost, NULL);

    // Compilar kernel
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
    err = clSetKernelArg(kernel, 6, sizeof(int), &no_of_nodes);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");

    size_t globalSize = no_of_nodes;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
    clFinish(queue);

    // Copiar resultados
    clEnqueueReadBuffer(queue, d_cost, CL_TRUE, 0, sizeof(cost), cost, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_updating, CL_TRUE, 0, sizeof(updating_mask), updating_mask, 0, NULL, NULL);

    // Mostrar resultados
    printf("Costos después de una iteración:\n");
    for (int i = 0; i < no_of_nodes; i++) {
        printf("Nodo %d: %d (update=%d)\n", i, cost[i], updating_mask[i]);
    }

    // Cleanup
    clReleaseMemObject(d_nodes); clReleaseMemObject(d_edges);
    clReleaseMemObject(d_mask); clReleaseMemObject(d_updating);
    clReleaseMemObject(d_visited); clReleaseMemObject(d_cost);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    return 0;
}
