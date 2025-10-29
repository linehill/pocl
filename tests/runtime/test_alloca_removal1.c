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

const char *kernelSource =
"#define fp float\n"
"\n"
"#define NUMBER_THREADS 256\n"
"\n"
"typedef struct params_common{\n"
"\n"
"	int sSize;\n"
"	int tSize;\n"
"	fp alpha;\n"
"\n"
"	int no_frames;\n"
"	int frame_rows;\n"
"\n"
"	int endoPoints;\n"
"\n"
"	int in_rows;\n"
"	int in_cols;\n"
"	int in_elem;\n"
"\n"
"	int in2_rows;\n"
"	int in2_cols;\n"
"	int in2_elem;\n"
"\n"
"	int conv_rows;\n"
"	int conv_elem;\n"
"	int ioffset;\n"
"	int joffset;\n"
"\n"
"	int in2_pad_add_rows;\n"
"	int in2_pad_add_cols;\n"
"	int in2_pad_cumv_rows;\n"
"	int in2_pad_cumv_cols;\n"
"	int in2_pad_cumv_elem;\n"
"\n"
"	int in2_pad_cumv_sel_rows;\n"
"	int in2_pad_cumv_sel_elem;\n"
"	int in2_pad_cumv_sel_rowlow;\n"
"	int in2_pad_cumv_sel_collow;\n"
"\n"
"	int in2_pad_cumv_sel2_rowlow;\n"
"	int in2_pad_cumv_sel2_collow;\n"
"	int in2_sub_cumh_rows;\n"
"	int in2_sub_cumh_elem;\n"
"\n"
"	int in2_sub_cumh_sel_rows;\n"
"	int in2_sub_cumh_sel_elem;\n"
"	int in2_sub_cumh_sel_rowlow;\n"
"	int in2_sub_cumh_sel_collow;\n"
"\n"
"	int in2_sub_cumh_sel2_rowlow;\n"
"	int in2_sub_cumh_sel2_collow;\n"
"	int in2_sub2_rows;\n"
"	int in2_sub2_elem;\n"
"\n"
"	int in2_sqr_rows;\n"
"	int in2_sqr_cols;\n"
"	int in2_sqr_elem;\n"
"\n"
"	int in2_sqr_sub2_elem;\n"
"\n"
"	int in_sqr_rows;\n"
"	int in_sqr_cols;\n"
"	int in_sqr_elem;\n"
"\n"
"	int tMask_rows;\n"
"	int tMask_cols;\n"
"	int tMask_elem;\n"
"\n"
"	int mask_rows;\n"
"	int mask_cols;\n"
"\n"
"	int mask_conv_rows;\n"
"	int mask_conv_cols;\n"
"	int mask_conv_elem;\n"
"	int mask_conv_ioffset;\n"
"	int mask_conv_joffset;\n"
"\n"
"} params_common;\n"
"\n"
"__kernel void __attribute__((always_inline))\n"
"kernel_gpu_opencl_impl(\n"
"					params_common d_common,              // 0\n"
"					__global fp* d_frame,                // 1\n"
"					int d_frame_no,                      // 2\n"
"					__global int* d_endoRow,             // 3\n"
"					__global int* d_endoCol,             // 4\n"
"					__global int* d_tEndoRowLoc,         // 5\n"
"					__global int* d_tEndoColLoc,         // 6\n"
"					__global int* d_epiRow,              // 7\n"
"					__global int* d_epiCol,              // 8\n"
"					__global int* d_tEpiRowLoc,          // 9\n"
"					__global int* d_tEpiColLoc,          // 10\n"
"					__global fp* d_endoT,                // 11\n"
"					__global fp* d_epiT,                 // 12\n"
"					__global fp* d_in2_all,              // 13\n"
"					__global fp* d_conv_all,             // 14\n"
"					__global fp* d_in2_pad_cumv_all,     // 15\n"
"					__global fp* d_in2_pad_cumv_sel_all, // 16\n"
"					__global fp* d_in2_sub_cumh_all,     // 17\n"
"					__global fp* d_in2_sub_cumh_sel_all, // 18\n"
"					__global fp* d_in2_sub2_all,         // 19\n"
"					__global fp* d_in2_sqr_all,          // 20\n"
"					__global fp* d_in2_sqr_sub2_all,     // 21\n"
"					__global fp* d_in_sqr_all,           // 22\n"
"					__global fp* d_tMask_all,            // 23\n"
"					__global fp* d_mask_conv_all,        // 24\n"
"					__global fp* d_in_mod_temp_all,      // 25\n"
"					__global fp* in_partial_sum_all,     // 26\n"
"					__global fp* in_sqr_partial_sum_all, // 27\n"
"					__global fp* par_max_val_all,        // 28\n"
"					__global int* par_max_coo_all,       // 29\n"
"					__global fp* in_final_sum_all,       // 30\n"
"					__global fp* in_sqr_final_sum_all,   // 31\n"
"					__global fp* denomT_all,             // 32\n"
"					__global fp* checksum)               // 33\n"
"\n"
"{\n"
"	int i;\n"
"	int row;\n"
"	int col;\n"
"	int ori_row;\n"
"	int ori_col;\n"
"	int position;\n"
"	fp sum;\n"
"	int pos_ori;\n"
"	fp temp;\n"
"	fp temp2;\n"
"	int location;\n"
"	int tMask_row; \n"
"	int tMask_col;\n"
"	fp largest_value_current = 0;\n"
"	fp largest_value = 0;\n"
"	int largest_coordinate_current = 0;\n"
"	int largest_coordinate = 0;\n"
"	fp fin_max_val = 0;\n"
"	int largest_col;\n"
"	int offset_row;\n"
"	int offset_col;\n"
"	int pointer;\n"
"	int ori_pointer;\n"
"\n"
"	int bx = get_group_id(0);\n"
"	int tx = get_local_id(0);\n"
"	int ei_new;\n"
"\n"
"	__global fp* d_common_change_d_frame = &d_frame[0];\n"
"\n"
"	int d_unique_point_no;\n"
"	__global int* d_unique_d_Row;\n"
"	__global int* d_unique_d_Col;\n"
"	__global int* d_unique_d_tRowLoc;\n"
"	__global int* d_unique_d_tColLoc;\n"
"	__global fp* d_in;\n"
"	if(bx < d_common.endoPoints){\n"
"		d_unique_point_no = bx;\n"
"		d_unique_d_Row = d_endoRow;\n"
"		d_unique_d_Col = d_endoCol;\n"
"		d_unique_d_tRowLoc = d_tEndoRowLoc;\n"
"		d_unique_d_tColLoc = d_tEndoColLoc;\n"
"		d_in = &d_endoT[d_unique_point_no * d_common.in_elem];\n"
"	}\n"
"	else{\n"
"		d_unique_point_no = bx-d_common.endoPoints;\n"
"		d_unique_d_Row = d_epiRow;\n"
"		d_unique_d_Col = d_epiCol;\n"
"		d_unique_d_tRowLoc = d_tEpiRowLoc;\n"
"		d_unique_d_tColLoc = d_tEpiColLoc;\n"
"		d_in = &d_epiT[d_unique_point_no * d_common.in_elem];\n"
"	}\n"
"\n"
"	__global fp* d_unique_d_in2 = &d_in2_all[bx*d_common.in2_elem];\n"
"	__global fp* d_unique_d_conv = &d_conv_all[bx*d_common.conv_elem];\n"
"	__global fp* d_unique_d_in2_pad_cumv = &d_in2_pad_cumv_all[bx*d_common.in2_pad_cumv_elem];\n"
"	__global fp* d_unique_d_in2_sub_cumh = &d_in2_sub_cumh_all[bx*d_common.in2_sub_cumh_elem];\n"
"	__global fp* d_unique_d_in2_sub2 = &d_in2_sub2_all[bx*d_common.in2_sub2_elem];\n"
"	__global fp* d_unique_d_in2_sqr = &d_in2_sqr_all[bx*d_common.in2_sqr_elem];\n"
"	__global fp* d_unique_d_in2_sqr_sub2 = &d_in2_sqr_sub2_all[bx*d_common.in2_sqr_sub2_elem];\n"
"	__global fp* d_unique_d_in_sqr = &d_in_sqr_all[bx*d_common.in_sqr_elem];\n"
"	__global fp* d_unique_d_tMask = &d_tMask_all[bx*d_common.tMask_elem];\n"
"	__global fp* d_unique_d_mask_conv = &d_mask_conv_all[bx*d_common.mask_conv_elem];\n"
"\n"
"	__global fp* in_partial_sum = &in_partial_sum_all[bx*d_common.in_cols];\n"
"	__global fp* par_max_val = &par_max_val_all[bx*d_common.mask_conv_rows];\n"
"	__global int* par_max_coo = &par_max_coo_all[bx*d_common.mask_conv_rows];\n"
"\n"
"	__global fp* in_final_sum = &in_final_sum_all[bx];\n"
"	__global fp* in_sqr_final_sum = &in_sqr_final_sum_all[bx];\n"
"	__global fp* denomT = &denomT_all[bx];\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in_elem){\n"
"\n"
"		row = (ei_new+1) % d_common.in_rows - 1;\n"
"		col = (ei_new+1) / d_common.in_rows + 1 - 1;\n"
"		if((ei_new+1) % d_common.in_rows == 0){\n"
"			row = d_common.in_rows - 1;\n"
"			col = col-1;\n"
"		}\n"
"\n"
"		ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;\n"
"		ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;\n"
"		ori_pointer = ori_col*d_common.frame_rows+ori_row;\n"
"\n"
"		d_in[col*d_common.in_rows+row] = d_common_change_d_frame[ori_pointer];\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in2_pad_cumv_elem){\n"
"\n"
"		row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;\n"
"		col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;\n"
"		if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){\n"
"			row = d_common.in2_pad_cumv_rows - 1;\n"
"			col = col-1;\n"
"		}\n"
"\n"
"		if(	row > (d_common.in2_pad_add_rows-1) &&\n"
"			row < (d_common.in2_pad_add_rows+d_common.in2_rows) && \n"
"			col > (d_common.in2_pad_add_cols-1) && \n"
"			col < (d_common.in2_pad_add_cols+d_common.in2_cols)){\n"
"			ori_row = row - d_common.in2_pad_add_rows;\n"
"			ori_col = col - d_common.in2_pad_add_cols;\n"
"			d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_conv[ori_col*d_common.in2_rows+ori_row];\n"
"		}\n"
"		else{\n"
"			d_unique_d_in2_pad_cumv[ei_new] = 0;\n"
"		}\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"	}\n"
"\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in2_pad_cumv_cols){\n"
"\n"
"		pos_ori = ei_new*d_common.in2_pad_cumv_rows;\n"
"\n"
"		sum = 0;\n"
"		\n"
"		for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){\n"
"			d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;\n"
"			sum = d_unique_d_in2_pad_cumv[position];\n"
"		}\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in2_sqr_elem){\n"
"\n"
"		temp = d_unique_d_in2[ei_new];\n"
"		d_unique_d_in2_sqr[ei_new] = temp * temp;\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in2_sub_cumh_rows){\n"
"\n"
"		sum = 0;\n"
"\n"
"		for(position = 0; position < d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){\n"
"			d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;\n"
"			sum = d_unique_d_in2_sub_cumh[position];\n"
"		}\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in2_sub2_elem){\n"
"\n"
"		temp = d_unique_d_in2_sub2[ei_new];\n"
"		temp2 = d_unique_d_in2_sqr_sub2[ei_new] - (temp / d_common.in_elem);\n"
"		if(temp2 < 0){\n"
"			temp2 = 0;\n"
"		}\n"
"		d_unique_d_in2_sqr_sub2[ei_new] = temp2;\n"
"		\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in_sqr_elem){\n"
"\n"
"		temp = d_in[ei_new];\n"
"		d_unique_d_in_sqr[ei_new] = temp * temp;\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"\n"
"	in_final_sum[0] = 0;\n"
"	for(i = 0; i<d_common.in_cols; i++){\n"
"		in_final_sum[0] = in_final_sum[0] + in_partial_sum[i];\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	if(tx == 0){\n"
"		denomT[0] = sqrt((float)(d_common.in_elem));\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.in2_sub2_elem){\n"
"		d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_conv[ei_new];\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	if(d_frame_no == 0){\n"
"	}\n"
"	else{\n"
"		tMask_row = d_unique_d_Row[d_common.no_frames];\n"
"	}\n"
"\n"
"// En este bucle si quito uno de los ifs sale que se vectoriza un bucle mas\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.tMask_elem){\n"
"\n"
"		location = tMask_col*d_common.tMask_rows + tMask_row;\n"
"\n"
"		if(ei_new==location){\n"
"			d_unique_d_tMask[ei_new] = 1;\n"
"		}\n"
"		else{\n"
"			d_unique_d_tMask[ei_new] = 0;\n"
"		}\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	ei_new = tx;\n"
"	while(ei_new < d_common.mask_conv_rows){\n"
"\n"
"		for(i=0; i<d_common.mask_conv_cols; i++){\n"
"			largest_value_current = d_unique_d_mask_conv[largest_coordinate_current];\n"
"			if(largest_value_current > largest_value){\n"
"				largest_coordinate = largest_coordinate_current;\n"
"			}\n"
"		}\n"
"		par_max_coo[ei_new] = largest_coordinate;\n"
"\n"
"		ei_new = ei_new + NUMBER_THREADS;\n"
"\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
"\n"
"	for(i = 0; i < d_common.mask_conv_rows; i++){\n"
"		if(par_max_val[i] > fin_max_val){\n"
"			fin_max_val = par_max_val[i];\n"
"		}\n"
"	}\n"
"\n"
"	offset_row = d_common.in_rows - d_common.sSize;\n"
"	offset_col = largest_col - d_common.in_cols - (d_common.sSize - d_common.tSize);\n"
"	pointer = d_common.no_frames+d_frame_no;\n"
"	d_unique_d_tRowLoc[pointer] = offset_row;\n"
"	d_unique_d_tColLoc[pointer] = offset_col;\n"
"\n"
"\n"
"}\n"
"\n"
"\n"
"__kernel void \n"
"kernel_gpu_opencl(\n"
"					params_common d_common,              // 0\n"
"					__global fp* d_frame,                // 1\n"
"					int d_frame_no,                      // 2\n"
"					__global int* d_endoRow,             // 3\n"
"					__global int* d_endoCol,             // 4\n"
"					__global int* d_tEndoRowLoc,         // 5\n"
"					__global int* d_tEndoColLoc,         // 6\n"
"					__global int* d_epiRow,              // 7\n"
"					__global int* d_epiCol,              // 8\n"
"					__global int* d_tEpiRowLoc,          // 9\n"
"					__global int* d_tEpiColLoc,          // 10\n"
"					__global fp* d_endoT,                // 11\n"
"					__global fp* d_epiT,                 // 12\n"
"					__global fp* d_in2_all,              // 13\n"
"					__global fp* d_conv_all,             // 14\n"
"					__global fp* d_in2_pad_cumv_all,     // 15\n"
"					__global fp* d_in2_pad_cumv_sel_all, // 16\n"
"					__global fp* d_in2_sub_cumh_all,     // 17\n"
"					__global fp* d_in2_sub_cumh_sel_all, // 18\n"
"					__global fp* d_in2_sub2_all,         // 19\n"
"					__global fp* d_in2_sqr_all,          // 20\n"
"					__global fp* d_in2_sqr_sub2_all,     // 21\n"
"					__global fp* d_in_sqr_all,           // 22\n"
"					__global fp* d_tMask_all,            // 23\n"
"					__global fp* d_mask_conv_all,        // 24\n"
"					__global fp* d_in_mod_temp_all,      // 25\n"
"					__global fp* in_partial_sum_all,     // 26\n"
"					__global fp* in_sqr_partial_sum_all, // 27\n"
"					__global fp* par_max_val_all,        // 28\n"
"					__global int* par_max_coo_all,       // 29\n"
"					__global fp* in_final_sum_all,       // 30\n"
"					__global fp* in_sqr_final_sum_all,   // 31\n"
"					__global fp* denomT_all,             // 32\n"
"					__global fp* checksum)               // 33\n"
"\n"
"{\n"
"if (d_frame_no == 0) return;\n"
"kernel_gpu_opencl_impl(d_common,\n"
"                       d_frame,\n"
"                       d_frame_no,\n"
"                       d_endoRow,\n"
"                       d_endoCol,\n"
"                       d_tEndoRowLoc,\n"
"                       d_tEndoColLoc,\n"
"                       d_epiRow,\n"
"                       d_epiCol,\n"
"                       d_tEpiRowLoc,\n"
"                       d_tEpiColLoc,\n"
"                       d_endoT,\n"
"                       d_epiT,\n"
"                       d_in2_all,\n"
"                       d_conv_all,\n"
"                       d_in2_pad_cumv_all,\n"
"                       d_in2_pad_cumv_sel_all,\n"
"                       d_in2_sub_cumh_all,\n"
"                       d_in2_sub_cumh_sel_all,\n"
"                       d_in2_sub2_all,\n"
"                       d_in2_sqr_all,\n"
"                       d_in2_sqr_sub2_all,\n"
"                       d_in_sqr_all,\n"
"                       d_tMask_all,\n"
"                       d_mask_conv_all,\n"
"                       d_in_mod_temp_all,\n"
"                       in_partial_sum_all,\n"
"                       in_sqr_partial_sum_all,\n"
"                       par_max_val_all,\n"
"                       par_max_coo_all,\n"
"                       in_final_sum_all,\n"
"                       in_sqr_final_sum_all,\n"
"                       denomT_all,\n"
"                       checksum);\n"
"}\n";

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

    cl_mem d_common = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_frame = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_endoRow = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_endoCol = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_tEndoRowLoc = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_tEndoColLoc = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_epiRow = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_epiCol = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_tEpiRowLoc = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_tEpiColLoc = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_endoT = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_epiT = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_conv = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2_pad_cumv = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2_pad_cumv_sel = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2_sub_cumh = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2_sub_cumh_sel = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2_sub2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2_sqr = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in2_sqr_sub2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in_sqr = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_tMask = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_mask_conv = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_in_mod_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem in_partial_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem in_sqr_partial_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem par_max_val = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem par_max_coo = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem in_final_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem in_sqr_final_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem denomT = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");
    cl_mem d_checksum = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer");

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clBuildProgram");
    kernel = clCreateKernel(program, "kernel_gpu_opencl", &err);
    TEST_ASSERT(kernel);
    CHECK_OPENCL_ERROR_IN("clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_common);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_frame);
    // Used to skip kernel execution
    int frame_no = 0;
    err = clSetKernelArg(kernel, 2, sizeof(int), (void *)&frame_no);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_endoRow);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_endoCol);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&d_tEndoRowLoc);

    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&d_tEndoColLoc);
    err = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&d_epiRow);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&d_epiCol);
    err = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&d_tEpiRowLoc);
    err = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&d_tEpiColLoc);
    err = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&d_endoT);
    err = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&d_epiT);
    err = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *)&d_in2);
    err = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void *)&d_conv);
    err = clSetKernelArg(kernel, 15, sizeof(cl_mem), (void *)&d_in2_pad_cumv);
    err = clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *)&d_in2_pad_cumv_sel);
    err = clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *)&d_in2_sub_cumh);
    err = clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *)&d_in2_sub_cumh_sel);
    err = clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *)&d_in2_sub2);
    err = clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *)&d_in2_sqr);
    err = clSetKernelArg(kernel, 21, sizeof(cl_mem), (void *)&d_in2_sqr_sub2);
    err = clSetKernelArg(kernel, 22, sizeof(cl_mem), (void *)&d_in_sqr);
    err = clSetKernelArg(kernel, 23, sizeof(cl_mem), (void *)&d_tMask);
    err = clSetKernelArg(kernel, 24, sizeof(cl_mem), (void *)&d_mask_conv);
    err = clSetKernelArg(kernel, 25, sizeof(cl_mem), (void *)&d_in_mod_temp);
    err = clSetKernelArg(kernel, 26, sizeof(cl_mem), (void *)&in_partial_sum);
    err = clSetKernelArg(kernel, 27, sizeof(cl_mem), (void *)&in_sqr_partial_sum);
    err = clSetKernelArg(kernel, 28, sizeof(cl_mem), (void *)&par_max_val);
    err = clSetKernelArg(kernel, 29, sizeof(cl_mem), (void *)&par_max_coo);
    err = clSetKernelArg(kernel, 30, sizeof(cl_mem), (void *)&in_final_sum);
    err = clSetKernelArg(kernel, 31, sizeof(cl_mem), (void *)&in_sqr_final_sum);
    err = clSetKernelArg(kernel, 32, sizeof(cl_mem), (void *)&denomT);
    err = clSetKernelArg(kernel, 33, sizeof(cl_mem), (void *)&d_checksum);

    size_t local_work_size[1];
    local_work_size[0] = 256;
    size_t global_work_size[1];
    global_work_size[0] = 256*51;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
    err = clFinish(queue);
    err = clReleaseKernel(kernel);
    err = clReleaseProgram(program);
    err = clReleaseCommandQueue(queue);
    err = clReleaseContext(context);
}
