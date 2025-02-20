/* Workgroup function generation test case for intra-kernel reductions.

   Copyright (c) 2025 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

__kernel void
test_kernel (global int *int_d, global int *out_d)
{
  float intra_kernel_sum_f = 0.0f;

  /* First convert the values to float. */
  global float *float_d = (global float *)int_d;
  float_d[get_global_id (0)] = (float)int_d[get_global_id (0)];

  barrier (CLK_GLOBAL_MEM_FENCE);

  if (get_local_id (0) == 0)
    {
      for (int i = 0; i < get_local_size (0); ++i)
        intra_kernel_sum_f
          += float_d[get_group_id (0) * get_local_size (0) + i];
    }

  barrier (CLK_GLOBAL_MEM_FENCE);

  if (get_local_id (0) == 0)
    {
      out_d[get_group_id (0)] = (int)intra_kernel_sum_f;
      /* Just to match the reduce.cl output. */
      out_d[get_num_groups (0) + get_group_id (0)] = (int)intra_kernel_sum_f;
    }
#if 0
  /* This is here to trigger the inlining related problem with relaxed math
     flags.  When we force inline the function, its attributes will be merged
     with the callee. If the attributes are not set to the relaxed ones before,
     we get the "play it safe" semantics. */
  float sum_f = work_group_reduce_add (float_d[get_global_id (0)]);
  if (get_local_id (0) == 0 && sum_f == intra_kernel_sum_f)
    {
      out_d[get_group_id (0)] = (int)intra_kernel_sum_f;
    }
#endif
}
