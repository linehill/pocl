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
  int sum_i = work_group_reduce_add (int_d[get_global_id (0)]);

  /*
    With 512 WIs per WG with 2 WGs the correct answers are:
    0: 131840
    1: 393984
  */

  /*
    Float reduction has the additional challenge with float ordering
    constraints when strict rounding is used which can prevent
    parallel reduction trees to be formed.

    First convert the values to float.
  */
  global float *float_d = (global float *)int_d;
  float_d[get_global_id (0)] = (float)int_d[get_global_id (0)];

  barrier (CLK_GLOBAL_MEM_FENCE);

  float sum_f = work_group_reduce_add (float_d[get_global_id (0)]);

  int_d[get_global_id (0)] = get_global_id (0);

  barrier (CLK_GLOBAL_MEM_FENCE);

  if (get_local_id (0) == 0)
    {
      out_d[get_group_id (0)] = sum_i;
      out_d[get_num_groups (0) + get_group_id (0)] = (int)sum_f;
    }
}
