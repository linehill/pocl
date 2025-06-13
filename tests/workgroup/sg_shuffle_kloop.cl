/* Workgroup function generation test case for sub-group shuffles.

   Copyright (c) 2025 John Pennycook / Intel
                      Pekka Jääskeläinen / Intel Finland Oy

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

#define SUB_GROUP_SIZE 16
#define half_sub_group_size (SUB_GROUP_SIZE / 2)

// This snippet is similar to one in a real HPC application although it looks
// weird. Each lane in one half of the sub-group is paired with every lane in
// the other half. The multiplication is used as a blend/select.
__attribute__ ((intel_reqd_sub_group_size (SUB_GROUP_SIZE))) __kernel void
test_kernel (global int *in_d, global int *out_d)
{
  // Load a unique input for each work-item.
  int x = in_d[get_global_id (0)];

  int sum = 0;
  int lid = get_sub_group_local_id ();
  int src = 0;

  for (int i = 0; i < half_sub_group_size; i++)
    {
      int src
        = (half_sub_group_size + ((lid + i) & (half_sub_group_size - 1)))
            * (lid < half_sub_group_size)
          + (((lid & (half_sub_group_size - 1)) + half_sub_group_size - i)
             & (half_sub_group_size - 1))
              * (lid >= half_sub_group_size);
      sum += sub_group_shuffle (x, src);
    }

  out_d[get_global_id (0)] = sum;
}
