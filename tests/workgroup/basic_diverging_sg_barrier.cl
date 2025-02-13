/* Test case for diverging subgroups. Work group method should allow
   synchronization on subgroup basis with subgroup barrier. 

   Copyright (c) 2025 Taio Nevalainen / Tampere University

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

#define SG_SIZE 2

__attribute ((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void
test_kernel (void)
{
  int gid_x = get_global_id (0);
  int gid_y = get_global_id (1);
  int gid_z = get_global_id (2);

  int sg_id = get_sub_group_id ();
  int sg_local_id = get_sub_group_local_id ();
  
  if (sg_id == 1) {
    printf ("WI:(%d %d %d) SG:[%d %d] - before subgroup barrier\n",
            gid_x, gid_y, gid_z, sg_id, sg_local_id);

    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    printf ("WI:(%d %d %d) SG:[%d %d] - after subgroup barrier\n",
            gid_x, gid_y, gid_z, sg_id, sg_local_id);
  } else {
    printf ("WI:(%d %d %d) SG:[%d %d] - avoided sg barrier\n",
            gid_x, gid_y, gid_z, sg_id, sg_local_id);
  }
  
}
