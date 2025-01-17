/* Workgroup function generation test case for when outer loop parallelization
   should not be performed for better vectorization opportunities over the
   inner (kernel) loop.

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
test_kernel (global int *d)
{
  /* Perform partial sums. The structure might be vectorizable as a reduction
     pattern by LLVM loopvec if we don't push the WI-loops inside the inner
     loop to flip the mem access pattern stepping to 4. */
  int sum = 0;
  for (int i = 0; i < 4; ++i) {
    printf ("i: %d gid: %d\n", i, get_global_id(0));
    /* The load memory access pattern is stride 1 over i, stride 4 over
       WI_X. */
    sum += d[get_local_id(0) * 4 + i];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  d[get_local_id(0)] = sum;
}
