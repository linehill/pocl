/* Workgroup function generation test case for outer loop parallelization
   of a loop with multiple breaks.

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
  for (int i = 0; i < 2048; ++i) {
    /* This should be detected as an uniform loop automatically and converted to
       a b-loop, meaning a's for all WIs should be executed first for each
       iteration. */
    printf ("a: i == %d lid == %d\n", i, get_local_id(0));
    /* A dummy never-taken break which cannot be optimized away. */
    if (d[0] == 1)
      break;
    printf ("b: i == %d lid == %d\n", i, get_local_id(0));
  }
  /* Just to make the output deterministic. */
  d[get_local_id(0)] = d[get_local_id(0)] * 2;
}
