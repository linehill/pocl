/* Built-ins for internal testing purposes.

   Copyright (c) 2026 Henry Linjamäki / Tampere University

   Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without
   restriction, including without limitation the rights to use, copy,
   modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/
/* This file hosts built-ins for internal testing. Beware, these
   built-ins are second class citizens and their behavior are not
   guaranteed or well-defined and thus they should be relied on in
   production grade applications.  */

extern size_t _wiloop_lower_bound_x;
extern size_t _wiloop_upper_bound_x;
extern size_t _wiloop_lower_bound_y;
extern size_t _wiloop_upper_bound_y;
extern size_t _wiloop_lower_bound_z;
extern size_t _wiloop_upper_bound_z;

/** Set work-item loop bounds to [lower_bound, upper_bound) for the
 * 'dim' dimension.
 *
 * The kernel behavior is undefined if (non-exhaustive list):
 *
 * - the range exceeds work-group-size at the given dimension.
 *
 * - the arguments are not dynamically uniform.
 *
 * Only effective for CPU devices using loopvec work-group method.
 */
void
__pocl_probe_set_wiloop_bounds (unsigned int dim,
                                size_t lower_bound,
                                size_t upper_bound)
{
  switch (dim)
    {
    default:
      break;
    case 0:
      _wiloop_lower_bound_x = lower_bound;
      _wiloop_upper_bound_x = upper_bound;
      break;
    case 1:
      _wiloop_lower_bound_y = lower_bound;
      _wiloop_upper_bound_y = upper_bound;
      break;
    case 2:
      _wiloop_lower_bound_z = lower_bound;
      _wiloop_upper_bound_z = upper_bound;
      break;
    }
}
