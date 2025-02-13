/* Header file for the fiber scheduler.

   Copyright (c) 2025 Tapio Nevalainen / Tampere University

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

/* Data structure that contains work-group information.
   Passed in from the kernel. */
typedef struct
{
  unsigned long local_size_x;
  unsigned long local_size_y;
  unsigned long local_size_z;
  unsigned long subgroup_size;
  /* Number of subgroups in workgroup */
  unsigned long n_subgroups;
  /* Number of workitems waiting at barrier */
  unsigned long waiting_count;
  /* Number of subgroup barriers that are active. Active subgroup barrier
     means one of more workitems in a subgroup has reached subgroup barrier. */
  unsigned long sg_barriers_active;
  /* Workitem counter for each subgroup. Counts the number of workitems that
     have reached a barrier (workgroup or subgroup barrier). */
  unsigned long *sg_wi_counter;
  /* Subgroup barrier counter for each subgroup. This holds the information
     whether subgroup is at sugbroup barrier. Can only have values 0 and 1. */
  unsigned long *sg_barrier_counter;
} wgState;

void __pocl_fiber_sched_init (wgState *wg_state,
                              unsigned long *sg_wi_counter,
                              unsigned long *sg_barrier_counter);

long __pocl_fiber_schedule_work_item (wgState *wg_state);

void __pocl_fiber_wg_barrier_reached (long local_id_x,
                                      long local_id_y,
                                      long local_id_z,
                                      wgState *wg_state);

void __pocl_fiber_sg_barrier_reached (long local_id_x,
                                      long local_id_y,
                                      long local_id_z,
                                      wgState *wg_state);

static void resolve_barriers (wgState *wg_state);

static void print_barrier_status (wgState *wg_state);