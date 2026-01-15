/* Implementation of work-group scheduler used with fiber method.

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

//#include <stdio.h>

/* Data structure that contains work-group information.
   Passed in from the kernel. */
typedef struct
{
  uint64_t local_size_x;
  uint64_t local_size_y;
  uint64_t local_size_z;
  uint64_t subgroup_size;
  /* Number of subgroups in workgroup */
  uint64_t n_subgroups;
  /* Number of workitems waiting at barrier */
  uint64_t waiting_count;
  /* Number of subgroup barriers that are active. Active subgroup barrier
     means one of more workitems in a subgroup has reached subgroup barrier. */
  uint64_t sg_barriers_active;
  /* Workitem counter for each subgroup. Counts the number of workitems that
     have reached a barrier (workgroup or subgroup barrier). */
  uint64_t *sg_wi_counter;
  /* Subgroup barrier counter for each subgroup. This holds the information
     whether subgroup is at sugbroup barrier. Can only have values 0 and 1. */
  uint64_t *sg_barrier_counter;
} wgState;

void __pocl_fiber_sched_init (wgState *wg_state,
                              uint64_t *sg_wi_counter,
                              uint64_t *sg_barrier_counter);

uint64_t __pocl_fiber_schedule_work_item (wgState *wg_state);

void __pocl_fiber_wg_barrier_eached (uint64_t local_id_x,
                                      uint64_t local_id_y,
                                      uint64_t local_id_z,
                                      wgState *wg_state);

void __pocl_fiber_sg_barrier_reached (uint64_t local_id_x,
                                      uint64_t local_id_y,
                                      uint64_t local_id_z,
                                      wgState *wg_state);

static void resolve_barriers (wgState *wg_state);

static void print_barrier_status (wgState *wg_state);


/* Enable debug prints from the Fiber scheduler. */
/* #define DEBUG_FIBER_SCHEDULER */

/*
 * Scheduler for the fiber work-group method.
 * Main functionalities are:
 *  (1) Decide the next work-item to be scheduled.
 *  (2) Resolve the barriers when appropriate.
 *
 * Barriers are tracked on a subgroup basis to support all kinds of subgroup
 * configurations and barrier-usage corner-cases in general.
 * There are two counters for each subgroup (in the data structure):
 *  1. sg_wi_counter[i] counts the number of work-items that have reached
 *     sg/wg barrier in subgroup i.
 *  2. sg_barrier_counter is not really a counter as it can only have
 *     values 0/1.
 *
 *  sg_barrier_counter[j] indicates whether subgroup j has an 'active'
 *  subgroup barrier.
 *
 * Subgroup barriers can be resolved by tracking the combination of
 * sg_wi_counter[i] and sg_barrier_counter[i].
 *
 */

/** Initializes the work-group state.
 *
 * Called by one work-item from the kernel.
 *
 * @param wg_state The work-group data structure.
 * @param sg_wi_counter Pointer to subgroup counters that store wi progression.
 * @param sg_barrier_counter Pointer to counters that store subgroup barrier
 *        status.
 */
void
__pocl_fiber_sched_init (wgState *wg_state,
                         uint64_t *sg_wi_counter,
                         uint64_t *sg_barrier_counter)
{
  wg_state->n_subgroups = (wg_state->local_size_x * wg_state->local_size_y
                           * wg_state->local_size_z)
                          / wg_state->subgroup_size;

  wg_state->waiting_count = 0;
  wg_state->sg_barriers_active = 0;

  /* Counter arrays are allocated in the kernel side. */
  wg_state->sg_wi_counter = sg_wi_counter;
  wg_state->sg_barrier_counter = sg_barrier_counter;

#ifdef DEBUG_FIBER_SCHEDULER

  printf ("fiber scheduler init: local size: (%d, %d, %d)\n",
          wg_state->local_size_x, wg_state->local_size_y,
          wg_state->local_size_z);
  printf ("sg-size: %d, n_subgroups: %d\n\n", wg_state->subgroup_size,
          wg_state->n_subgroups);

#endif
}

/** Provides the linear ID of the workitem that will be scheduled next.
 *
 * Scheduling works so that lowest-id subgroup that still have workitems
 * not reached a barrier has priority.
 *
 * @param wg_state The work-group data structure.
 * @return The linear ID of the workitem within the work-group.
 */
uint64_t
__pocl_fiber_schedule_work_item (wgState *wg_state)
{
  /* First check if barriers can be resolved. */
  resolve_barriers (wg_state);

  uint64_t next_wi = 0;

  /* Go through all subgroups */
  for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
    {

      /* Choose the first sub-group that has work-items which have not
         yet reached a barrier. */
      if (wg_state->sg_wi_counter[i] < wg_state->subgroup_size)
        {
          /* Adjust the id so that return value will be linearized id within
             the work-group. */
          next_wi = i * wg_state->subgroup_size + wg_state->sg_wi_counter[i];
          break;
        }
    }

#ifdef DEBUG_FIBER_SCHEDULER
  printf ("Next WI: %ld\n\n", next_wi);
#endif

  return next_wi;
}

/** Registers workitem reaching a work-group barrier.
 *
 * Workitem progression is stored based on linear ids, so this
 * requires linear conversion as the kernel works with 3-dimensional ids.
 *
 * @param local_id_x Local 'x' id of current workitem.
 * @param local_id_y Local 'y' id of current workitem.
 * @param local_id_z Local 'z' id of current workitem.
 * @param wg_state The work-group data structure.
 */
void
__pocl_fiber_wg_barrier_reached (uint64_t local_id_x,
                                 uint64_t local_id_y,
                                 uint64_t local_id_z,
                                 wgState *wg_state)
{
  /* Need to linearize the ID as counters are stored in that way. */
  uint64_t linearId
    = ((local_id_z * wg_state->local_size_y * wg_state->local_size_x)
       + (local_id_y * wg_state->local_size_x) + local_id_x);

  /* Subgroup id for incrementing the counter of the desired subgroup. */
  uint64_t sg_id = linearId / wg_state->subgroup_size;

  wg_state->waiting_count++;
  wg_state->sg_wi_counter[sg_id]++;

#ifdef DEBUG_FIBER_SCHEDULER
  uint64_t sg_local_id = linearId % wg_state->subgroup_size;
  printf ("wg-barrier reached\n");
  printf ("LinearID: %d\tlocal_id_x: %d\tlocal_id_y: %d\tlocal_id_z: %d\t"
          "sg_id: %d\t sg_local_id: %d\n",
          linearId, local_id_x, local_id_y, local_id_z, sg_id, sg_local_id);
  print_barrier_status (wg_state);
#endif
}

/** Registers a workitem reaching a sub-group barrier.
 *
 * Workitem progression is stored based on linear ids, so this
 * requires linear conversion as the kernel works with 3-dimensional ids.
 *
 * @param local_id_x Local 'x' id of current workitem.
 * @param local_id_y Local 'y' id of current workitem.
 * @param local_id_z Local 'z' id of current workitem.
 * @param wg_state The work-group data structure.
 */
void
__pocl_fiber_sg_barrier_reached (uint64_t local_id_x,
                                 uint64_t local_id_y,
                                 uint64_t local_id_z,
                                 wgState *wg_state)
{
  /* Linearize wg id */
  uint64_t linearId
    = ((local_id_z * wg_state->local_size_y * wg_state->local_size_x)
       + (local_id_y * wg_state->local_size_x) + local_id_x);

  /* Calculate subgroup related IDs. */
  uint64_t sg_id = linearId / wg_state->subgroup_size;
  uint64_t sg_local_id = linearId % wg_state->subgroup_size;

  /* Only increase the sg barrier counter when the first wi of
     the subgroup comes in. */
  if (wg_state->sg_wi_counter[sg_id] == 0)
    {
      wg_state->sg_barriers_active++;
      wg_state->sg_barrier_counter[sg_id]++;
    }

  wg_state->waiting_count++;
  wg_state->sg_wi_counter[sg_id]++;

#ifdef DEBUG_FIBER_SCHEDULER
  printf ("sg-barrier reached\n");
  printf ("sg_id: %d\t sg_local_id: %d\n", sg_id, sg_local_id);
  print_barrier_status (wg_state);

#endif
}

/** Resolves work-group/subgroup barriers.
 *
 * Check whether conditions are fulfilled such that all workitems have reached
 * a barrier and can be 'freed' to move on.
 *
 * Case 1. If all WIs in the work-group are waiting (indicated by
 * waiting_count), zero the counters.
 *
 * Case 2. If the subgroup barrier is 'active' (indicated by
 *         sg_barrier_counter[i]) for any subgroup, and the workitem counter
 *         (indicated by sg_wi_counter[i]) within that subgroup equals the
 *         subgroup size, zero counters for that specific subgroup.
 *
 * @param wg_state The work-group data structure.
 *
 */
static void
resolve_barriers (wgState *wg_state)
{

#ifdef DEBUG_FIBER_SCHEDULER
  printf ("Barrier status before resolving barriers\n");
  printf ("waiting_count: %d\tsg_barriers_active: %d\t",
          wg_state->waiting_count, wg_state->sg_barriers_active);

  for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
    {
      printf ("sg_wi_counter[%d]: %d\t", i, wg_state->sg_wi_counter[i]);
    }
  printf ("\n");
  for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
    {
      printf ("sg_barrier_status[%d]: %d\t", i,
              wg_state->sg_barrier_counter[i]);
    }
  printf ("\n\n");
#endif

  /* Case 1: No subgroup-barriers are 'active'. */
  if (!wg_state->sg_barriers_active)
    {

      /* Just check if all work-items are waiting (reached a wg-barrier). */
      if (wg_state->waiting_count
          == (wg_state->subgroup_size * wg_state->n_subgroups))
        {

          /* Zero counters and waiting count. */
          for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
            {
              wg_state->sg_wi_counter[i] = 0;
            }
          wg_state->waiting_count = 0;
        }

      /* Case 2: There are subgroup-barriers 'active'. */
    }
  else
    {

      /* Have to check the status of each subgroup */
      for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
        {

          /* Subgroup barrier is 'active' for subgroup i and all work-items of
             corresponding subgroup have reached it. */
          if (wg_state->sg_barrier_counter[i] == 1
              && (wg_state->sg_wi_counter[i] == wg_state->subgroup_size))
            {

              wg_state->sg_wi_counter[i] = 0;

              wg_state->sg_barrier_counter[i] = 0;

              /* The waiting_count is still global for WG so just subtract. */
              wg_state->waiting_count
                = wg_state->waiting_count - wg_state->subgroup_size;

              wg_state->sg_barriers_active--;
            }
        }
    }

#ifdef DEBUG_FIBER_SCHEDULER
  printf ("Barrier status after resolving barriers\n");
  printf ("waiting_count: %d\tsg_barriers_active: %d\t",
          wg_state->waiting_count, wg_state->sg_barriers_active);

  for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
    {
      printf ("sg_wi_counter[%d]: %d\t", i, wg_state->sg_wi_counter[i]);
    }
  printf ("\n");
  for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
    {
      printf ("sg_barrier_status[%d]: %d\t", i,
              wg_state->sg_barrier_counter[i]);
    }
  printf ("\n\n");
#endif
}

#ifdef DEBUG_FIBER_SCHEDULER
/** Prints debug information about barrier statuses.
 *
 * For each subgroup, outputs the number of workitems waiting.
 *
 * @param wg_state The work-group data structure.
 */
static void
print_barrier_status (wgState *wg_state)
{
  for (uint64_t i = 0; i < wg_state->n_subgroups; i++)
    {

      printf (" %d ", wg_state->sg_wi_counter[i]);
    }
  printf ("\n\n");
}
#endif
