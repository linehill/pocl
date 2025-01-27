// Implementation of work-group scheduler used with fiber method.
//
// Copyright (c) 2025 Tapio Nevalainen / Tampere University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <stdio.h>
#include "fiber_scheduler.h"

// Scheduler for the fiber work-group method.
// Main functionalities are:
// (1) Decide next work-item to be scheduled.
// (2) Resolve barriers when appropriate.
//
// Barriers are tracked subgroup-wise to enable subgroup barrier semantics.
// There are two counters for each subgroup (in the data structure):
// 1. sg_wi_counter[i] counts the number of work-items that have reached 
//    sg/wg barrier in subgroup i.
// 2. sg_barrier_counter is not really a counter as it can only have values 0/1.
//    sg_barrier_counter[j] indicates whether subgroup j has 'active' subgroup
//    barrier.
//
// Subgroup barriers can be resolved by tracking the combination of 
// sg_wi_counter[i] and sg_barrier_counter[i].


// #define DEBUG_FIBER_SCHEDULER

// Called by one work-item from the kernel.
// Initializes values in the data structure.
void
__pocl_fiber_sched_init(wgState *wgState, unsigned long *sg_wi_counter,
                  unsigned long *sg_barrier_counter)
{  
  wgState->n_subgroups = (wgState->local_size_x*wgState->local_size_y*wgState->local_size_z)/wgState->subgroup_size;
  wgState->waiting_count = 0;
  wgState->sg_barriers_active = 0;

  // Counter arrays are allocated in the kernel side.
  wgState->sg_wi_counter = sg_wi_counter;
  wgState->sg_barrier_counter = sg_barrier_counter;

#ifdef DEBUG_FIBER_SCHEDULER
 
  printf("fiber scheduler init: local size: (%d, %d, %d)\n",
         wgState->local_size_x, wgState->local_size_y, wgState->local_size_z);
  printf("sg-size: %d, n_subgroups: %d\n\n", wgState->subgroup_size,
         wgState->n_subgroups);

#endif
}

// Returns the local linear ID of the work-item that will be scheduled next.
long
__pocl_fiber_schedule_work_item(wgState *wgState)
{
  // First check if barriers can be resolved.
  resolve_barriers(wgState);

  long next_wi = 0;

  // Go through all subgroups
  for (int i = 0; i< wgState->n_subgroups; i++) {
        
    // Choose the first sub-group that has work-items which have not 
    // yet reached a barrier. 
    if (wgState->sg_wi_counter[i] < wgState->subgroup_size) {
      // Adjust the id so that return value will be linearized id within
      // the workgroup.
      next_wi = i*wgState->subgroup_size + wgState->sg_wi_counter[i];
        break;
    }
  }

#ifdef DEBUG_FIBER_SCHEDULER
  printf("Next WI: %ld\n\n",next_wi);
#endif

  return next_wi;
}


// Register work-item reaching a work-group barrier.
void
__pocl_fiber_wg_barrier_reached(long local_id_x, long local_id_y,
                                long local_id_z, wgState *wgState)
{
    // Need to linearize the ID as counters are stored in that way.
    unsigned int linearId =
        ((local_id_z * wgState->local_size_y * wgState->local_size_x)
        + (local_id_y * wgState->local_size_x) 
        + local_id_x);

    // Calculate subgroup related IDs.
    unsigned int sg_id = linearId / wgState->subgroup_size;
    unsigned int sg_local_id = linearId % wgState->subgroup_size;


    wgState->waiting_count++;
    wgState->sg_wi_counter[sg_id]++;

#ifdef DEBUG_FIBER_SCHEDULER
    printf("wg-barrier reached\n");
    printf("LinearID: %d\tlocal_id_x: %d\tlocal_id_y: %d\tlocal_id_z: %d\t"
           "sg_id: %d\t sg_local_id: %d\n" ,linearId, local_id_x, local_id_y,
           local_id_z,sg_id, sg_local_id);
    print_barrier_status(wgState);
#endif
}

// Register work-item reaching a sub-group barrier.
void
__pocl_fiber_sg_barrier_reached(long local_id_x, long local_id_y,
                                long local_id_z, wgState *wgState)
{
    // Linearize wg id
    unsigned int linearId =
        ((local_id_z * wgState->local_size_y * wgState->local_size_x)
        + (local_id_y * wgState->local_size_x)
        + local_id_x);

    // Calculate subgroup related IDs.
    int sg_id = linearId / wgState->subgroup_size;
    int sg_local_id = linearId % wgState->subgroup_size;

    // Only increase the sg barrier counter when the first wi of
    // the subgroup comes in.
    if (wgState->sg_wi_counter[sg_id] == 0) {
        wgState->sg_barriers_active++;
        wgState->sg_barrier_counter[sg_id]++;
    }
    
    wgState->waiting_count++;
    wgState->sg_wi_counter[sg_id]++;
    
#ifdef DEBUG_FIBER_SCHEDULER
    printf("sg-barrier reached\n");
    printf("sg_id: %d\t sg_local_id: %d\n",sg_id, sg_local_id);
    print_barrier_status(wgState);
    
#endif

}

// Resolve the barriers (workgroup/subgroup). 
// Zero the barrier counters depending on the situation. 
static void
resolve_barriers(wgState *wgState)
{   

#ifdef DEBUG_FIBER_SCHEDULER
  printf("Barrier status before resolving barriers\n");
  printf("waiting_count: %d\tsg_barriers_active: %d\t",
         wgState->waiting_count,wgState->sg_barriers_active);

  for(int i = 0; i<wgState->n_subgroups; i++){
    printf("sg_wi_counter[%d]: %d\t",i,wgState->sg_wi_counter[i]);
  }
  printf("\n");
  for(int i = 0; i<wgState->n_subgroups; i++){
    printf("sg_barrier_status[%d]: %d\t",i,wgState->sg_barrier_counter[i]);
  }
  printf("\n\n");
#endif

  // Case 0: No subgroup-barriers are 'active'.
  if (!wgState->sg_barriers_active) {
        
    // Just check if all work-items are waiting (reached a wg-barrier).
    if (wgState->waiting_count
        == (wgState->subgroup_size * wgState->n_subgroups)) {
      
      // Zero counters and waiting count.      
      for (int i = 0; i<wgState->n_subgroups; i++) {
        wgState->sg_wi_counter[i] = 0;
      }
      wgState->waiting_count = 0;
    }
    
  
  // Case 1: There are subgroup-barriers 'active'.
  }else{

    // Have to check the status of each subgroup
    for (int i = 0; i< wgState->n_subgroups; i++) {


      // Subgroup barrier is 'active' for subgroup i and all work-items of
      // corresponding subgroup have reached it.
      if (wgState->sg_barrier_counter[i] == 1 &&
          (wgState->sg_wi_counter[i] == wgState->subgroup_size)) {

        wgState->sg_wi_counter[i] = 0;

        wgState->sg_barrier_counter[i] = 0;

        // The waiting_count is still global for WG so just subtract.
        wgState->waiting_count =
            wgState->waiting_count - wgState->subgroup_size;

        wgState->sg_barriers_active--;
      }
    }
  }

#ifdef DEBUG_FIBER_SCHEDULER
  printf("Barrier status after resolving barriers\n");
  printf("waiting_count: %d\tsg_barriers_active: %d\t",
         wgState->waiting_count,wgState->sg_barriers_active);

  for(int i = 0; i<wgState->n_subgroups; i++){
    printf("sg_wi_counter[%d]: %d\t",i,wgState->sg_wi_counter[i]);
  }
  printf("\n");
  for(int i = 0; i<wgState->n_subgroups; i++){
    printf("sg_barrier_status[%d]: %d\t",i,wgState->sg_barrier_counter[i]);
  }
  printf("\n\n");
#endif

}

#ifdef DEBUG_FIBER_SCHEDULER
// Outputs the status of barriers.
static void 
print_barrier_status(wgState *wgState)
{

    for (int i = 0; i < wgState->n_subgroups; i++) {
        
        printf(" %d ", wgState->sg_wi_counter[i]);
    }
    printf("\n\n");
}
#endif
