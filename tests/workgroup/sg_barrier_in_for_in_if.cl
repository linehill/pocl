#define SG_SIZE 2
#define N_LOOP_ITER 4

__attribute__ ((intel_reqd_sub_group_size (SG_SIZE)))
__kernel void
test_kernel ()
{

  int global_id = get_global_id (0);
  int local_id = get_local_id (0);
  int sg_id = get_sub_group_id ();
  int wg_id = get_group_id (0);
  int sg_local_id = get_sub_group_local_id ();

  if (sg_id == 0)
    {

      for (int i = 0; i < N_LOOP_ITER; i++)
        {
          printf ("Before sub-group barrier (i=%d): Global ID: %d\tLocal ID:"
                  " %d\tWorkgroup ID: %d\tSubgroup ID: %d\tSubgroup local ID:"
                  " %d\n",
                  i, global_id, local_id, wg_id, sg_id, sg_local_id);

          sub_group_barrier (CLK_LOCAL_MEM_FENCE);

          printf ("After sub-group barrier (i=%d): Global ID: %d\tLocal ID:"
                  " %d\tWorkgroup ID: %d\tSubgroup ID: %d\tSubgroup local ID:"
                  " %d\n",
                  i, global_id, local_id, wg_id, sg_id, sg_local_id);
        }
    }
  printf ("Global ID: %d\tLocal ID: %d\tWorkgroup ID: %d\tSubgroup ID:"
          " %d\tSubgroup local ID: %d\n",
          global_id, local_id, wg_id, sg_id, sg_local_id);
}
