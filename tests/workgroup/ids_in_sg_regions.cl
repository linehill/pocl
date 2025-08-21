#define SUB_GROUP_SIZE 8

__attribute__ ((intel_reqd_sub_group_size (SUB_GROUP_SIZE))) __kernel void
test_kernel (void)
{
  // Get global and local IDs
  int global_id_x = get_global_id (0);
  int global_id_y = get_global_id (1);
  int global_id_z = get_global_id (2);
  int local_id_x = get_local_id (0);
  int local_id_y = get_local_id (1);
  int local_id_z = get_local_id (2);
  int sg_id = get_sub_group_id ();
  int wg_id_x = get_group_id (0);
  int wg_id_y = get_group_id (1);
  int wg_id_z = get_group_id (2);
  int sg_local_id = get_sub_group_local_id ();

  sub_group_barrier (CLK_LOCAL_MEM_FENCE);

  printf ("Group:(%d, %d, %d), Global:(%d, %d, %d), Local:(%d, %d, %d), SG: "
          "(%d, %d)\n",
          wg_id_x, wg_id_y, wg_id_z, global_id_x, global_id_y, global_id_z,
          local_id_x, local_id_y, local_id_z, sg_id, sg_local_id);

  sub_group_barrier (CLK_LOCAL_MEM_FENCE);
}