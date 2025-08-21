#define SUB_GROUP_SIZE 16

__attribute__ ((intel_reqd_sub_group_size (SUB_GROUP_SIZE))) __kernel void
test_kernel (void)
{

  int local_id_x = get_local_id (0);
  int local_id_y = get_local_id (1);
  int local_id_z = get_local_id (2);
  int sg_id = get_sub_group_id ();
  int sg_size = get_sub_group_size ();

  if (local_id_x == 0 && local_id_y == 0)
    {
      printf ("subgroup size: %d\n", sg_size);
    }

  if (sg_id == 0)
    {
      printf ("subgroup 0 B: local-id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
      // a = a + 4;
      sub_group_barrier (CLK_LOCAL_MEM_FENCE);
      printf ("subgroup 0 A: local-id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
    }
  else if (sg_id == 1)
    {
      printf ("subgroup 1 B: local-id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
      sub_group_barrier (CLK_LOCAL_MEM_FENCE);
      printf ("subgroup 1 A: local-id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
    }
  else if (sg_id == 2)
    {
      printf ("subgroup 2 B: local-id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
      sub_group_barrier (CLK_LOCAL_MEM_FENCE);
      printf ("subgroup 2 A: local_id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
    }
  else
    {
      printf ("subgroup x B: local_id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
      sub_group_barrier (CLK_LOCAL_MEM_FENCE);
      printf ("subgroup x A: local_id: (%d, %d), sg-id: %d\n", local_id_x,
              local_id_y, sg_id);
    }

  printf ("local_id: (%d, %d), sg-id: %d\n", local_id_x, local_id_y, sg_id);
}