#define SHUFFLE_AMOUNT 2

#if defined(cl_khr_subgroups)
__kernel void test_kernel ()
{

  int res = get_local_linear_id ();
  res = sub_group_shuffle_up (res, SHUFFLE_AMOUNT);
  printf ("%d\n", res);
}

#else
#error this test requires cl_khr_subgroups extension
#endif
