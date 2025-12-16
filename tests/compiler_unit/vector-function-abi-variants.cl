kernel void test_vector_variant(__global const float* a, __global float* b)
{
  size_t i = get_global_id(0);
  b[i] = erf(a[i]);
}
