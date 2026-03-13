kernel void test_vectorization(__global const float *restrict a,
                               __global float *restrict b,
                               __global int *restrict c) {
  size_t i = get_global_id(0);
  float f = 0;
  //============================================================================
  // Vectorized via vector-function-abi-variant
  //============================================================================
  f += atan2(a[i], a[i]);
  f += erfc(a[i]);
  f += erf(a[i]);
  f += expm1(a[i]);

  f += lgamma(a[i]);
  f += remainder(a[i], a[i]);
  f += rootn(a[i], a[i]);
  f += tgamma(a[i]);
  f += pown(a[i], i);

  b[i] = f;
}
