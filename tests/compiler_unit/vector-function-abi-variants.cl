kernel void test_vector_variant(__global const float *a, __global float *b,
                                __global int *c) {
  size_t i = get_global_id(0);
  float f = 0;
  f += atan2(a[i], a[i]);
  f += cbrt(a[i]);
  f += erfc(a[i]);
  f += erf(a[i]);
  f += expm1(a[i]);
  f += lgamma(a[i]);
  // The pointer parameter doesn't yet vectorize cleanly.
  // f += lgamma_r(a[i], c+i);
  f += native_powr(a[i], a[i]);
  f += powr(a[i], a[i]);
  f += remainder(a[i], a[i]);
  // f += remquo(a[i], a[i], c+i);
  f += rootn(a[i], a[i]);
  f += tgamma(a[i]);
  b[i] = f;
}
