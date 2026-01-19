kernel void test_vectorization(__global const float * restrict a, __global float * restrict b, __global int * restrict c) {
  size_t i = get_global_id(0);
  float f = 0;
  //============================================================================
  // Vectorized via vector-function-abi-variant
  //============================================================================
  f += atan2(a[i], a[i]);
  f += cbrt(a[i]);
  f += erfc(a[i]);
  f += erf(a[i]);
  f += expm1(a[i]);
  f += lgamma(a[i]);
  // The pointer parameter doesn't yet vectorize cleanly due to LLVM
  // limitations as of LLVM 21.
  // f += lgamma_r(a[i], c+i);
  f += native_powr(a[i], a[i]);
  f += powr(a[i], a[i]);
  f += remainder(a[i], a[i]);
  // f += remquo(a[i], a[i], c+i);
  f += rootn(a[i], a[i]);
  f += tgamma(a[i]);
  f += pown(a[i], i);
  //f += frexp(a[i], c+i);
  f += ldexp(a[i], i);

  //============================================================================
  // Vectorized via LLVM veclib
  //============================================================================

  f += acos(a[i]);
  f += asin(a[i]);
  f += atan(a[i]);
  f += ceil(a[i]);
  f += copysign(a[i], (float)(c[i]));
  f += cos(a[i]);
  f += exp(a[i]);
  f += exp2(a[i]);
  f += exp10(a[i]);
  f += fabs(a[i]);
  f += floor(a[i]);
  f += fma(a[i], b[i], (float)(c[i]));
  f += fmax(a[i], b[i]);
  f += fmin(a[i], b[i]);
  f += log(a[i]);
  f += log2(a[i]);
  f += log10(a[i]);
  f += pow(a[i], a[i]);
  f += rint(a[i]);
  f += round(a[i]);
  f += rsqrt(a[i]);
  f += sin(a[i]);
  f += sqrt(a[i]);
  f += tan(a[i]);
  f += trunc(a[i]);

  //============================================================================
  // These do not lower as function calls currently
  //============================================================================
  // f += fmod(a[i], b[i]);
  // f += mad(a[i], b[i], (float)(c[i]));
  // f += maxmag(a[i], b[i]);
  // f += minmag(a[i], b[i]);

  b[i] = f;
}
