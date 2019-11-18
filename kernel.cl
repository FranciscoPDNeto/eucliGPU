void __kernel euclidean(__global const unsigned char *A, __global float *B) {
  unsigned long taskIndex = get_global_id(0);
  B[taskIndex] = A[taskIndex] + A[taskIndex];
}