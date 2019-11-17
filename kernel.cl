void kernel euclidean(global const unsigned char *A, global unsigned char *B) {
  unsigned long taskIndex = get_global_id(0);
  B[taskIndex] = A[taskIndex] + A[taskIndex];
}