#ifndef ARITHMETIC_H
#define ARITHMETIC_H

template <int x, int y> constexpr __device__ __host__ inline int exact_div() {
  static_assert(x % y == 0);
  static_assert(x >= y);
  return x / y;
}

constexpr __device__ __host__ inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

#endif
