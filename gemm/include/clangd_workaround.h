#define HOST_CODE_BELOW                                                                                                \
  extern "C" int printf(const char *fmt, ...);                                                                         \
  extern void operator delete[](void *ptr) _GLIBCXX_USE_NOEXCEPT;                                                      \
  extern void *operator new[](__SIZE_TYPE__ size);

#define DEVICE_CODE_BELOW                                                                                                \
  extern "C" __device__ int printf(const char *fmt, ...);                                                                         \
  extern __device__ void operator delete[](void *ptr) _GLIBCXX_USE_NOEXCEPT;                                                      \
  extern __device__ void *operator new[](__SIZE_TYPE__ size);
