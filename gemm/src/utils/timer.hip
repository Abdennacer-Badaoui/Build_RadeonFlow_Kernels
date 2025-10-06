#include "../../include/timer.h"

// Define HIPRT_CB if not already defined
#ifndef HIPRT_CB
#define HIPRT_CB
#endif

// Forward declaration of KernelTimer
class KernelTimer;

// Static callback function for hipStreamAddCallback
static void HIPRT_CB eventCallback(hipStream_t stream, hipError_t status, void* userData) {
  if (status != hipSuccess) return;
  
  KernelTimer* timer = static_cast<KernelTimer*>(userData);
  float elapsed_time;
  
  // Use the getter methods to access the private members
  HOST_TYPE(Event_t) start_event = timer->get_start_event();
  HOST_TYPE(Event_t) stop_event = timer->get_stop_event();
  
  LIB_CALL(HOST_TYPE(EventElapsedTime)(&elapsed_time, start_event, stop_event));
  
  size_t calc_ops = timer->get_calc_ops();
  double flops = static_cast<double>(calc_ops);
  double gflops_val = (flops / (elapsed_time * 1e-3)) / 1e9;

  // Store results in the provided pointers
  float* time_ptr = timer->get_time_ptr();
  float* gflops_ptr = timer->get_gflops_ptr();
  
  if (time_ptr != nullptr) {
    *time_ptr = elapsed_time;
  }
  if (gflops_ptr != nullptr) {
    *gflops_ptr = static_cast<float>(gflops_val);
  }
  
  // Call user callback if provided
  timer->execute_callback(elapsed_time);
  timer->set_callback_executed(true);
}

KernelTimer::KernelTimer(size_t calc_ops, float *time, float *gflops)
    : calc_ops(calc_ops), time_ptr(time), gflops_ptr(gflops), user_data(nullptr), 
      callback(nullptr), callback_executed(false) {
  LIB_CALL(HOST_TYPE(EventCreate)(&start));
  LIB_CALL(HOST_TYPE(EventCreate)(&stop));
}

void KernelTimer::start_timer(hipStream_t stream) { 
  LIB_CALL(HOST_TYPE(EventRecord)(start, stream));
  callback_executed = false;
}

void KernelTimer::stop_timer(hipStream_t stream) {
  LIB_CALL(HOST_TYPE(EventRecord)(stop, stream));
  // Instead of synchronizing, add a callback to the stream that will be called when the event completes
  LIB_CALL(hipStreamAddCallback(stream, eventCallback, this, 0));
}

void KernelTimer::set_callback(TimerCompletionCallback cb, void* data) {
  callback = cb;
  user_data = data;
}

void KernelTimer::execute_callback(float elapsed_time) {
  if (callback && !callback_executed) {
    callback(elapsed_time, calc_ops, time_ptr, gflops_ptr, user_data);
  }
}

void KernelTimer::synchronize() {
  // If callback hasn't been executed yet, synchronize and wait for event completion, then manually execute callback
  if (!callback_executed) {
    LIB_CALL(HOST_TYPE(EventSynchronize)(stop));
    float elapsed_time;
    LIB_CALL(HOST_TYPE(EventElapsedTime)(&elapsed_time, start, stop));
    
    double flops = static_cast<double>(calc_ops);
    double gflops_val = (flops / (elapsed_time * 1e-3)) / 1e9;

    // Store results in the provided pointers
    if (time_ptr != nullptr) {
      *time_ptr = elapsed_time;
    }
    if (gflops_ptr != nullptr) {
      *gflops_ptr = static_cast<float>(gflops_val);
    }
    
    // Execute callback
    if (callback) {
      callback(elapsed_time, calc_ops, time_ptr, gflops_ptr, user_data);
    }
    callback_executed = true;
  }
}

KernelTimer::~KernelTimer() {
  // Synchronize during destruction to ensure callback is executed
  synchronize();
  LIB_CALL(HOST_TYPE(EventDestroy)(start));
  LIB_CALL(HOST_TYPE(EventDestroy)(stop));
}