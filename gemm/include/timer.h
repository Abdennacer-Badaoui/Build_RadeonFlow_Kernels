#pragma once
#include "gpu_types.h"
#include "gpu_libs.h"
#include <memory>

typedef void (*TimerCompletionCallback)(float elapsed_time, size_t calc_ops, float *time_ptr, float *gflops_ptr,
                                        void *user_data);

class KernelTimer {
  private:
    size_t calc_ops;
    HOST_TYPE(Event_t) start, stop;
    float *time_ptr;
    float *gflops_ptr;
    void *user_data;
    TimerCompletionCallback callback;
    bool callback_executed;

  public:
    KernelTimer(size_t calc_ops, float *time, float *gflops);

    void start_timer(hipStream_t stream = 0);
    void stop_timer(hipStream_t stream = 0);
    void set_callback(TimerCompletionCallback cb, void *data = nullptr);

    // Wait for the timer to complete and execute the callback if set
    void synchronize();

    // Getter methods for the callback
    HOST_TYPE(Event_t) get_start_event() const { return start; }
    HOST_TYPE(Event_t) get_stop_event() const { return stop; }
    size_t get_calc_ops() const { return calc_ops; }
    float *get_time_ptr() const { return time_ptr; }
    float *get_gflops_ptr() const { return gflops_ptr; }
    void execute_callback(float elapsed_time);
    void set_callback_executed(bool executed) { callback_executed = executed; }
    bool is_callback_executed() const { return callback_executed; }

    ~KernelTimer();
};

class KernelTimerScoped {
  private:
    std::shared_ptr<KernelTimer> timer;
    hipStream_t stream;

  public:
    KernelTimerScoped(std::vector<std::shared_ptr<KernelTimer>> &timers, size_t calc_ops, float *time, float *gflops,
                      hipStream_t stream = 0)
        : timer(std::make_shared<KernelTimer>(calc_ops, time, gflops)), stream(stream) {
        timers.push_back(timer);
        timer->start_timer(stream);
    }

    ~KernelTimerScoped() { timer->stop_timer(stream); }
};