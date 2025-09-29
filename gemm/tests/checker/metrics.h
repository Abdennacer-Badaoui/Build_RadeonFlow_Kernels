#pragma once


struct PerfMetricEntry {
    char name[20];
    float time;
    float gflops;
};

struct PerfMetrics {
    int count = 0;
    PerfMetricEntry entries[20];
};