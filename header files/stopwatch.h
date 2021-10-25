#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>

class stopwatch
{
public:
    inline stopwatch()
    {
        start = std::chrono::steady_clock::now();
    }

    inline ~stopwatch()
    {
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Process took " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() <<
            "ms\t (" << duration.count() << "us)\t (" <<
            std::chrono::duration_cast<std::chrono::seconds>(duration).count() << " s)\t (" <<
            std::chrono::duration_cast<std::chrono::minutes>(duration).count() << " mins)\n" << std::defaultfloat;
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
};

/*

time a function :

auto t0 = std::chrono::steady_clock::now();
f();
auto t1 = std::chrono::steady_clock::now();
// std::cout << nanoseconds{t-t0}.count << "ns\n";
// std::cout << std::chrono::duration<double>{t1-t0}.count(); // print in floating point seconds
// std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();


*/