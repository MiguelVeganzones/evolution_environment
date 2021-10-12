//To use in various threads, uncoment mutex to use the same instance of mt19937 
//I think initialicing random::init once per thread does not work

#include "Random.h"
#include <mutex>
#include <chrono>

std::mt19937  random::s_random_engine;
std::uniform_int_distribution<std::mt19937::result_type> random::s_distribution;

std::mutex mu_randint;
std::mutex mu_randfloat;
std::mutex mu_randnormal;

void random::init()
{
    s_random_engine.seed((unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

float random::randfloat()
{
    return (float)s_distribution(random::s_random_engine) / (float)std::numeric_limits<uint32_t>::max();
}

float random::mt_randfloat()
{
    mu_randfloat.lock();
    const float f = (float)s_distribution(random::s_random_engine) / (float)std::numeric_limits<uint32_t>::max();
    mu_randfloat.unlock();
    return f;
}

int random::randint(int Min = 0, int Max = 10)
{   
    std::uniform_int_distribution<int> U(Min, Max);
    return U(random::s_random_engine);
}

int random::mt_randint(int Min = 0, int Max = 10) //multi threaded randint
{
    std::uniform_int_distribution<int> U(Min, Max);
    mu_randint.lock();
    const int n = U(random::s_random_engine);
    mu_randint.unlock();
    return n;
}

float random::randnormal(const float avg, const float stddev) {
    std::normal_distribution<float> N(avg, stddev);
    return N(random::s_random_engine);
}

float random::mt_randnormal(const float avg, const float stddev) {
    std::normal_distribution<float> N(avg, stddev);
    mu_randnormal.lock();
    const float n = N(random::s_random_engine);
    mu_randnormal.unlock();
    return n;
}