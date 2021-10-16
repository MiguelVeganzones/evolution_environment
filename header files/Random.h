#pragma once
#include <random>

static class random 
{
public:

	static void init();

	static float randfloat();

	static int randint(int Min, int Max);

	static int mt_randint(int Min, int Max);

	static float mt_randfloat();

	static float randnormal(const float avg = 0.f, const float stddev = 1.f);

	static float mt_randnormal(const float avg = 0.f, const float stddev = 1.f);

private:
	static std::mt19937 s_random_engine;
	static std::uniform_int_distribution<std::mt19937::result_type> s_distribution;
};