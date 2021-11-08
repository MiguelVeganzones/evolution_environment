#pragma once
#include "pch.h"
#include "CppUnitTest.h"
#include "c4_evo_environment.h"
#include "small_GA_matrix.h"
#include "Random.h"
#include "Random.cpp" //random linker issue

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace unittests
{
	TEST_CLASS(unittests)
	{
	public:
		
		TEST_METHOD(assert_mat_mat_mul)
		{
			// setup
			random::init();
			
			unsigned int n = 10;
			unsigned int m = 10;
			unsigned int k = 10;
		
			for (int i = 0; i < 5; ++i) {
				_matrix::matrix<float> m1(random::randfloat, 0, n, k);
				_matrix::matrix<float> m2(random::randfloat, 0, k, m);

				Assert::AreEqual(_matrix::d1_distance(_matrix::dot(m1, m2), _matrix::big_mat_dot(m1, m2)), 0.0);

				n *= 1.2;
				m *= 2.3;
				k *= 1.5;
			}
		}

		TEST_METHOD(assert_mat_vec_mul)
		{
			// setup
			random::init();

			unsigned int n = 10;
			unsigned int k = 10;

			for (int i = 0; i < 5; ++i) {
				_matrix::matrix<float> m1(random::randfloat, 0, n, k);

				std::valarray<float> v1(k);
				for (auto& e : v1) e = random::randfloat();

				Assert::AreEqual(abs(_matrix::dot(m1, v1) - _matrix::big_mat_dot(m1, v1)).sum(), 0.f);

				n *= 1.2;
				k *= 1.5;
			}
		}

		TEST_METHOD(assert_vec_mat_mul)
		{
			// setup
			random::init();

			unsigned int m = 10;
			unsigned int k = 10;

			for (int i = 0; i < 5; ++i) {
				_matrix::matrix<float> m2(random::randfloat, 0, k, m);

				std::valarray<float> v2(k);
				for (auto& e : v2) e = random::randfloat();

				Assert::AreEqual(abs(_matrix::dot(v2, m2) - _matrix::big_mat_dot(v2, m2)).sum(), 0.f);

				m *= 1.2;
				k *= 1.5;
			}
		}

	};
}
