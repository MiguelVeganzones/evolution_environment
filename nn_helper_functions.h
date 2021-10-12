#pragma once
#include <valarray>

namespace _nn_func {

	static inline std::valarray<float> relu(const std::valarray<float>& v) {
		return v.apply([](float f) {return f > 0 ? f : 0; });
	}

	static inline std::valarray<float> sigmoid(const std::valarray<float>& v) {
		return v.apply([](float f) {return 1.f / (1.f + exp(-f)); });
	}

	static inline std::valarray<float> softmax(const std::valarray<float>& v) {
		std::valarray<float> ret = exp(v);
		return ret / ret.sum();
	}

	static inline std::valarray<float> equality(const std::valarray<float>& v) {
		return v;
	}
}