#pragma once
#include "ga_nn.h"
#include <vector>
#include <atomic>

namespace _c4_brain {

	class c4_brain {
	private:
		mutable std::unique_ptr<_ga_nn::neural_net> p_net;
		std::atomic<uint_fast16_t> wins;
		std::atomic<uint_fast16_t> losses;
		std::atomic<uint_fast16_t> ties;
		uint_fast16_t ID;
		std::vector<uint_fast8_t> shape;
		static uint_fast16_t s_ID;

	public:
		c4_brain(const std::vector<uint_fast8_t>& _shape);
		c4_brain(const _ga_nn::neural_net& nn);
		c4_brain(const c4_brain& brain);

		uint_fast8_t weigh(const board& current_board, const bool player) const;
		//inline static uint_fast8_t s_weigh(const _c4_brain::c4_brain& b, const board& _board) { return b.weigh(_board); }

		inline void won() { ++wins; }
		inline void lost() { ++losses; }
		inline void tied() { ++ties; }

		inline const float get_fitness() const {
			return (wins + losses + ties) != 0 ? ((float(wins) + (float(ties) / 3.f)) / float(wins + losses + ties)) : 0;
		}
		inline const std::unique_ptr<_ga_nn::neural_net>& get_net() const { return p_net; }
		inline float get_winrate() const { return (wins + losses + ties) != 0 ? (float(wins) / float(wins + losses + ties)) : 0; }
		inline const std::vector<uint_fast8_t> get_shape() const { return shape; }
		inline const uint_fast32_t parameter_count() const { return p_net->parameter_count(); }
		void print_stats() const;
		void inline reset() { wins = 0; losses = 0; ties = 0; }

		static inline uint_fast16_t get_current_ID() { return s_ID; }

		void mutate(float p = 0.1f, float avg = 0.f, float stddev = 0.1f);
		void store(const char* const file_name) const;

		friend std::ostream& operator <<(std::ostream& os, const _c4_brain::c4_brain& brain);
	};

	const std::pair<c4_brain, c4_brain> crossover(const c4_brain& brain1, const c4_brain& brain2);
	const std::pair<c4_brain, c4_brain> crossover(c4_brain* brain1, c4_brain* brain2);

	const _matrix::matrix<double> population_variability(const std::vector<c4_brain*>& brains);
	const _matrix::matrix<double> population_variability(const std::vector<std::reference_wrapper<c4_brain>>& brains);

	_c4_brain::c4_brain read(const char* const file_name);

	std::ostream& operator <<(std::ostream& os, const _c4_brain::c4_brain& brain);
	inline bool operator == (const c4_brain& b1, const c4_brain& b2) { return b1.get_net() == b2.get_net(); }
}