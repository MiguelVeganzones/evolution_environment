#pragma once
#include "ga_nn.h"
#include "board.h"
#include "non_parallel_minimax.h"
#include <vector>
#include "c4_brain.h"
#include "non_parallel_minimax.h"


namespace _c4_evo_env {
	constexpr uint8_t p_size = 20;

	/// <summary>
	/// Selects the fittest individuals among a population to produce the next generation
	/// </summary>
	/// <param name="brains"> Vector of brain individuals, potential parents of the next generation </param>
	/// <param name="winrate"> Winrate of every brain after playing against each other </param>
	/// <param name="alpha"> Weight of distance between individuals to obtain fitness </param>
	/// <param name="n"> Number of parents to select </param>
	/// <returns> vector of pointers to selected brains </returns>
	std::vector<_c4_brain::c4_brain*> selection_operator(const std::vector<_c4_brain::c4_brain*>& brains,
		const std::valarray<double>& winrate, float alpha, uint_fast8_t n);

	void tournament(const std::vector<_c4_brain::c4_brain*>& prev_gen, const std::vector<_c4_brain::c4_brain*>& curr_gen,
		uint_fast8_t prev_depth, const uint_fast8_t curr_depth, const bool print = 0);

		/// <summary>
		/// This function will randomly generate a population of c4_brains, make them compete against each other, select the fittest individuals
		/// that will then breed a new generation and then iterate to simulate evolution
		/// </summary>
		/// <param name="pop_size"> Population size of each generation </param>
		/// <param name="parents"> Number of parents that will breed the next generagtion </param>
		/// <param name="epochs"> Number of generations for which the simulation will run </param>
		/// <param name="_shape"> Shape of the neural net inside the brains </param>
		/// <param name="cur_depth"> Search depth of the current generation when competing </param>
		/// <param name="prev_depth"> Search depth of the previous generation when competing </param>
		/// <param name="control_epochs"> Number of epochs between each progress check against a control population </param>
		/// <returns> Copy of the best brain of the last generation by value </returns>
	const _c4_brain::c4_brain simulate_evolution(const uint_fast8_t pop_size, const uint_fast8_t parents,
		const uint_fast8_t epochs, const std::vector<uint_fast8_t>& _shape,
		const uint_fast8_t cur_depth = 2, const uint_fast8_t prev_depth = 3, const uint_fast8_t control_epochs = 10,
		const uint_fast8_t top_n = 1, const float mutation_p = 0.2, const uint_fast8_t control_group_size = 10);

	const _c4_brain::c4_brain simulate_evolution(std::vector<_c4_brain::c4_brain>&& _brains, const uint_fast8_t parents,
		const uint_fast8_t epochs,
		const uint_fast8_t cur_depth = 2, const uint_fast8_t prev_depth = 3, const uint_fast8_t control_epochs = 10,
		const uint_fast8_t top_n = 1, const float mutation_p = 0.2, const uint_fast8_t control_group_size = 10);

	const _c4_brain::c4_brain simulate_evolution_helper(std::vector<_c4_brain::c4_brain>&& _brains, const uint_fast8_t parents, const uint_fast8_t epochs,
		const std::vector<uint_fast8_t>& _shape, const uint_fast8_t cur_depth, const uint_fast8_t prev_depth, const uint_fast8_t control_epochs,
		const uint_fast8_t top_n, const float mutation_p, const uint_fast8_t control_group_size);

	//part meiosis, part mitosis. Top n individuals get into the new gen as is 
	std::vector<_c4_brain::c4_brain> breed_new_gen(const std::vector<_c4_brain::c4_brain*>& parents, 
		const uint_fast8_t pop_size, const uint_fast8_t top_n, const float mutation_p);

	void check_progress(const std::vector<_c4_brain::c4_brain*>& control_group,
		const std::vector<_c4_brain::c4_brain*>& cur_gen, const uint_fast8_t depth);

	void _round(const std::vector<_c4_brain::c4_brain*>& prev_gen, _c4_brain::c4_brain* curr,
		const uint_fast8_t prev_depth, const uint_fast8_t curr_depth);

	inline float alpha(const uint_fast8_t i, const uint_fast8_t epochs) { return 1 - i / epochs; }
	inline uint_fast8_t make_even(const uint_fast8_t n) { return (n / 2) * 2; }
}