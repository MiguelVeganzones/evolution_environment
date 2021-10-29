#pragma once
#include "c4_evo_environment.h"
#include "non_parallel_minimax.h"
#include <cmath>
#include <execution>

std::vector<_c4_brain::c4_brain*> _c4_evo_env::selection_operator(const std::vector<_c4_brain::c4_brain*>& brains,
	std::valarray<double>&& _winrate, float alpha, uint_fast8_t n)
{
	assert(brains.size() == winrate.size());
	const uint_fast8_t _size = brains.size();

	std::vector<uint_fast8_t> _parents(n);

	const _matrix::matrix<double> variability_matrix = _c4_brain::population_variability(brains);
	std::valarray<double> weight_vector(_size);

	uint_fast8_t i, j, _index;

	_index = std::distance(std::begin(_winrate), std::max_element(std::begin(_winrate), std::end(_winrate))); //index of max winrate
	_parents[0] = _index;
	float _prev_winrate = 0;
	float max_d; //max distance to normalice distances every iteration

	for (i = 1; i < n; ++i) {
		_prev_winrate = _winrate[_index];
		_winrate[_index] = std::numeric_limits<double>::lowest();

		max_d = *std::max_element(std::begin(variability_matrix[_index]), std::end(variability_matrix[_index]));
		weight_vector = variability_matrix[_index] / max_d * _prev_winrate * alpha + _winrate;

		_index = std::distance(std::begin(weight_vector), std::max_element(std::begin(weight_vector), std::end(weight_vector))); //index of max weight
		_parents[i] = _index;
	}

	std::vector<_c4_brain::c4_brain*> ret;
	ret.reserve(n);

	for (auto p_idx: _parents) {
		ret.push_back(brains[p_idx]);
	}

	return ret;
}

void _c4_evo_env::tournament(const std::vector<_c4_brain::c4_brain*>& prev_gen, const std::vector<_c4_brain::c4_brain*>& curr_gen,
	const uint_fast8_t prev_depth, const uint_fast8_t curr_depth, const bool print)
{
	std::for_each(std::execution::par_unseq, std::begin(curr_gen), std::end(curr_gen), [&](_c4_brain::c4_brain* curr) {
		_round(prev_gen, curr, prev_depth, curr_depth);
		});

	if (print) {
		for (const auto& e : curr_gen) {
			e->print_stats();
		}
	}
}		

const _c4_brain::c4_brain _c4_evo_env::simulate_evolution(const uint_fast8_t pop_size, const uint_fast8_t parents, const uint_fast16_t epochs,
	const std::vector<uint_fast8_t>& _shape, const uint_fast8_t cur_depth, const uint_fast8_t prev_depth, const uint_fast8_t control_epochs,
	const uint_fast8_t top_n, const float mutation_p, const uint_fast8_t control_group_size)
{
	//random::init();
	//randomly initialize the first generation.
	std::vector<_c4_brain::c4_brain> brains;

	brains.reserve(pop_size);

	for (uint_fast8_t i = 0; i < pop_size; ++i) {
		brains.emplace_back(_shape);
	}

	return simulate_evolution_helper(std::move(brains), parents, epochs, _shape, cur_depth, prev_depth,
		control_epochs, top_n, mutation_p, control_group_size);

}

const _c4_brain::c4_brain _c4_evo_env::simulate_evolution(std::vector<_c4_brain::c4_brain>&& _brains, const uint_fast8_t parents,
	const uint_fast16_t epochs, const uint_fast8_t cur_depth, const uint_fast8_t prev_depth,
	const uint_fast8_t control_epochs, const uint_fast8_t top_n, const float mutation_p, const uint_fast8_t control_group_size)
{
	return simulate_evolution_helper(std::move(_brains), parents, epochs, _brains[0].get_shape(),
		cur_depth, prev_depth, control_epochs, top_n, mutation_p, control_group_size);
}

const _c4_brain::c4_brain _c4_evo_env::simulate_evolution_helper(std::vector<_c4_brain::c4_brain>&& _brains, const uint_fast8_t parents, 
	const uint_fast16_t epochs, const std::vector<uint_fast8_t>& _shape, const uint_fast8_t cur_depth, 
	const uint_fast8_t prev_depth, const uint_fast8_t control_epochs,
	const uint_fast8_t top_n, const float mutation_p, const uint_fast8_t control_group_size)
{
	const uint_fast8_t pop_size = _brains.size();

	//random::init();
	//utiity
	uint_fast8_t j;
	uint_fast16_t i = 0;
	std::array<uint_fast8_t, 2> gi({ 0,1 }); //gen index
	std::valarray<double> weights(pop_size);
	std::vector<_c4_brain::c4_brain*> _parents(parents, nullptr);

	//brain vectors a and b will hold previous or current generation depending on the epoch

	std::array<std::vector<_c4_brain::c4_brain>, 2> brains;

	brains[gi[0]].reserve(pop_size);
	
	brains[gi[1]] = std::vector<_c4_brain::c4_brain>(std::make_move_iterator(_brains.begin()),
													 std::make_move_iterator(_brains.end()));

	std::array<std::vector<_c4_brain::c4_brain*>, 2> gen; //gen[gi[0]] is previous gen, gen[gi[1]] is current generation

	for (std::vector<_c4_brain::c4_brain*>& e : gen) {
		e.reserve(pop_size);
	}

	//only for first gen, cur gen and prev gen are the same 
	for (i = 0; i < pop_size; ++i) {
		gen[gi[0]].push_back(&brains[gi[1]][i]);
		gen[gi[1]].push_back(&brains[gi[1]][i]);
	}

	//control group
	std::vector<_c4_brain::c4_brain> control_group;
	std::vector<_c4_brain::c4_brain*> control_gen;
	control_group.reserve(control_group_size);
	control_gen.reserve(control_group_size);

	for (i = 0; i < control_group_size; ++i) {
		control_group.emplace_back(_shape);
		control_gen.push_back(&control_group[i]);
	}

	//competition and breeding
	for (i = 0; i < epochs; ++i) {

		tournament(gen[gi[0]], gen[gi[1]], prev_depth, cur_depth);

		for (j = 0; j < pop_size; ++j) {
			weights[j] = gen[gi[1]][j]->get_fitness();
		}

		//for (int k = 0; k < top_n; ++k) {
		//	std::cout << "prev " << k << " == curr " << k << ": " << (*gen[gi[1]][k]->get_net().get() == *gen[gi[0]][k]->get_net().get())<<"\n";
		//}

		if (i == (epochs - 1)) {
			uint_fast8_t ret_i = std::distance(std::begin(weights), std::max_element(std::begin(weights), std::end(weights)));
			//for (const auto& e : gen[gi[0]]) {
			//	e->print_stats();
			//}
			for (const auto& e : gen[gi[1]]) {
				e->print_stats();
			}
			std::cout << "Best winrate in the last epoch: " << weights[ret_i] << "\n";
			return brains[gi[1]][ret_i];
		}

		_parents = selection_operator(gen[gi[1]], std::move(weights), alpha(i, epochs), parents);


		//std::cout << "Parents: \n";
		//for (const auto& e : _parents) e->print_stats();
		//std::cout << "\nEnd parents\n";


		//for (const auto& e : gen[gi[0]]) {
		//	e->print_stats();
		//}
		//for (const auto& e : gen[gi[1]]) {
		//	e->print_stats();
		//}

		//brains[gi[0]].clear();
		//for (j = 0 ; j < pop_size;++j){
		//	brains[gi[0]].emplace_back(_shape);
		//}

		brains[gi[0]] = breed_new_gen(_parents, pop_size, top_n, mutation_p, alpha(i, epochs));

		//brains[bi[0]] = new gen;

		for (j = 0; j < pop_size; ++j) {
			gen[gi[0]][j] = &brains[gi[0]][j]; //store next gen pointes in previous gen, next current gen
		}

		//std::cout << "\n###############################\n\n";

		//for (const auto& e : gen[gi[1]]) {
		//	e->print_stats();
		//}

		if ((i % control_epochs) == 0) {
			std::cout << "\nPopulation variability: \n" << _c4_brain::population_variability(gen[gi[1]]);
			std::cout << "\nControl tournament: " << std::endl;
			check_progress(control_gen, gen[gi[1]], cur_depth);
		}

		std::swap(gi[0], gi[1]);
	}

	assert(0); throw 0;
}

std::vector<_c4_brain::c4_brain> _c4_evo_env::breed_new_gen(const std::vector<_c4_brain::c4_brain*>& parents,
	const uint_fast8_t pop_size, const uint_fast8_t top_n, const float mutation_p, const float alpha)
{
	assert((top_n < pop_size) and (mutation_p <= 1) and (mutation_p >= 0) and (parents.size() >= top_n));

	uint_fast8_t i, a, b;
	uint_fast8_t meio = make_even(pop_size - top_n), mito = pop_size - meio - top_n;
	const uint_fast8_t np = parents.size();

	assert(meio + mito + top_n == pop_size);

	std::vector<_c4_brain::c4_brain> ret;
	ret.reserve(pop_size);

	for (i = 0; i < top_n; ++i) {
		ret.emplace_back(*parents[i]->get_net().get());
	}

	for (i = 0; i < meio; i += 2) {
		a = random::randint(0, np - 1);

		do {
			b = random::randint(0, np - 1);
		} while (a == b); //unefficient but will affect performance minimally

		std::pair<_c4_brain::c4_brain, _c4_brain::c4_brain> b_pair = _c4_brain::crossover(parents[a], parents[b]);
		ret.push_back(std::move(b_pair.first));
		ret.push_back(std::move(b_pair.second));
	}

	if(mito){
		a = random::randint(0, np - 1);
		_ga_nn::neural_net temp(*parents[a]->get_net().get());
		temp.mutate(0.2f, random::randfloat() * 0.2f, random::randfloat() * 0.2f);
		ret.emplace_back(std::move(temp));
	}

	for (i = top_n; i < pop_size; ++i) {
		if (random::randfloat() < mutation_p) {
			const float stddev = 0.38 * alpha + 0.02; //decreases linearly with alpha: 0.4 >= x >= 0.02
												     // if alpha == alpha(iter, epochs), x decreases linearly with epochs			
			ret[i].mutate(0.1f, 0, stddev);
		}
	}

	return ret;
}

void _c4_evo_env::check_progress(const std::vector<_c4_brain::c4_brain*>& control_group, const std::vector<_c4_brain::c4_brain*>& cur_gen, 
	const uint_fast8_t depth)
{
	for (auto& e : cur_gen) {
		e->reset();
	}

	tournament(control_group, cur_gen, depth, depth);

	uint_fast8_t n = cur_gen.size();
	std::vector<float> f(n); //fitness

	for (int i = 0; i < n; ++i) {
		//std::cout << "Brain " << i << " winrate: " << (w[i] = (cur_gen[i]->get_winrate())) << "\n";
		f[i] = (cur_gen[i]->get_fitness());
	}

	float m = 0, s = 0; //mu and sigma
	for (auto e : f) { m += e; }
	m /= n;
	
	float temp = 0;
	for (auto e : f) {
		temp += pow(e - m, 2);
	}

	s = sqrt(temp / (n - 1));

	std::cout << "Best performance: " << *std::max_element(f.begin(), f.end()) <<
		"\nWorst performance: " << *std::min_element(f.begin(), f.end()) <<
		"\nThe data fits a normal distribution N(" << m << ", " << s << ")\n";

	//for (const auto& e : cur_gen) {
	//	e->print_stats();
	//}

}

void _c4_evo_env::_round(const std::vector<_c4_brain::c4_brain*>& prev_gen, _c4_brain::c4_brain* curr,
	const uint_fast8_t prev_depth, const uint_fast8_t curr_depth) {
	//current plays second
	_non_parallel_helper_foo::clear_seen_boards();
	int_fast8_t res;
	
	for (const auto& p : prev_gen) {
		res = np_ai_play_ai(p, curr, prev_depth, curr_depth, 1, 1);
		switch (res) {
		case 0:
			curr->lost();
			break;
		case 1:
			curr->won();
			break;
		case -1:
			curr->tied();
			break;
		}
		//current plays first
		res = np_ai_play_ai(curr, p, curr_depth, prev_depth, 1, 0);
		switch (res) {
		case 0:
			curr->won();
			break;
		case 1:
			curr->lost();
			break;
		case -1:
			curr->tied();
			break;
		}
	}
	_non_parallel_helper_foo::clear_seen_boards();
	
}

