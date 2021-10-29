#pragma once
#include "c4_brain.h"
#include <fstream>
#include <variant>
#include <string>
#include "small_GA_matrix.h"

using namespace _c4_brain;

uint_fast16_t _c4_brain::c4_brain::s_ID = 0;

_c4_brain::c4_brain::c4_brain(const std::vector<uint_fast8_t>& _shape) : shape{ _shape }
{
	p_net = std::make_unique<_ga_nn::neural_net>(_ga_nn::neural_net(_shape));
}

_c4_brain::c4_brain::c4_brain(const _ga_nn::neural_net& nn) : shape{ nn.get_shape() }
{
	p_net = std::make_unique<_ga_nn::neural_net>(nn);
}

_c4_brain::c4_brain::c4_brain(const c4_brain& brain) : shape{ brain.get_shape() }
{
	p_net = std::make_unique<_ga_nn::neural_net>(*brain.get_net());
}

_c4_brain::c4_brain::c4_brain(c4_brain&& other) noexcept : p_net(std::move(other.p_net)), shape(std::move(other.shape)) {

}

uint_fast8_t _c4_brain::c4_brain::weigh(const board& current_board, const bool player) const {
	//get from board to _matrix::matrix<float>
	_matrix::matrix<float> matrix_board(b_const::sy, b_const::sx);
	for (uint_fast8_t j = 0; j < b_const::sy; ++j) {
		for (uint_fast8_t i = 0; i < b_const::sx; ++i) {
			matrix_board(j, i) = current_board(j, i);
		}
	}
	//std::cout << matrix_board << std::endl;

	return uint_fast8_t(p_net->forard_pass(matrix_board)[player] * 253) + 1;
}

void _c4_brain::c4_brain::print_stats() const
{
	std::cout <<
		"\n ............. C4 Brain " << (int)ID << " ............." <<
		"\nWins : " << (int)wins <<
		"\nLosses : " << (int)losses <<
		"\nTies : " << (int)ties <<
		"\nWin ratio : " << get_winrate() <<
		"\nFitness : " << get_fitness() <<
		"\n( ";
	for (int i = 0; i < shape.size(); i += 2) {
		std::cout << "(" << (int)shape[i] << ", " << (int)shape[i + 1] << ") ,";
		}
	std::cout << "\b) \nParameter count: " << parameter_count() <<
		"\n---------------------------------------\n";
}

void _c4_brain::c4_brain::mutate(float p, float avg, float stddev)
{
	p_net->get_head()->mutate(p, avg, stddev);
	for (auto& hidden : p_net->get_hidden())hidden->mutate(p, avg, stddev);
	p_net->get_tail()->mutate(p, avg, stddev);
}

void _c4_brain::c4_brain::store(const char* const file_name) const
{
	std::fstream out;

	out.open(file_name, std::ios::out);
	if (!out) {
		std::cout << "File not created!"; throw 0;
	}

	for (const auto e : shape) {
		out << int(e) << ' ';
	}
	out << "\n\n";

	for (const _ga_nn::node& n : p_net->get_head()->get()) {
		for (const auto& row : n.get().get()) {
			for (auto col : row) {
				out << col << ',';
			}
			out << '\n';
		}
		out << '\n';
	}

	for (const auto& h : p_net->get_hidden()) {
		for (const _ga_nn::node& n : h->get()) {
			for (const auto& row : n.get().get()) {
				for (auto col : row) {
					out << col << ',';
				}
				out << '\n';
			}
			out << '\n';
		}
	}

	for (const _ga_nn::node& n : p_net->get_tail()->get()) {
		for (const auto& row : n.get().get()) {
			for (auto col : row) {
				out << col << ',';
			}
			out << '\n';
		}
		out << '\n';
	}

	out.close();
}

_c4_brain::c4_brain _c4_brain::read(const char* const file_name)
{
	std::fstream in;
	in.open(file_name, std::ios::in);
	if (!file_name) {
		std::cout << "Unable to open file.\n"; throw 0;
	}

	unsigned char c;
	std::vector<uint_fast8_t> _shape(1);

	while (in >> std::noskipws >> c) {
		if (c == ' ') {
			_shape.push_back(0);
		}
		else if (c == '\n') { _shape.pop_back(); break; }
		else {
			_shape.back() = _shape.back() * 10 + (uint_fast8_t)(c - '0');
		}
	}

	std::string line;

	uint_fast8_t n = _shape.size();
	uint_fast8_t i, ii, jj, k, kk, x, y, prev_x, prev_y, next_x, next_y;

	_ga_nn::neural_net ret(_shape);
	
	for (k = 0; k < n; k += 2) {//iterate over layers
		y = _shape[k];
		x = _shape[k + 1];
		prev_x = k - 1 >= 0 ? _shape[k - 1] : 1;
		prev_y = k - 2 >= 0 ? _shape[k - 2] : 1;
		next_x = k + 3 < n ? _shape[k + 3] : 1;
		next_y = k + 2 < n ? _shape[k + 2] : 1;

		_matrix::matrix<float> mat(prev_x * prev_y, next_x * next_y);
		for (kk = 0; kk < y * x; ++kk) {
			for (jj = 0; jj < prev_x * prev_y; ++jj) {
				for (ii = 0; ii < next_x * next_y; ++ii) {
					std::getline(in, line, ',');
					mat(jj, ii) = std::stof(line);
				}
			}
			if (k == 0) {
				ret.get_head()->set(kk, mat);
			}
			else if (k == n - 2) {
				ret.get_tail()->set(kk, mat);
			}
			else {
				i = (k - 2) / 2;
				ret.get_hidden()[i]->set(kk, mat);
			}

			std::getline(in, line, '\n'); //read blank line
		}
	}

	return std::move(_c4_brain::c4_brain(ret));
}

const std::pair<_c4_brain::c4_brain, _c4_brain::c4_brain> _c4_brain::crossover(const c4_brain& brain1, const c4_brain& brain2)
{
	std::pair<_ga_nn::neural_net, _ga_nn::neural_net> ret_nets = _ga_nn::crossover(*brain1.get_net().get(), *brain2.get_net().get());
	return std::make_pair<c4_brain, c4_brain>(std::move(ret_nets.first), std::move(ret_nets.second));
}

const std::pair<_c4_brain::c4_brain, _c4_brain::c4_brain> _c4_brain::crossover(const c4_brain* brain1, const c4_brain* brain2)
{
	std::pair<_ga_nn::neural_net, _ga_nn::neural_net> ret_nets = _ga_nn::crossover(*brain1->get_net().get(), *brain2->get_net().get());
	return std::make_pair<c4_brain, c4_brain>(std::move(ret_nets.first), std::move(ret_nets.second));
}

const _matrix::matrix<double> _c4_brain::population_variability(const std::vector<c4_brain*>& brains)
{
	std::vector<_ga_nn::neural_net*> nets;
	nets.reserve(brains.size());
	for (const auto& b : brains) {
		nets.push_back(b->get_net().get());
	}
	return _ga_nn::population_variability(std::move(nets), 0);
}

const _matrix::matrix<double> _c4_brain::population_variability(const std::vector<std::reference_wrapper<c4_brain>>& brains)
{
	std::vector<std::reference_wrapper<_ga_nn::neural_net>> nets;
	nets.reserve(brains.size());
	for (const auto& b : brains) {
		nets.push_back(*b.get().get_net());
	}
	return _ga_nn::population_variability(std::move(nets));
}

std::ostream& _c4_brain::operator<<(std::ostream& os, const _c4_brain::c4_brain& brain)
{
	os <<
		"\n\n###################################################################\n" <<
		"\n ............. C4 Brain " << (int)brain.ID << " ............." <<
		"\nWins : " << (int)brain.wins <<
		"\nLosses : " << (int)brain.losses <<
		"\nTies :" << (int)brain.ties <<
		"\nWin ratio : " << brain.get_winrate() <<
		"\nFitness : " << brain.get_fitness() <<
		"\n( ";
	const auto _shape = brain.get_shape();
		for (int i = 0; i < _shape.size(); i += 2) {
			os << "(" << (int)_shape[i] << ", " << (int)_shape[i + 1] << ") ,";
		}
	std::cout << "\b) \nParameter count: " << brain.parameter_count() <<
		"\n-------------------------------------------\n" <<
		*brain.get_net() <<
		"######################################################################\n";
	return os; 
}
