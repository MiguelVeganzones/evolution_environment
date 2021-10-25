#pragma once
#include "ga_nn.h"
#include <vector>
#include <assert.h>
#include "board.h"
#include <fstream>
#include <iterator>

#ifndef NDEBUG
#define NDEBUG
#endif

using namespace _ga_nn;

node::node(const uint_fast8_t _inputs, const uint_fast8_t _outputs) : inputs{ _inputs }, outputs{ _outputs },
weights{ _matrix::matrix<float>(random::randfloat, -0.5f, inputs, outputs) }{

}

node::node(const _matrix::matrix<float>& _weights) : inputs{ _weights.get_m() }, outputs{ _weights.get_n() }, weights{ _weights } {

}

void node::mutate(const float p, const float avg, const float stddev, float(*randnormal)(float, float)) {
	weights.mutate(randnormal, p, avg, stddev);
}

std::valarray<float> node::forward_pass(const std::valarray<float>& v) const {
	return _matrix::dot(v, weights);
}

void node::set(const _matrix::matrix<float>& new_weights) {
	assert(new_weights.get_m() == inputs && new_weights.get_n() == outputs);
	weights = new_weights;
}

//void _ga_nn::node::store(const char* const file_name) const {
//	weights.store_csv(file_name);
//}

layer::layer(const uint_fast8_t _y, const uint_fast8_t _x, const uint_fast8_t _m, const uint_fast8_t _n) :
	x{ _x }, y{ _y }, m{ _m }, n{ _n }, nodes{ std::vector<node>(x * y) }
{
	for (auto& e : nodes) {
		e = node(m, n);
	}
}

_ga_nn::layer::layer(const uint_fast8_t _y, const uint_fast8_t _x, const std::vector<node>& _nodes) :
	nodes{ _nodes }, x{ _x }, y{ _y }, m{ _nodes[0].get_m() }, n{ _nodes[0].get_n() }{

}

_ga_nn::layer::layer(const uint_fast8_t _y, const uint_fast8_t _x, std::vector<node>&& _nodes) :
	m{ _nodes[0].get_m() }, n{ _nodes[0].get_n() }, nodes{ std::move(_nodes) }, x{ _x }, y{ _y }{

}

void _ga_nn::layer::mutate(float p, float avg, float stddev){
	for (auto& node : nodes) {
		node.mutate(p, avg, stddev);
	}
}

//void _ga_nn::layer::store(const char* const file_name) const{
//	for (const auto& node : nodes) {
//		node.store(file_name);
//	}
//}

_ga_nn::in_layer::in_layer(const uint_fast8_t _y, const uint_fast8_t _x, const uint_fast8_t _next_y, const uint_fast8_t _next_x) :
	layer(_y, _x, 1, _next_y* _next_x), next_x{ _next_x }, next_y{ _next_y } {

}

//constructor from vector of nodes
_ga_nn::in_layer::in_layer(const uint_fast8_t _y, const uint_fast8_t _x, const uint_fast8_t _next_y,
	const uint_fast8_t _next_x, const std::vector<node>& _nodes) : 
	layer(_y, _x, _nodes), next_y{_next_y}, next_x{_next_x}{

}

_ga_nn::in_layer::in_layer(const uint_fast8_t _y, const uint_fast8_t _x, const uint_fast8_t _next_y,
	const uint_fast8_t _next_x, std::vector<node>&& _nodes) :
next_y{ _next_y }, next_x{ _next_x }, layer(_y, _x, std::move(_nodes)) {

}

std::vector<_matrix::matrix<float>> _ga_nn::in_layer::forward_pass(const _matrix::matrix<float>& data,
	std::valarray<float>(*foo)(const std::valarray<float>&)) const
{
	assert(data.get_m() == y && data.get_n() == x);
	std::vector<_matrix::matrix<float>> ret(next_x * next_y, _matrix::matrix<float>(y, x));
	std::valarray<float> z;

	for (int j = 0; j < y; ++j) {
		for (int i = 0; i < x; ++i) {
			z = foo(nodes[j * x + i].forward_pass(std::valarray<float>(data(j, i), 1)));
			for (unsigned int k = 0; k < ret.size(); ++k) {
				ret[k](j, i) = z[k];
			}
		}
	}
	return ret;
}

_ga_nn::hidden_layer::hidden_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x,
	const uint_fast8_t _next_y, const uint_fast8_t _next_x) : layer(_y, _x, _prev_y* _prev_x, _next_y* _next_x),
	prev_x{ _prev_x }, prev_y{ _prev_y }, next_x{ _next_x }, next_y{ _next_y }{

}

_ga_nn::hidden_layer::hidden_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x,
	const uint_fast8_t _next_y, const uint_fast8_t _next_x, const std::vector<node>& _nodes) : 
	layer(_y, _x, _nodes), prev_x{ _prev_x }, prev_y{ _prev_y }, next_x{ _next_x }, next_y{ _next_y }{

}

_ga_nn::hidden_layer::hidden_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x,
	const uint_fast8_t _next_y, const uint_fast8_t _next_x, std::vector<node>&& _nodes) :
	prev_x{ _prev_x }, prev_y{ _prev_y }, next_x{ _next_x }, next_y{ _next_y }, layer(_y, _x, std::move(_nodes)) {

}

std::vector<_matrix::matrix<float>> _ga_nn::hidden_layer::forward_pass(const std::vector<_matrix::matrix<float>>& data,
	std::valarray<float>(*foo)(const std::valarray<float>&)) const
{
	assert(data[0].get_m() == prev_y && data[0].get_n() == prev_x);
	std::vector<_matrix::matrix<float>> ret(next_x * next_y, _matrix::matrix<float>(y, x));

	std::valarray<float> z(next_x * next_y);
	std::valarray<float> v(prev_x * prev_y);

	//iterate over every node in the layer
	for (int n = 0; n < x * y; ++n) {
		//construct entry vector to node: v
		for (int j = 0; j < prev_y; ++j) {
			for (int i = 0; i < prev_x; ++i) {
				v[j * prev_x + i] = data[n](j, i);
			}
		}
		//forward pass through the node
		z = foo(nodes[n].forward_pass(v));

		//std::cout << "z: \n";
		//for (auto& e : z)std::cout << e << " ";
		//std::cout << std::endl;

		for (int k = 0; k < ret.size(); ++k) {
			//return node output to matrix
			ret[k](int(n / x), n% x) = z[k];
		}
	}
	return ret;
}

_ga_nn::out_layer::out_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x) :
	layer(_y, _x, _prev_y* _prev_x, 1), prev_x{ _prev_x }, prev_y{ _prev_y } {

}

_ga_nn::out_layer::out_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y,
	const uint_fast8_t _x, const std::vector<node>& _nodes) :
	layer(_y, _x, _nodes), prev_x{ _prev_x }, prev_y{ _prev_y }{

}

_ga_nn::out_layer::out_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y,
	const uint_fast8_t _x, std::vector<node>&& _nodes) :
	prev_x{ _prev_x }, prev_y{ _prev_y }, layer(_y, _x, std::move(_nodes)) {

}

std::valarray<float> _ga_nn::out_layer::forward_pass(const std::vector<_matrix::matrix<float>>& data,
	std::valarray<float>(*foo)(const std::valarray<float>&)) const
{
	assert(data[0].get_m() == prev_y && data[0].get_n() == prev_x);
	std::valarray<float> ret(x * y);

	std::valarray<float> v(prev_x * prev_y);

	for (int k = 0; k < x * y; ++k) {
		//construct entry vector to node: v
		for (int j = 0; j < prev_y; ++j) {
			for (int i = 0; i < prev_x; ++i) {
				v[j * prev_x + i] = data[k](j, i);
			}
		}
		//forward pass through the node
		ret[k] = nodes[k].forward_pass(v)[0]; //only one output per node
	}

	return foo(ret);
}

_ga_nn::neural_net::neural_net(const std::vector<uint_fast8_t>& v) {
	assert(v.size() % 2 == 0 && v.size() >= 6);

	shape = v;

	const uint_fast8_t s = v.size();
	p_hidden.resize(int(s / 2) - 2);

	uint_fast8_t i = 0;
	p_head = std::make_unique<_ga_nn::in_layer>(_ga_nn::in_layer(v[i], v[i + 1], v[i + 2], v[i + 3]));

	for (int j = 0, i = 2; i < s - 2; i += 2, ++j) {
		p_hidden[j] = std::make_unique<_ga_nn::hidden_layer>(_ga_nn::hidden_layer(v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2], v[i + 3]));
	}

	i = s - 2;
	p_tail = std::make_unique<_ga_nn::out_layer>(_ga_nn::out_layer(v[i - 2], v[i - 1], v[i], v[i + 1]));
}

/// <summary>
///
/// </summary>
/// <param name="v"></param>
/// <param name="_nodes"></param>
_ga_nn::neural_net::neural_net(const std::vector<uint_fast8_t>& v, const std::vector<node>& _nodes)
{
	assert(v.size() % 2 == 0 && v.size() >= 6);

	shape = v;

	const uint_fast8_t s = v.size();
	p_hidden.resize(int(s / 2) - 2);

	uint_fast8_t i = 0;

	std::vector<node>::const_iterator start = _nodes.begin();
	std::vector<node>::const_iterator end = start + v[i] * v[i + 1];

	std::vector<node> head_nodes(start, end);

	p_head = std::make_unique<_ga_nn::in_layer>(_ga_nn::in_layer(v[i], v[i + 1], v[i + 2], v[i + 3], head_nodes));

	for (int j = 0, i = 2; i < s - 2; i += 2, ++j) {

		start = end;
		end += v[i] * v[i + 1];

		std::vector<node> hidden_nodes_i(start, end);
		p_hidden[j] = std::make_unique<_ga_nn::hidden_layer>
			(_ga_nn::hidden_layer(v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2], v[i + 3], hidden_nodes_i));
	}

	i = s - 2;
	start = end;
	end += v[i] * v[i + 1];
	std::vector<node> tail_nodes(start, end);
	p_tail = std::make_unique<_ga_nn::out_layer>(_ga_nn::out_layer(v[i - 2], v[i - 1], v[i], v[i + 1], tail_nodes));
}

/// <summary>
///
/// </summary>
/// <param name="v"></param>
/// <param name="_nodes"></param>
_ga_nn::neural_net::neural_net(const std::vector<uint_fast8_t>& v, std::vector<node>&& _nodes)
{
	assert(v.size() % 2 == 0 && v.size() >= 6);
	shape = v;

	const uint_fast8_t s = v.size();
	p_hidden.resize(int(s / 2) - 2);

	uint_fast8_t i = 0;

	std::move_iterator start = std::make_move_iterator(_nodes.begin());
	std::move_iterator end = (start + v[i] * v[i + 1]);

	p_head = std::make_unique<_ga_nn::in_layer>(_ga_nn::in_layer(v[i], v[i + 1], v[i + 2], v[i + 3], std::vector<node>(start, end)));

	for (int j = 0, i = 2; i < s - 2; i += 2, ++j) {

		start = end;
		end += v[i] * v[i + 1];

		p_hidden[j] = std::make_unique<_ga_nn::hidden_layer>
			(_ga_nn::hidden_layer(v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2], v[i + 3], std::vector<node>(start, end)));
	}

	i = s - 2;
	start = end;
	end += v[i] * v[i + 1];
	p_tail = std::make_unique<_ga_nn::out_layer>(_ga_nn::out_layer(v[i - 2], v[i - 1], v[i], v[i + 1], std::vector<node>(start, end)));
}

_ga_nn::neural_net::neural_net(const neural_net& net): shape{net.shape}
{
	p_head = std::make_unique<_ga_nn::in_layer>(*net.get_head());

	const auto d = net.p_hidden.size();
	p_hidden.reserve(d);
	for (unsigned short int i = 0; i < d; ++i) {
		p_hidden.push_back(std::make_unique<_ga_nn::hidden_layer>(*net.get_hidden()[i]));
	}

	p_tail = std::make_unique<_ga_nn::out_layer>(*net.get_tail());
}

const uint_fast16_t _ga_nn::neural_net::parameter_count() const
{
	uint_fast8_t i = 0, j; 

	uint_fast32_t s = shape[i] * shape[i + 1] * shape[i + 2] * shape[i + 3]; //head nodes
	
	for (j = 0, i = 2; j < depth() - 1; ++j, i+=2) { //hidden nodes
		s += shape[i-2] * shape[i-1] * shape[i] * shape[i + 1] * shape[i + 2] * shape[i + 3];
	}

	s += shape[i-2] * shape[i-1] * shape[i] * shape[i + 1]; //tail nodes

	return s;
}

std::valarray<float> _ga_nn::neural_net::forard_pass(const _matrix::matrix<float>& data) const
{
	//std::cout << "..--------------------------------------------------\n";

	std::vector<_matrix::matrix<float>> _data = p_head->forward_pass(data);
	//for (auto& e : _data)std::cout << e << std::endl;

	//std::cout << "..--------------------------------------------------\n";

	for (int i = 0; i < depth() - 1; ++i) {
		_data = p_hidden[i]->forward_pass(_data);
		//std::cout << "//--------------------------------------------------\n";
		//for (auto& e : _data)std::cout << e << std::endl;
		//std::cout << "//--------------------------------------------------\n";
	}

	return p_tail->forward_pass(_data);
}

void _ga_nn::neural_net::mutate(float p, float avg, float stddev)
{
	p_head->mutate(p, avg, stddev);
	for (auto& hidden : p_hidden)hidden->mutate(p, avg, stddev);
	p_tail->mutate(p, avg, stddev);
}

//void _ga_nn::neural_net::store(const char* const file_name) const
//{
//	p_head->store(file_name);
//	for (auto& p : p_hidden) {
//		//p->store(file_name);
//	}
//	//p_tail->store(file_name);
//}

const std::pair<neural_net, neural_net> _ga_nn::crossover(const neural_net& net1, const neural_net& net2)
{
	//static std::pair<matrix<T>, matrix<T>> x_crossover(const matrix<T>& mat1, const matrix<T>& mat2)
	
	assert(net1.get_shape() == net2.get_shape());

	const auto _shape = net1.get_shape();
	uint_fast8_t _size = 0;

	uint_fast8_t x, y;
	uint_fast8_t i;

	for (i = 0; i < _shape.size(); i += 2) {
		_size += _shape[i] * _shape[i + 1];
	}

	std::vector<std::pair<node,node>> ret_nodes;
	ret_nodes.reserve(_size);

	//head
	x = net1.get_head()->get_x();
	y = net1.get_head()->get_y();

	const std::vector<_ga_nn::node>& head_nodes_v1 = net1.get_head()->get();
	const std::vector<_ga_nn::node>& head_nodes_v2 = net2.get_head()->get();

	for (i = 0; i < x * y; ++i) {
		ret_nodes.push_back(std::move(_matrix::x_crossover(head_nodes_v1[i].get(), head_nodes_v2[i].get())));
	}

	//hidden
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers1 = net1.get_hidden();
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers2 = net2.get_hidden();

	for (uint_fast8_t j = 0; j < _shape.size() / 2 - 2; ++j) { //iter over hidden layers
		x = hidden_layers1[j]->get_x();
		y = hidden_layers1[j]->get_y();

		for (i = 0; i < x * y; ++i) {
			ret_nodes.push_back(std::move(_matrix::x_crossover(hidden_layers1[j]->get()[i].get(), hidden_layers2[j]->get()[i].get())));
		}
	}

	//tail
	x = net1.get_tail()->get_x();
	y = net1.get_tail()->get_y();

	const std::vector<_ga_nn::node>& tail_nodes_v1 = net1.get_tail()->get();
	const std::vector<_ga_nn::node>& tail_nodes_v2 = net2.get_tail()->get();

	for (i = 0; i < x * y; ++i) {
		ret_nodes.push_back(std::move(_matrix::x_crossover(tail_nodes_v1[i].get(), tail_nodes_v2[i].get())));
	}

	std::vector<node> new_nodes1, new_nodes2;
	new_nodes1.reserve(_size);
	new_nodes2.reserve(_size);

	for (auto it = std::make_move_iterator(ret_nodes.begin()),
		end = std::make_move_iterator(ret_nodes.end()); it != end; ++it) 
	{
		new_nodes1.push_back(it->first);
		new_nodes2.push_back(it->second);
	}

	return std::make_pair(neural_net(_shape, std::move(new_nodes1)), neural_net(_shape, std::move(new_nodes2)));
}

const neural_net _ga_nn::average(const neural_net& net1, const neural_net& net2)
{
	assert(net1.get_shape() == net2.get_shape());

	const auto _shape = net1.get_shape();
	uint_fast8_t _size = 0;

	uint_fast8_t x, y;
	uint_fast8_t i;

	for (i = 0; i < _shape.size(); i += 2) {
		_size += _shape[i] * _shape[i + 1];
	}

	std::vector<node> ret_nodes;
	ret_nodes.reserve(_size);

	//head
	x = net1.get_head()->get_x();
	y = net1.get_head()->get_y();

	const std::vector<_ga_nn::node>& head_nodes_v1 = net1.get_head()->get();
	const std::vector<_ga_nn::node>& head_nodes_v2 = net2.get_head()->get();

	for (i = 0; i < x * y; ++i) {
		ret_nodes.push_back(_matrix::average(head_nodes_v1[i].get(), head_nodes_v2[i].get()));
	}

	//hidden
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers1 = net1.get_hidden();
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers2 = net2.get_hidden();

	for (uint_fast8_t j = 0; j < _shape.size() / 2 - 2; ++j) { //iter over hidden layers
		x = hidden_layers1[j]->get_x();
		y = hidden_layers1[j]->get_y();

		for (i = 0; i < x * y; ++i) {
			ret_nodes.push_back(_matrix::average(hidden_layers1[j]->get()[i].get(), hidden_layers2[j]->get()[i].get()));
		}
	}

	//tail
	x = net1.get_tail()->get_x();
	y = net1.get_tail()->get_y();

	const std::vector<_ga_nn::node>& tail_nodes_v1 = net1.get_tail()->get();
	const std::vector<_ga_nn::node>& tail_nodes_v2 = net2.get_tail()->get();

	for (i = 0; i < x * y; ++i) {
		ret_nodes.push_back(_matrix::average(tail_nodes_v1[i].get(), tail_nodes_v2[i].get()));
	}

	return neural_net(_shape, std::move(ret_nodes));
}

const neural_net _ga_nn::mutation(const neural_net& net)
{
	const std::vector<uint_fast8_t> _shape = net.get_shape();

	uint_fast8_t _size = 0;
	for (uint_fast8_t i = 0; i < _shape.size(); i += 2) {
		_size += _shape[i] * _shape[i + 1];
	}

	std::vector<_ga_nn::node> ret_nodes;
	ret_nodes.reserve(_size);

	uint_fast8_t x, y;

	//head
	x = net.get_head()->get_x();
	y = net.get_head()->get_y();

	std::vector<_ga_nn::node> head_nodes = net.get_head()->get();

	uint_fast8_t i;

	for (i = 0; i < x * y; ++i) {
		head_nodes[i].mutate();
		ret_nodes.push_back(std::move(head_nodes[i]));
	}

	//hidden
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers = net.get_hidden();

	for (uint_fast8_t j = 0; j < _shape.size() / 2 - 2; ++j) { //iter over hidden layers
		x = hidden_layers[j]->get_x();
		y = hidden_layers[j]->get_y();

		std::vector<node> hidden_nodes_j = hidden_layers[j]->get();
		for (i = 0; i < x * y; ++i) {
			hidden_nodes_j[i].mutate();
			ret_nodes.push_back(std::move(hidden_nodes_j[i]));
		}
	}

	//tail
	x = net.get_tail()->get_x();
	y = net.get_tail()->get_y();

	std::vector<_ga_nn::node> tail_nodes = net.get_tail()->get();

	for (i = 0; i < x * y; ++i) {
		tail_nodes[i].mutate();
		ret_nodes.push_back(std::move(tail_nodes[i]));
	}

	return neural_net(_shape, std::move(ret_nodes));
}

std::ostream& _ga_nn::operator<<(std::ostream& os, const neural_net& nn)
{
	const std::vector<uint_fast8_t>& shape = nn.get_shape();
	const std::vector<unsigned short int> _shape(shape.begin(), shape.end());

	assert(_shape.size() % 2 == 0 && _shape.size() >= 6);

	uint64_t s = 0;
	
	int i = 0, j;

	std::cout << "Head shape: " << "(" << _shape[i] << " , " << _shape[i + 1] << ")" << "\n";
	std::cout << "Head size: " << _shape[i] * _shape[i+1] << "\n-------------------\n";

	for (const _ga_nn::node& e : nn.get_head()->get()) {
		std::cout << e.get() << std::endl; 
		s += (uint64_t)e.get().get_m() * (uint64_t)e.get().get_n();
	}

	for (i = 0, j = 2; i < nn.depth() - 1; ++i, j+=2) {
		std::cout << "Hidden " << i << " shape: " << "(" << _shape[i + 2] << " , " << _shape[i + 3] << ")" << "\n";
		std::cout << "Hidden " << i << " size: " << _shape[j] * _shape[j + 1] << "\n-------------------\n";
		for (const _ga_nn::node& e : nn.get_hidden()[i]->get()) {
			std::cout << e.get() << std::endl;
			s += (uint64_t)e.get().get_m() * (uint64_t)e.get().get_n();
		}
	}

	std::cout << "Tail shape: " << "(" << _shape[j] << " , " << _shape[j + 1] << ")" << "\n";
	std::cout << "Tail size: " << _shape[j] * _shape[j + 1] << "\n-------------------\n";
	for (const _ga_nn::node& e : nn.get_tail()->get()) {
		std::cout << e.get() << std::endl;
		s += (uint64_t)e.get().get_m() * (uint64_t)e.get().get_n();
	}

	std::cout << "Net has " << s << " parameters\n\n";

	std::cout << std::flush;
	return os;
}

bool _ga_nn::operator==(const neural_net& nn1, const neural_net& nn2)
{
	assert(nn1.get_shape() == nn2.get_shape());

	const std::vector<uint_fast8_t> _shape = nn1.get_shape();

	uint_fast8_t x, y;
	uint_fast8_t i;

	//head
	const std::vector<_ga_nn::node>& head_nodes_v1 = nn1.get_head()->get();
	const std::vector<_ga_nn::node>& head_nodes_v2 = nn2.get_head()->get();

	if (head_nodes_v1 != head_nodes_v2) { return false; }

	//hidden
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers1 = nn1.get_hidden();
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers2 = nn2.get_hidden();

	for (uint_fast8_t j = 0; j < _shape.size() / 2 - 2; ++j) { //iter over hidden layers
		x = hidden_layers1[j]->get_x();
		y = hidden_layers1[j]->get_y();

		for (i = 0; i < x * y; ++i) {
			if (hidden_layers1[j]->get()[i].get() != hidden_layers2[j]->get()[i].get()) { return false; }
		}
	}

	//tail
	const std::vector<_ga_nn::node>& tail_nodes_v1 = nn1.get_tail()->get();
	const std::vector<_ga_nn::node>& tail_nodes_v2 = nn2.get_tail()->get();

	if (tail_nodes_v1 != tail_nodes_v2) { return false; }

	return true;
}

bool _ga_nn::operator!=(const neural_net& nn1, const neural_net& nn2)
{
	return  !(nn1 == nn2);
}

/// <summary>
/// Measure pseudo l1 distance (d1 distance) between two nets
/// </summary>
/// <param name="net1"> First nnet </param>
/// <param name="net2"> Second nnet </param>
/// <returns> Sum of all the element-wise absolute differences as a double </returns>
double _ga_nn::d1_distance(const _ga_nn::neural_net& net1, const _ga_nn::neural_net& net2)
{
	assert(net1.get_shape() == net2.get_shape());

	const std::vector<uint_fast8_t> _shape = net1.get_shape();

	uint_fast8_t i = 0;

	double _d1_distance = 0; 

	//head distance

	const std::vector<_ga_nn::node>& head1 = net1.get_head()->get();
	const std::vector<_ga_nn::node>& head2 = net2.get_head()->get();

	for (i = 0; i < head1.size(); ++i) {
		_d1_distance += _matrix::d1_distance(head1[i].get(), head2[i].get());
	}

	//hidden distance

	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers1 = net1.get_hidden();
	const std::vector<std::unique_ptr<_ga_nn::hidden_layer>>& hidden_layers2 = net2.get_hidden();

	for (uint_fast8_t j = 0; j < net1.depth() - 1; ++j){

		const std::vector<_ga_nn::node>& nodes_hid1_j = hidden_layers1[j]->get();
		const std::vector<_ga_nn::node>& nodes_hid2_j = hidden_layers2[j]->get();

		for (i = 0; i < nodes_hid1_j.size(); ++i) {
			_d1_distance += _matrix::d1_distance(nodes_hid1_j[i].get(), nodes_hid2_j[i].get());
		}
	}

	//tail distance

	const std::vector<_ga_nn::node>& tail1 = net1.get_tail()->get();
	const std::vector<_ga_nn::node>& tail2 = net2.get_tail()->get();

	for (i = 0; i < tail1.size(); ++i) {
		_d1_distance += _matrix::d1_distance(tail1[i].get(), tail2[i].get());
	}

	return _d1_distance;
}

const _matrix::matrix<double> _ga_nn::population_variability(const std::vector<_ga_nn::neural_net*>& nets)
{
	const uint_fast8_t _size = nets.size();
	_matrix::matrix<double> distance_matrix(_size, _size, double(NAN));

	double d_ij;

	for (uint_fast8_t j = 0; j < _size - 1; ++j) {
		for (uint_fast8_t i = j + 1; i < _size; ++i) {
			d_ij = d1_distance(*nets[j], *nets[i]);
			distance_matrix(j, i) = d_ij;
			distance_matrix(i, j) = d_ij;
		}
	}

	return distance_matrix;
}

const _matrix::matrix<double> _ga_nn::population_variability(const std::vector<std::reference_wrapper<_ga_nn::neural_net>>& nets)
{
	const uint_fast8_t _size = nets.size();
	_matrix::matrix<double> distance_matrix(_size, _size, double(NAN));

	double d_ij;

	for (uint_fast8_t j = 0; j < _size - 1; ++j) {
		for (uint_fast8_t i = j + 1; i < _size; ++i) {
			d_ij = d1_distance(nets[j], nets[i]);
			distance_matrix(j, i) = d_ij;
			distance_matrix(i, j) = d_ij;
		}
	}

	return distance_matrix;
}
