#pragma once
#include "small_GA_matrix.h"
#include "nn_helper_functions.h"
#include "Random.h"
#include <valarray>
#include <vector>
#include <memory>
#include "board.h"


namespace _ga_nn {
	
	class node {
	private:
		uint_fast8_t inputs = 0;
		uint_fast8_t outputs = 0;
		mutable _matrix::matrix<float> weights;

	public:
		inline node() = default;
		node(const uint_fast8_t _inputs, const uint_fast8_t _outputs);
		node(const _matrix::matrix<float>& _weights);

		//mutate every weight with a probability of p, according to a normal distribution (by default): Normal(avg, stddev)
		void mutate(const float p = .5f, const float avg = 0, const float stddev = .2f, float(*randnormal)(float, float) = random::randnormal);

		std::valarray<float> forward_pass(const std::valarray<float>& v);

		//getters
		inline const _matrix::matrix<float>& get() const { return weights; }
		inline const uint_fast8_t get_m() const { return inputs; }
		inline const uint_fast8_t get_n() const { return outputs; }
	
		//weight set, new weights must be of the same size and shape
		void set(const _matrix::matrix<float>& new_weights);
		void store(const char* const file_name) const;

		friend bool operator==(const node& n1, const node& n2);
		friend bool operator!=(const node& n1, const node& n2);

		inline ~node() = default;
	};

	inline bool operator==(const node& n1, const node& n2) {
		return n1.weights == n2.weights;
	}

	inline bool operator!=(const node& n1, const node& n2) {
		return n1.get() == n2.get();
	}

	class layer {
	protected:
		uint_fast8_t x, y; // shape of the layer
		uint_fast8_t m, n; // shape of the matrix in each node
		mutable std::vector<node> nodes;
	public:
		layer(const uint_fast8_t _y, const uint_fast8_t _x, const uint_fast8_t m, const uint_fast8_t n);
		layer(const uint_fast8_t _y, const uint_fast8_t _x, const std::vector<node>& _nodes);

		inline const std::vector<node>& get() const { return nodes; }
		inline const uint_fast8_t get_x() const { return x; }
		inline const uint_fast8_t get_y() const { return y; }
		inline const uint_fast8_t get_m() const { return m; }
		inline const uint_fast8_t get_n() const { return n; }

		inline void set(uint_fast8_t p, const _matrix::matrix<float> mat) { nodes[p] = _ga_nn::node(mat); }

		void mutate(float p, float avg, float stddev);

		inline ~layer() = default;
	};

	class in_layer : public layer {
	private:
		const uint_fast8_t next_x, next_y;

	public:
		in_layer(const uint_fast8_t _y, const uint_fast8_t _x, const uint_fast8_t _next_y, const uint_fast8_t _next_x);
		in_layer(const uint_fast8_t _y, const uint_fast8_t _x, const uint_fast8_t _next_y, const uint_fast8_t _next_x, const std::vector<node>& _nodes);

		inline const uint_fast8_t get_next_x() const { return next_x; }
		inline const uint_fast8_t get_next_y() const { return next_y; }

		std::vector<_matrix::matrix<float>> forward_pass(const _matrix::matrix<float>& data,
			std::valarray<float>(*foo)(const std::valarray<float>&) = _nn_func::equality) const;

		inline ~in_layer() = default;
	};

	class hidden_layer : public layer {
	private:
		const uint_fast8_t next_x, next_y;
		const uint_fast8_t prev_x, prev_y;

	public:
		hidden_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x,
			const uint_fast8_t _next_y, const uint_fast8_t _next_x);
		hidden_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x,
			const uint_fast8_t _next_y, const uint_fast8_t _next_x, const std::vector<node>& _nodes);

		inline const uint_fast8_t get_next_x() const { return next_x; }
		inline const uint_fast8_t get_next_y() const { return next_y; }
		inline const uint_fast8_t get_prev_x() const { return prev_x; }
		inline const uint_fast8_t get_prev_y() const { return prev_y; }

		std::vector<_matrix::matrix<float>> forward_pass(const std::vector<_matrix::matrix<float>>& data,
			std::valarray<float>(*foo)(const std::valarray<float>&) = _nn_func::relu) const;

		inline ~hidden_layer() = default;
	};

	class out_layer : public layer {
	private:
		const uint_fast8_t prev_x, prev_y;

	public:
		out_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x);
		out_layer(const uint_fast8_t _prev_y, const uint_fast8_t _prev_x, const uint_fast8_t _y, const uint_fast8_t _x, const std::vector<node>& _nodes);

		inline const uint_fast8_t get_prev_x() const { return prev_x; }
		inline const uint_fast8_t get_prev_y() const { return prev_y; }

		std::valarray<float> forward_pass(const std::vector<_matrix::matrix<float>>& data,
			std::valarray<float>(*foo)(const std::valarray<float>&) = _nn_func::softmax) const;

		inline ~out_layer() = default;
	};

	class neural_net {
	private:
		mutable std::unique_ptr<in_layer> p_head;
		mutable std::vector<std::unique_ptr<hidden_layer>> p_hidden;
		mutable std::unique_ptr<out_layer> p_tail;
		std::vector<uint_fast8_t> shape;
	public:
		neural_net(const std::vector<uint_fast8_t>& v);
		neural_net(const std::vector<uint_fast8_t>& v, const std::vector<node>& _nodes);
		neural_net(const neural_net& net);

		//cont getters
		inline const std::unique_ptr<in_layer>& get_head() const { return p_head; }
		inline const std::vector<std::unique_ptr<hidden_layer>>& get_hidden() const { return p_hidden; }
		inline const std::unique_ptr<out_layer>& get_tail() const { return p_tail; }
		inline const std::vector<uint_fast8_t> get_shape() const { return shape; }

		//non-const getters
		inline std::unique_ptr<in_layer>& get_head() { return p_head; }
		inline std::vector<std::unique_ptr<hidden_layer>>& get_hidden() { return p_hidden; }
		inline std::unique_ptr<out_layer>& get_tail() { return p_tail; }

		inline const uint_fast8_t depth() const { return p_hidden.size() + 1; }
		const uint_fast16_t parameter_count() const;

		std::valarray<float> forard_pass(const _matrix::matrix<float>& data) const;

		void mutate(float p = 0.1f, float avg = 0.f, float stddev = 0.1f);
		void store(const char* const file_name) const;

		//friend operators
		friend std::ostream& operator<<(std::ostream& os, const neural_net& nn);
		friend bool operator == (const neural_net& nn1, const neural_net& nn2);
		friend bool operator != (const neural_net& nn1, const neural_net& nn2);
	};

	bool operator == (const neural_net& nn1, const neural_net& nn2);
	bool operator != (const neural_net& nn1, const neural_net& nn2);

	const neural_net mutation(const neural_net& net);

	//x-crossorver element-wise the nodes of the net
	const std::pair<neural_net, neural_net> crossover(const neural_net& net1, const neural_net& net2);

	const neural_net average(const neural_net& net1, const neural_net& net2);

	std::ostream& operator<<(std::ostream& os, const neural_net& nn);

	double d1_distance(const _ga_nn::neural_net& net1, const _ga_nn::neural_net& net2);

	const _matrix::matrix<double> population_variability(const std::vector<_ga_nn::neural_net*>& nets);
	const _matrix::matrix<double> population_variability(const std::vector<std::reference_wrapper<_ga_nn::neural_net>>& nets);
}