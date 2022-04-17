#include "board.h"
#include "c4minimax.h"
#include "ga_nn.h"
#include "c4_evo_environment.h"
#include "stopwatch.h"
#include "c4_brain.h"
#include "Random.h"
#include "ga_static_nn.h"
#include "ga_static_matrix.h"

int main0() {
	//probar operator() board

	//mutate y tal

	std::ios_base::sync_with_stdio(false);
	stopwatch temp;

	std::vector<uint_fast8_t> _shape({2, 2, 3, 3, 3, 3, 1, 1});
	//std::vector<uint_fast8_t> _shape({1, 1, 3, 3, 1, 1 });

	_ga_nn::neural_net net1(_shape), net2(_shape);

	std::cout << net1 << std::endl << net2 << std::endl;


	std::cout << "-------------------------------\n---------------------------------\n";
	std::cout << "-------------------------------\n---------------------------------\n";


	const auto new_net_pair = _ga_nn::crossover(net1, net2);

	std::cout << new_net_pair.first << "#############################################\n#############################################\n" <<
		new_net_pair.second << std::endl;

	std::cout << net1 << std::endl << net2 << std::endl;
	std::cout << _ga_nn::average(net1, net2);

	return EXIT_SUCCESS;
}

int main1() {
	std::ios_base::sync_with_stdio(false);
	stopwatch temp;

	std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 1, 1 });

	_ga_nn::neural_net net1(_shape);

	_ga_nn::neural_net net2 = _ga_nn::mutation(net1);

	_ga_nn::neural_net net3(_shape);

	_ga_nn::neural_net net4(net1);

	std::cout << net1 << std::endl << "#############################################\n#############################################\n\n\n" 
			  << net2 << std::endl;

	std::cout << "distance between nets 1 & 1: " << _ga_nn::d1_distance(net1, net1) << std::endl;
	std::cout << "distance between nets 1 & 2: " << _ga_nn::d1_distance(net1, net2) << std::endl;
	std::cout << "distance between nets 1 & 3: " << _ga_nn::d1_distance(net1, net3) << std::endl;
	std::cout << "distance between nets 1 & 4: " << _ga_nn::d1_distance(net1, net4) << std::endl;

	std::cout << "net1 == net1: " << ((net1 == net1) ? "true" : "false") << std::endl;
	std::cout << "net1 == net2: " << ((net1 == net2) ? "true" : "false") << std::endl;
	std::cout << "net1 == net3: " << ((net1 == net3) ? "true" : "false") << std::endl;
	std::cout << "net1 == net4: " << ((net1 == net4) ? "true" : "false") << std::endl;

	std::cout << _ga_nn::population_variability({ net1, net2, net3, net4 });

	return EXIT_SUCCESS;
}

int main2() 
{
	std::vector<uint_fast8_t> _shape({ 2, 2, 2, 2, 1, 1 });

	_c4_brain::c4_brain b1(_shape);
	_c4_brain::c4_brain b2(_shape);

	const auto ret_bs = _c4_brain::crossover(b1, b2);

	std::cout << b1 <<
		b2 << 
		ret_bs.first <<
		ret_bs.second << std::endl;

	return EXIT_SUCCESS;
}

int main3() {
	//random::init();

	_matrix::matrix<double> a;
	_matrix::matrix<double> b;

	{
		stopwatch s;

		std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 3, 3, 1, 1 });

		_ga_nn::neural_net net1(_shape);
		_ga_nn::neural_net net2(_shape);
		_ga_nn::neural_net net3(_shape);
		_ga_nn::neural_net net4(_shape);

		_c4_brain::c4_brain b1(net1);
		_c4_brain::c4_brain b2(net2);
		_c4_brain::c4_brain b3(net3);
		_c4_brain::c4_brain b4(net4);

		a = _ga_nn::population_variability({ net1, net2, net3, net4 });
		b = _c4_brain::population_variability({ b1, b2, b3, b4 });
	}

	std::cout << "\n";
	std::cout << a << std::endl;
	std::cout << b << std::endl;

	return EXIT_SUCCESS;
}

int main4() {
	//std::cout << "here\n";

	std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 3, 3, 1, 1 });
	
	_c4_brain::c4_brain b1(_shape);
	_c4_brain::c4_brain b2(_shape);

	std::cout << b1 << "\n" << b2 << std::endl;

	np_ai_play_ai(&b1, &b2, 3, 3);

	return EXIT_SUCCESS;
}

int main5() {
	random::init();

	stopwatch s;

	std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 2, 2, 1, 2 });

	_c4_brain::c4_brain b1(_shape);
	//_c4_brain::c4_brain b2(_shape);

	//std::cout << b1 << "\n"; //<< b2 << std::endl;

	std::array<uint_fast8_t, 42> arr{
	0,0,0,2,0,0,0,
	0,0,0,1,0,0,0,
	0,0,2,2,0,0,0,
	0,0,0,1,0,0,0,
	0,0,1,2,2,0,0,
	0,0,2,1,1,2,1,
	};

	board b(arr);

	int n = 20;
	std::vector<int> rets(n);
 
	{
		stopwatch s1;
		for (int i = 0; i < n; ++i) {
			b1.mutate(0.1f, random::randfloat() - 0.5f, random::randfloat() * 0.5f);
			rets[i] = b1.weigh(b, 0);
		}
	}

	std::cout << "\n" << (int)b1.weigh(b, 0) << std::endl;

	std::cout << *std::max_element(rets.begin(), rets.end()) << " " << *std::min_element(rets.begin(), rets.end()) << std::endl;

	return EXIT_SUCCESS;
}

int main6() {
	random::init();
	stopwatch s;
	//std::cout << "here\n";

	std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 3, 3, 1, 2 });

	_c4_brain::c4_brain b1(_shape);
	_c4_brain::c4_brain b2(_shape);

	//std::cout << b1 << "\n" << b2 << std::endl;

	int n = 50;
	short int res;
	std::vector<int> vres(3);

	for (int i = 0; i < n; ++i) {
		res = np_ai_play_ai(&b1, &b2, 2, 3, i == n-1);
		
		switch (res) {
		case 1:
			++vres[1];
			break;
		case 0:
			++vres[0];
			break;
		case -1:
			++vres[2];
			break;
		}

		if(1)
			b2.mutate();

		else if(res == 0) 
			b2.mutate();

		else {
			if (random::randint(0, 1))
				b1.mutate();
			else
				b2.mutate();
		}

		if (i % 100 == 0)std::cout <<"Iteration: " << i << "\n";
	}

	std::cout << "b1 wins: " << vres[0] << "\nb2 wins: " << vres[1] << "\nties: " << vres[2] << std::endl;

	return EXIT_SUCCESS;
}

int main7() {
	std::ios_base::sync_with_stdio(false);
	stopwatch s;
	random::init();

	std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 3, 3, 1, 2 });

	int n = 15;

	std::vector<_c4_brain::c4_brain> brains;
	brains.reserve(n);

	for (int i = 0; i < n; ++i) {
		brains.emplace_back(_shape);
	}

	std::cout << "Brains.size():  " << brains.size() << "\n" <<
		"Brain ID: " << _c4_brain::c4_brain::get_current_ID() << std::endl;

	std::vector<_c4_brain::c4_brain*> gen;
	gen.reserve(n);

	for (int i = 0; i < n; ++i) {
		gen.push_back(&brains[i]);
	}

	{
		stopwatch t;
		_c4_evo_env::tournament(gen, gen, 3, 3);
	}

	std::vector<float> winrates;
	winrates.reserve(n);

	for (auto b : gen) {
		b->print_stats();
		winrates.push_back(b->get_winrate());
	}

	std::cout << "Max winrate: " << *std::max_element(winrates.begin(), winrates.end()) << std::endl;

	std::cout << "Brain ID: " << _c4_brain::c4_brain::get_current_ID() << std::endl;

	return EXIT_SUCCESS;
}

int main8() {
	stopwatch s;
	random::init();

	std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 3, 3, 1, 2 });

	int n = 5;

	std::vector<_c4_brain::c4_brain*> gen;

	gen.reserve(n);

	std::vector<_c4_brain::c4_brain> brains1(n, _shape), brains2(n, _shape);

	std::cout << "Brains.size():  " << brains1.size() << "\n" << _c4_brain::c4_brain::get_current_ID() << std::endl;

	for (int i = 0; i < n; ++i) {
		gen.push_back(&brains1[i]);
	}

	_c4_evo_env::tournament(gen, gen, 3, 3);

	for (auto b : gen) {
		b->print_stats();
	}

	for (const auto p : gen) {
		for (const auto c : gen) {
			std::cout << np_ai_play_ai(p, c, 3, 3) << " " << np_ai_play_ai(c, p, 3, 3) << "\n";
		}
	}

	std::cout << _c4_brain::c4_brain::get_current_ID() << std::endl;

	return EXIT_SUCCESS;
}

int main9() {
	stopwatch s0;

	random::init();

	int i, j, jj, n = 10000, m = 20, s;
	double d = 0, dd, max = -INFINITY, min = INFINITY;
	float f1, f2;
	std::vector<double> w(m);

	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j) {
			s = 0;
			for (jj = 0; jj < m; ++jj) {
				f1 = random::randfloat();
				f2 = random::randfloat();

				if (f1 > f2) {
					++s;
				}
			}
			w[j] += (double(s) / m);
		}

		dd = *std::max_element(w.begin(), w.end());

		if (dd < min) min = dd;
		if (dd > max) max = dd;

		d += dd;
		for (auto& e : w) e = 0;
	}

	std::cout << (d / n) << std::endl;
	std::cout << "Min: " << min << "\nMax: " << max << std::endl;
	return EXIT_SUCCESS;
}

int main11() {

	int pop_size = 5, epochs = 5;
	const std::vector<uint_fast8_t> _shape({ 6, 7, 3, 3, 3, 3, 1, 2 });

	random::init();
	//utiity
	uint_fast8_t i;
	std::array<uint_fast8_t, 2> gi({ 0,1 }); //gen index
	std::vector<float> winratios(pop_size);

	//randomly initialize the first generation.
	//brain vectors a and b will hold previous or current generation depending on the epoch

	std::array<std::vector<_c4_brain::c4_brain>, 2> brains;

	for (std::vector<_c4_brain::c4_brain>& e : brains) {
		e.reserve(pop_size);
	}

	for (i = 0; i < pop_size; ++i) {
		brains[gi[0]].emplace_back(_shape);
	}

	std::array<std::vector<_c4_brain::c4_brain*>, 2> gen; //gen[bi[0]] is prev, gen[bi[1]] is current

	for (std::vector<_c4_brain::c4_brain*>& e : gen) {
		e.reserve(pop_size);
	}

	//only for first gen, cur gen and prev gen are the same 
	for (i = 0; i < pop_size; ++i) {
		gen[gi[0]].push_back(&brains[gi[0]][i]);
		gen[gi[1]].push_back(&brains[gi[0]][i]);
	}

	for (i = 0; i < epochs - 1; ++i) {
		_c4_evo_env::tournament(gen[gi[0]], gen[gi[1]], 3, 3, 1);
		break;
	}
	return EXIT_SUCCESS;
}

int main12() {
	stopwatch s;

	std::vector<uint_fast8_t> _shape = { 6,7,3,3,3,3,1,2 };
	const char* const file_name("../_c4_brains/prueba1.txt");
	_c4_brain::c4_brain b(_shape);
	b.store(file_name);

	auto d(b);

	_c4_brain::c4_brain e(_shape);

	auto c = _c4_brain::read(file_name);

	auto a(b);
	a.mutate();

	std::cout << _c4_brain::population_variability({ a, b, c, d, e });

	a.print_stats();
	return EXIT_SUCCESS;
}

int main13() {
	stopwatch s;
	const char* const file_name1("../_c4_brains/prueba1.txt");
	const char* const file_name2("../_c4_brains/prueba2.txt");

	auto b = _c4_brain::read(file_name1);
	auto c = _c4_brain::read(file_name2);
	std::cout << _c4_brain::population_variability({ b,c });

	return EXIT_SUCCESS;
}

int main14() {
	stopwatch s;
	random::init();

	const std::vector<uint_fast8_t> _shape({ 6, 7, 4, 4, 1, 2 });

	auto b = _c4_evo_env::simulate_evolution(15, 3, 100, _shape, 2, 3, 5, 2, 0.4, 10);

	std::cout << "\nBrain ID: " << _c4_brain::c4_brain::get_current_ID() << std::endl;

	//np_i_play_ai(&b, 5, random::randint(0,1));
	const char* const file_name("../_c4_brains/674412_25_7_500_3_3_4_04__08_11_21.txt");

	b.store(file_name);

	//std::cout << _c4_brain::population_variability({ c, b }) << std::endl;

	return EXIT_SUCCESS;
}

int main15() {
	//random::init();
	stopwatch s;
	const char* const file_name("../_c4_brains/674412_15_3_100.txt");
	const std::vector<uint_fast8_t> _shape({ 6, 7, 4, 4, 1, 2 });
	auto b = _c4_brain::read(file_name);
	_c4_brain::c4_brain b1(_shape);

	std::cout << b << b1 << std::endl;

	std::vector<_c4_brain::c4_brain*> p_gen = { &b, &b1 };

	//std::vector<_c4_brain::c4_brain> gen;
	//gen.reserve(15);
	//for (int i = 0; i < 15; ++i) {
	//	gen.emplace_back(_shape);
	//}
	 
	//auto c = _c4_evo_env::simulate_evolution(_c4_evo_env::breed_new_gen(p_gen, 15, 2, 0.8), 3, 100, 2, 3, 5, 2, 0.3, 10);
	auto c = _c4_evo_env::simulate_evolution(15, 3, 100, _shape, 2, 3, 5, 2, 0.3, 10);

	const char* const ret_file_name("../_c4_brains/674412_15_3_100_V2.txt");
	c.store(ret_file_name);

	return EXIT_SUCCESS;
}

int main16() {

	const char* const file_name1("../_c4_brains/67333312_15_3_100_2_3_2_02__26_10_21.txt");
	auto b1 = _c4_brain::read(file_name1);

	const char* const file_name2("../_c4_brains/67333312_15_3_100_2_3_2_02__26_10_21.txt");
	auto b2 = _c4_brain::read(file_name2);

	np_ai_play_ai(&b1, &b2, 7, 7, 0, 0, 1);

	return EXIT_SUCCESS;
}

int main17() {
	stopwatch s;

	random::init();
  ga_sm::static_matrix<int, 15, 10> sm{};

	sm.fill<random::randint>(0 ,10);

	auto psm = &sm;
	{
		stopwatch s1;
		auto sm2(sm);
	}

	std::cout << psm->operator()(0,0);
	std::cout << (*psm)(0, 0);

	std::cout << sm << std::endl;
	std::cout << sm[0];

	return EXIT_SUCCESS;
}

int main18() {
	using namespace ga_sm;

	random::init();

	static_matrix<float, 10, 10> sm1;
	sm1.fill<random::randfloat>();

	static_matrix<float, 10, 10> sm2;
	sm2.fill<random::randfloat>();

	std::cout << sm1 << std::endl;
	std::cout << sm2 << std::endl;

	sm1.rescale_L_1_1_norm();

	std::cout << sm1 << std::endl;

	

	return EXIT_SUCCESS;
}

int main19() {
	using namespace ga_sm;
	random::init();
	const int N = 20;
	
	//std::cout << sm1 << std::endl;

	//sm1.fill<random::randint>(0, 20);
	//sm2.fill<random::randint>(0, 20);
	//sm3.fill<random::randint>(0, 20);

	{
		std::cout << "stack\n";
		stopwatch s;

		static_matrix<float, N, N> sm1;
		static_matrix<float, N, N> sm2;
		//sm3;

		sm1.fill<random::randfloat>();
		sm2.fill<random::randfloat>();
		static_matrix<float, N, N> sm3 = matrix_mul(sm1, sm2);

		std::cout << sm3(0, 0) << std::endl;
		
	}

	//std::cout << sm1 << "\n" << sm2 << std::endl;
	//std::cout << sm3 << std::endl;

	//_matrix::matrix<float> m1(random::randfloat, 0, 24, 25);
	//_matrix::matrix<float> m2(random::randfloat, 0, 25, 23);
	//_matrix::matrix<float> m3(24, 23);

	//{
	//	stopwatch s;
	//	m3 = _matrix::dot(m1, m2);
	//}


	//std::cout << sm1 << std::endl;
	//std::cout << sm2 << std::endl;
	//std::cout << sm3 << std::endl;

	return EXIT_SUCCESS;
}

int main20() {
	using namespace ga_sm;
	{
		std::cout << "randinit\n";
		stopwatch s;
		random::init();
	}
	const int N = 20;

	{
		std::cout << "heap\n";
		stopwatch s;
		
		auto sm1 = std::make_unique<static_matrix<float, N, N>>();
		auto sm2 = std::make_unique<static_matrix<float, N, N>>();

		sm1->fill<random::randfloat>();
		sm2->fill<random::randfloat>();
		auto sm3 = std::make_unique<static_matrix<float, N, N>>
			(matrix_mul(*sm1, *sm2));

		std::cout << (*sm3)(0, 0) << std::endl;

	}
	//std::cout << sm1 << std::endl;
	return EXIT_SUCCESS;
}

int main21() {
	using namespace ga_sm;
	random::init();
	const int N = 5;

	//std::cout << sm1 << std::endl;

	static_matrix<float, N, N+1> sm1;
	
	sm1.fill<random::randfloat>();
	std::cout << sm1 << std::endl;

	if (RREF(sm1)) {
		std::cout << sm1;
	}

	return EXIT_SUCCESS;
}

constexpr bool main22() {

	using namespace ga_sm;
	
	constexpr size_t N = 4;

	
	static_matrix<int, N, N> sm1{};
	static_matrix<double, N, N> sm2;

	sm1 = { 1,2,3,4,5,6,7,8,9,10,11,1,13,4,5,6 };
	sm2 = cast_to<double>(sm1);

	bool b{0};

  auto [invertible, sm3] = inverse(sm2);

	if (invertible) {
		//auto a = PII_LUDecomposition(sm2);
		//std::cout << matrix_mul(sm2, sm3) << std::endl;
		b = (matrix_mul(sm2, sm3) == identity_matrix<double, N>());
	}

	return b;
}


#include <type_traits>
bool main23() {
	using namespace ga_sm;

	using T = static_matrix<float, 5, 5>;

	std::cout << std::is_trivial<T>::value;
	std::cout << std::is_trivially_copyable<T>::value;
	std::cout << std::is_standard_layout<T>::value;

	using A = static_matrix<int, 5, 5>;

	std::cout << std::is_trivial<A>::value;
	std::cout << std::is_trivially_copyable<A>::value;
	std::cout << std::is_standard_layout<A>::value;


	return 0;
}

bool main24() {

	using namespace ga_sm;

	constexpr size_t N = 4;


  static_matrix<int, N, N> sm1;
	static_matrix<double, N, N> sm2{}, sm11{}, sm12{};

	sm1 = { 1,2,3,4,5,6,7,8,9,10,11,1,13,4,5,6 };
  sm12 = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

  std::cout << matrix_average<double, N, N>(sm12, sm12, sm11) << std::endl;
	sm2 = cast_to<double>(sm1);

	bool b{ 0 };
  std::cout << sm2 << std::endl;

  auto [invertible, sm3] = inverse(sm2);

	if (invertible) {
		std::cout << sm3 << std::endl;
		std::cout << determinant(sm1) << "\n";
		std::cout << matrix_mul(sm2, sm3) << std::endl;
		b = (matrix_mul(sm2, sm3) == identity_matrix<double, N>());
	}

	return b;
}

int matrix_mul_bench() {
  using namespace ga_sm;
  constexpr size_t N = 50;
  static_matrix<float, N, N> sm1{};
  static_matrix<float, N, 1> sm2{}, sm3{}, sm4{};

  {
    stopwatch s;
    sm1.fill<random::randfloat>();
    sm2.fill<random::randfloat>();
  }
  {
    stopwatch s;
    sm3 = matrix_mul(sm1, sm2);
  } 
  {
    stopwatch s;
    sm4 = matrix_vector_mul_float_avx(sm1, sm2);
  }

  std::cout << (sm3 == sm4) << "\n";

  return 0;
}


int main25() {

  //random::init();
  stopwatch s0;
  using namespace ga_sm_nn;
  //constexpr ga_sm_nn::Matrix_Size a1{ 2,2 };
  //constexpr ga_sm_nn::Matrix_Size a2{ 4,4 };

  //ga_sm_nn::static_neural_net<float, a1, a2 , a2> nn{};

  ga_sm::static_matrix<float, 6, 7> in{};
  in.fill<random::randfloat>();

  //ga_sm_nn::layer<float, ga_sm_nn::Layer_Shape{ 3,1 }, 6, 4> l1{};

  //l1.initialize<random::randnormal, 0, 1>();

  //std::cout << in << std::endl;
  //std::cout << l1;

  constexpr Layer_Shape a1{ 1,1 };
  constexpr Layer_Shape a2{ 2,2 };
  constexpr Layer_Shape a21{ 2,1 };
  constexpr Layer_Shape a3{ 3,3 };
  constexpr Layer_Shape a4{ 4,4 };
  constexpr Layer_Shape a6{ 6,6 };
  constexpr Layer_Shape a67{ 6,7 };
  constexpr Layer_Shape a10{ 10,10 };



  static_neural_net<float, a67, a6, a6, a6, a3, a21> nn1{}, nn2{};
  //
  //

  std::array<decltype(nn1), 2>a{ nn1,nn2 };
  std::cout << "L11:\n" << population_variability(std::array{nn1, nn2});


  nn1.init<random::randnormal, 0, 1>();
  nn2.init<random::randnormal, 0, 1>();

  nn1.print_net();
  nn2.print_net();

  std::cout << "L11:\n" << population_variability(std::array{ nn1, nn2 });

  auto ret = x_crossover(nn1, nn2);

  //nn1.print_net();
  //nn2.print_net();
  //ret.first.print_net();
  //ret.second.print_net();

  decltype(nn2) nn3{};

  nn2.store("prueba0.txt");
  nn3.load("prueba0.txt");

  nn3.init<random::randnormal, 0, 1>();

  std::cout << "L11:\n" << population_variability(std::array{ nn1, nn2, nn3, ret.first, ret.second });

  std::cout << sizeof(decltype(nn3)) << '\n';
  std::cout << nn3.parameter_count() << "\n";
  std::cout << ((double)sizeof(decltype(nn3)) / (double)nn3.parameter_count()) << "\n";


  //nn3.print_net();

  //std::cout << nn3.parameter_count() << "\n";
  std::cout << "Output: \n" << nn2.forward_pass(in) << '\n';
  double sum = 0;
  {
    stopwatch s1;
    for (size_t i = 0; i < 1e4; ++i) {
      sum += static_cast<double>(nn2.forward_pass(in)(0,0));
      in.mutate<random::randfloat>(0, 1, 0.2, 0.4);
    }
  }
  std::cout << sum << '\n';


  _ga_nn::neural_net net1({ 6,7,4,4,3,3,2,2,2,1 });

  
  //std::cout << net1.parameter_count() << "\n";
  //_matrix::matrix<float> m(6, 7);
  //{
  //  stopwatch s2;
  //  net1.forard_pass(m);
  //}

  return EXIT_SUCCESS;
}


int main() {
  //main14();
  //main25();
  main24();
  using namespace ga_sm;
  //constexpr size_t N = 20;

  //auto sm1 = std::make_unique<static_matrix<float, N, N>>();
  //auto sm2 = std::make_unique<static_matrix<float, N, N>>();

  //sm1->fill<random::randfloat>();
  //sm2->fill<random::randfloat>();
  //
  //auto sm3 = std::make_unique<static_matrix<float, N, N>>(matrix_mul(*sm1.get(), *sm2.get()));

  //std::cout << sm3->operator()(0, 0) << "\n";

  matrix_mul_bench();
  std::cout << ".......\n";


  constexpr size_t N = 50, M = 40, K = 5e5;
  static_matrix<float, M, N> sm1{};
  static_matrix<float, N, 1> sm2{};
  static_matrix<float, M, 1> a{}, b{};

  sm1.fill<random::randfloat>();
  sm2.fill<random::randfloat>();

  float randn{};
  {
    stopwatch s1;
    for (size_t i = 0; i < K; ++i) {
      randn += random::randfloat();
    }
  }
  float randn2{};
  {
    stopwatch s1;
    for (size_t i = 0; i < K; ++i) {
      randn2 += random::randfloat();
    }
  }

  std::cout << randn << "\n" << randn2 << "\n";

  //std::cout << sm1 << '\n' << sm2 << '\n';
  float ad{}, bd{};
  {
    stopwatch s;
    for (size_t i = 0; i < K; ++i) {
      sm1(random::randint(0,4), 0) = random::randfloat();
      sm2(1, 1) = random::randfloat();
      a = matrix_mul(sm1, sm2);
      ad += a(0, 0);
    }
  }
  {
    stopwatch s;
    for (size_t i = 0; i < K; ++i) {
      sm1(random::randint(0, 4),0) = random::randfloat();
      sm2(1,1) = random::randfloat();
      b = matrix_vector_mul_float_avx(sm1, sm2);
      bd += b(0, 0);
    }
  }

  sm1.fill<random::randfloat>();
  sm2.fill<random::randfloat>();
  a = matrix_mul(sm1, sm2);
  b = matrix_vector_mul_float_avx(sm1, sm2);

  std::cout << ad << " " << bd << "\n";
  std::cout << nearly_equals(a, b) << std::endl;

  return EXIT_SUCCESS;
}