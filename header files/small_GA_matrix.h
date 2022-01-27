#pragma once
#include <vector>
#include <iostream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <valarray>
#include "Random.h"
#include <numeric>
#include <algorithm>
#include <assert.h>

#ifndef NDEBUG
#define NDEBUG 1
#endif // !NDEBUG


//----------------------------------------------------------------------
//
// header only declarations due to templeted class
// 
//----------------------------------------------------------------------


//row-major small dense matrix implementation with basic functionality for genetic algorithms 
namespace _matrix {

	template<class T>
	class matrix {
	private:
		uint_fast8_t m;
		uint_fast8_t n;
		std::vector<std::valarray<T>> data;

	public:
		//default ctor
		matrix(const uint_fast8_t _m = 1, const uint_fast8_t _n = 1, const T _default = T(0)) :
			m{ _m }, n{ _n } {
			assert(m != 0 && n != 0);
			data.resize(m);
			for (int i = 0; i < m; ++i) {
				data[i].resize(n, _default);
			}
		}

		matrix<T>(const matrix<T>&) = default;
		matrix<T>(matrix<T>&&) = default;

		matrix(T(*foo)(), const float offset = 0, const uint_fast8_t _m = 1, const uint_fast8_t _n = 1) :
			m{ _m }, n{ _n } {
			assert(m != 0 && n != 0);
			data.resize(m);
			for (int i = 0; i < m; ++i) {
				data[i].resize(n);
				for (auto& e : data[i]) { e = foo() + offset; }
			}
		}

		matrix(const std::vector<std::valarray<T>>& _data) :
			m{ (uint_fast8_t)_data.size() }, n{ (uint_fast8_t)_data[0].size() }, data{ _data }{

		}

		//row vector ctor
		matrix(const std::valarray<T>& _data) :
			m{ (uint_fast8_t)_data.size() }, n{ 1 }, data{ std::vector<std::valarray<T>>(1, std::valarray<T>(_data)) }{

		}

		matrix(const matrix<T>& mat1, const matrix<T>& mat2) :
			m{ uint_fast8_t(mat1.get_m() + mat2.get_m()) }, n{ mat1.get_n() }{
			assert(mat1.get_n() == mat2.get_n());

			data.resize(m);
			const uint_fast8_t m1 = mat1.get_m();
			const uint_fast8_t m2 = mat2.get_m();

			const auto _data1 = mat1.get();
			const auto _data2 = mat2.get();

			for (int i = 0; i < m1; ++i) {
				data[i] = _data1[i];
			}

			for (int i = m1; i < m2 + m1; ++i) {
				data[i] = _data2[i];
			}
		}

		matrix<T>& operator=(const matrix<T>& other) {
			if (this == &other)
				return *this;
			m = other.get_m();
			n = other.get_n();
			data = other.get();
			return *this;
		}

		//read-write 2d index operator
		inline T& operator()(const uint_fast8_t j, const uint_fast8_t i) {
			assert(j < m&& i < n);
			return data[j][i];
		}

		//read-only 2d index operator
		inline const T& operator()(const uint_fast8_t j, const uint_fast8_t i) const {
			assert(j < m&& i < n);
			return data[j][i];
		}

		//read only row index operator
		inline const std::valarray<T>& operator[](const uint_fast8_t j) const {
			assert(j < m);
			return data[j];
		}

		//read-write row index operator
		inline std::valarray<T>& operator[](const uint_fast8_t j) {
			assert(j < m);
			return data[j];
		}

		//read only slicing operator
		const matrix<T> operator[](const std::array<uint_fast8_t, 2> slice_index) const {
			uint_fast8_t a = slice_index[0], b = slice_index[1];
			assert(a < m&& b <= m && a < b);

			std::vector<std::valarray<T>> ret_data(b - a, std::valarray<T>(n));
			for (uint_fast8_t i = a; i < b; ++i) {
				ret_data[i - a] = data[i];
			}
			return matrix<T>(ret_data);
		}

		//--------
		//utility
		//--------

		void populate(T(*foo)(), const T scale = 1, const T offset = 0) {
			for (int i = 0; i < m; ++i)
				for (int j = 0; j < n; ++j)
					data[i][j] = foo() * scale + offset;
		}

		//mutate the elements of the matrix
		void mutate(T(*randnormal)(float, float), const float p, const float avg = 0, const float stddev = 1) {
			for (int i = 0; i < m; ++i)
				for (int j = 0; j < n; ++j)
					data[i][j] += randnormal(avg, stddev) * (random::randfloat() < p);
		}

		//read only access to the data
		inline constexpr const std::vector<std::valarray<T>>& get() const { return data; }

		inline constexpr const uint_fast8_t get_m() const { return m; }

		inline constexpr const uint_fast8_t get_n() const { return n; }

		//write access to data
		inline void set(const std::vector<std::valarray<T>>& _data) {
			data = _data;
		}

		const matrix<T> operator *(const matrix<T>& other) const {//elementwise multiplication
			assert(n == other.get_n() && m == other.get_m());

			std::vector<std::valarray<T>> ret_data(m, std::valarray<T>(n));

			for (int j = 0; j < m; ++j) {
				ret_data[j] = data[j] * other[j];
			}
			return ret_data;
		}

		const matrix<T> operator *(const T& p) const {//elementwise multiplication

			std::vector<std::valarray<T>> ret_data(m, std::valarray<T>(n));

			for (int j = 0; j < m; ++j) {
				ret_data[j] = data[j] * p;
			}
			return ret_data;
		}

		template<class T>
		friend std::ostream& operator<<(std::ostream& os, const matrix<T>& mat);

		void store_csv(const char* const file_name) const {
			//std::ofstream out(file_name);

			std::fstream out;
			out.open(file_name, std::ios::out);
			if (!out) {
				std::cout << "File not created!"; throw 0;
			}

			for (auto& row : data) {
				for (auto col : row)
					out << col << ',';
				out << '\n';
			}
			out.close();
		}

		//default dtor as no dynamic memory is used
		~matrix() {};
	};

	//------------------------------------------------
	//
	// non-member functions
	//
	//------------------------------------------------

	//measure diference between two matrices
	template<class T>
	double d1_distance(const matrix<T>& mat1, const matrix<T>& mat2) {
		assert(mat1.get_m() == mat2.get_m() && mat1.get_n() == mat2.get_n());

		double _distance = 0;
		for (int j = 0; j < mat1.get_m(); ++j) {
			_distance += abs(mat1[j] - mat2[j]).sum();
		}
		return _distance / (double(mat1.get_n()) * double(mat1.get_m()));
	}

	template<class T>
	matrix<T> eye(const uint_fast8_t m) {
		matrix mat(m, m, 0);
		for (int i = 0; i < m; ++i) {
			mat(i, i) = 1;
		}
		return mat;
	}

	/// <summary>
	/// Creates an identity matrix with swaped rows and columns
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="m"></param>
	/// <param name="v"></param>
	/// <returns></returns>
	template<class T>
	matrix<T> custom_eye(const uint_fast8_t m, const std::vector<uint_fast8_t>& v) {
		assert(m == v.size());
		matrix mat(m, m, 0);
		for (int i = 0; i < m; ++i) {
			mat(i, v[i]) = 1;
		}
		return mat;
	}

	template<class T>
	matrix<T> average(const matrix<T>& mat1, const matrix<T>& mat2) {
		assert(mat1.get_m() == mat2.get_m() && mat1.get_n() == mat2.get_n());

		const uint_fast8_t m = mat1.get_m();
		std::vector<std::valarray<T>> ret_data(m, std::valarray<T>(mat1.get_n()));

		for (int j = 0; j < m; ++j) {
			ret_data[j] = (mat1[j] + mat2[j]) / 2;
		}

		return ret_data;
	}

	//crossover two matrices swapping rows
	template<class T>
	std::pair<matrix<T>, matrix<T>> row_crossover(const matrix<T>& mat1, const matrix<T>& mat2) {
		assert(mat1.get_m() == mat2.get_m() && mat1.get_n() == mat2.get_n());

		const uint_fast8_t m = mat1.get_m();
		const uint_fast8_t a = random::randint(1, m - 1);

		const matrix<T> ret1(mat1[{0, a}], mat2[{a, m}]);
		const matrix<T> ret2(mat2[{0, a}], mat1[{a, m}]);

		return std::make_pair(ret1, ret2);
	}

	template<class T>
	std::pair<matrix<T>, matrix<T>> x_crossover(const matrix<T>& mat1, const matrix<T>& mat2) {
		assert(mat1.get_m() == mat2.get_m() && mat1.get_n() == mat2.get_n());

		const uint_fast8_t m = mat1.get_m();
		const uint_fast8_t n = mat1.get_n();

		// deal with special cases m == 1 || n == 1
		const uint_fast8_t a = m == 1 ? 0 : random::randint(1, m - 1);
		const uint_fast8_t b = n == 1 ? 0 : random::randint(1, n - 1);

		int i, j;

		std::vector<std::valarray<T>> data1(m, std::valarray<T>(n));

		std::vector<std::valarray<T>> data2(m, std::valarray<T>(n));

		const matrix<T>* p_mat1;//non constant pointer to constant matrix
		const matrix<T>* p_mat2;

		for (j = 0; j < m; ++j) {
			for (i = 0; i < n; ++i) {
				if (j < a == i < b) { //xor j<a and i<b, equivalent to (j<a && i<b) || (j>a && i>b)
					p_mat1 = &mat1;
					p_mat2 = &mat2;
				}
				else {
					p_mat1 = &mat2;
					p_mat2 = &mat1;
				}
				data1[j][i] = (*p_mat1)(j, i);
				data2[j][i] = (*p_mat2)(j, i);
			}
		}

		return std::make_pair(data1, data2);
	}

	template<class T>
	matrix<T> transpose(const matrix<T>& mat) {
		const uint_fast8_t _m = mat.get_n(), _n = mat.get_m();
		std::vector<std::valarray<T>> _data(_m, std::valarray<T>(_n));

		for (int j = 0; j < _m; ++j) {
			for (int i = 0; i < _n; ++i) {
				_data[j][i] = mat(i, j);
			}
		}
		return _data;
	}

	//returns adjunct matrix of mat referred to element ( y , x ) 
	template<class T>
	matrix<T> adjunct(const matrix<T>& mat, const uint_fast8_t n, const uint_fast8_t y, const uint_fast8_t x) {
		assert(x <= n && y <= n);
		std::vector<std::valarray<T>> ret_data(n - 1, std::valarray<T>(n - 1));
		uint_fast8_t _i = 0, _j = 0;

		for (int j = 0; j < n; ++j) {
			if (j == y) { continue; }
			_i = 0;
			for (int i = 0; i < n; ++i) {
				if (i == x) { continue; }
				ret_data[_j][_i++] = mat(j, i);
			}
			++_j;
		}
		return ret_data;
	}

	//matrix-matrix dot product implementation
	template<class T>
	matrix<T> big_mat_dot(const matrix<T>& mat1, const matrix<T>& mat2) { //dot product  with transpose for better data locality
		assert(mat1.get_n() == mat2.get_m());

		const auto _m = mat1.get_m(); //return dim m
		const auto _n = mat2.get_n(); //return dim n
		std::vector<std::valarray<T>> ret_data(_m, std::valarray<T>(_n));

		const auto mat2_t = transpose(mat2); // transpose to ease and speed up multiplication by reducing cache misses (hopefully)
		for (int j = 0; j < _m; ++j) {
			for (int i = 0; i < _n; ++i) {
				ret_data[j][i] = std::inner_product(std::begin(mat1[j]), std::end(mat1[j]), std::begin(mat2_t[i]), T(0));
			}
		}
		return ret_data;
	}

	//matrix-matrix dot product implementation
	template<class T>
	matrix<T> dot(const matrix<T>& mat1, const matrix<T>& mat2) { //smaller matrix dot product implementation
		assert(mat1.get_n() == mat2.get_m());

		const auto _m = mat1.get_m(); //return dim m
		const auto _n = mat2.get_n(); //return dim n
		const auto _k = mat1.get_n(); //return inner dim k
		std::vector<std::valarray<T>> ret_data(_m, std::valarray<T>(_n));
		for (int i = 0; i < _m; ++i) {
			for (int k = 0; k < _k; ++k) {
				for (int j = 0; j < _n; ++j) {
					ret_data[i][j] += mat1(i, k) * mat2(k, j);
				}
			}
		}
		return ret_data;
	}

	//to be implemented
	//template<class T>
	//matrix<T> fast_dot(const matrix<T>& mat1, const matrix<T>& mat2) {
	//	assert(mat1.get_n() == mat2.get_m());

	//	const auto _m = mat1.get_m(); //return dim m
	//	const auto _n = mat2.get_n(); //return dim n
	//	std::vector<std::valarray<T>> ret_data(_m, std::valarray<T>(_n));
	//	return matrix<T>(ret_data);
	//}

	//vector-matrix dot product implementation
	template<class T>
	std::valarray<T> big_mat_dot(const std::valarray<T>& v, const matrix<T>& mat) {//vector-matrix dot product implementation, returns ROW vector as a valarray
		assert(v.size() == mat.get_m());

		const auto _n = mat.get_n(); //return dim n
		std::valarray<T> ret_data(_n);

		const auto mat_t = transpose(mat); // transpose to ease and speed up multiplication by reducing cache misses (hopefully)
		for (int i = 0; i < _n; ++i) {
			ret_data[i] = std::inner_product(std::begin(v), std::end(v), std::begin(mat_t[i]), T(0));
		}
		return ret_data;
	}

	//vector-matrix dot product implementation
	template<class T>
	std::valarray<T> dot(const std::valarray<T>& v, const matrix<T>& mat) {//vector-matrix dot product implementation, returns ROW vector as a valarray
		assert(v.size() == mat.get_m());

		const auto _n = mat.get_n(); //return dim n
		std::valarray<T> ret_data(_n);

		T _val(0), j(0);

		for (int i = 0; i < v.size(); ++i) {
			for (int j = 0; j < _n ; ++j) {
				ret_data[j] += v[i] * mat(i, j);
			}
		}
		return ret_data;
	}

	//matrix-vector dot product implementation
	template<class T>
	std::valarray<T> big_mat_dot(const matrix<T>& mat, const std::valarray<T>& v) { 	//matrix-vector dot product implementation, returns COLUMN vector as a valarray

		assert(v.size() == mat.get_n());

		const auto _m = mat.get_m(); //return dim m
		std::valarray<T> ret_data(_m);

		const auto mat_t = transpose(mat); // transpose to ease and speed up multiplication by reducing cache misses (hopefully)
		for (int j = 0; j < _m; ++j) {
			ret_data[j] = std::inner_product(std::begin(mat[j]), std::end(mat[j]), std::begin(v), T(0));
		}
		return ret_data;
	}

	template<class T>
	std::valarray<T> dot(const matrix<T>& mat, const std::valarray<T>& v) { 	//matrix-vector dot product implementation, returns COLUMN vector as a valarray

		assert(v.size() == mat.get_n());

		const auto _m = mat.get_m(); //return dim m
		std::valarray<T> ret_data(_m);

		for (int j = 0; j < _m; ++j) {
			ret_data[j] = std::inner_product(std::begin(mat[j]), std::end(mat[j]), std::begin(v), T(0));
		}
		return ret_data;
	}

	template<class T>
	T determinant(const matrix<T>& mat) {
		return std::get<1>(PII_LUDecomposition(mat));
	}

	template<class U = double, class T>
	U slow_determinant(const matrix<T>& mat) {
		assert(mat.get_m() == mat.get_n());

		const uint_fast8_t n = mat.get_m();
		U det = 0;

		if (n == 1) { return mat(0, 0); }

		if (n == 2) {
			return (U)mat(0, 0) * (U)mat(1, 1) - (U)mat(1, 0) * (U)mat(0, 1);
		}

		for (int x = 0; x < n; ++x) {
			det += ((x % 2 == 0 ? 1 : -1) * (U)mat(0, x) * slow_determinant<U>(adjunct(mat, n, 0, x)));
		}
		return det;
	}

	template<class T>
	bool nearly_equals(const matrix<T>& mat1, const matrix<T>& mat2, const float epsilon)
	{
		assert(mat1.get_m() == mat2.get_m() && mat1.get_n() == mat2.get_n());

		for (unsigned short int i = 0; i < mat1.get().size(); ++i) {
			for (bool b : (std::abs(mat1.get()[i] - mat2.get()[i]) < epsilon))
				if (!b) { return false; }
		}

		return true;
	}

	template<class T>
	std::ostream& operator<<(std::ostream& os, const matrix<T>& mat) {
		os << std::fixed;
		os << std::setprecision(4);
		for (int i = 0; i < mat.get_m(); ++i) {
			for (int j = 0; j < mat.get_n(); ++j) {
				os << mat(i, j) << ", ";
			}
			os << "\n";
		}
		os << std::defaultfloat;
		return os;
	}

	template<class T>
	const bool operator==(const matrix<T>& mat1, const matrix<T>& mat2)
	{
		assert(mat1.get_m() == mat2.get_m() && mat1.get_n() == mat2.get_n());

		for (unsigned short int i = 0; i < mat1.get().size(); ++i) {
			for (bool b : mat1.get()[i] == mat2.get()[i])
				if (!b) { return false; }
		}

		return true;
	}

	template<class T>
	const bool operator!=(const matrix<T>& mat1, const matrix<T>& mat2)
	{
		return !(mat1 == mat2);
	}

	//source
	//http://web.archive.org/web/20150701223512/http://download.intel.com/design/PentiumIII/sml/24504601.pdf
	//efficient LU decompoition
	template<class T>
	std::tuple<bool, double, matrix<T>, std::vector<uint_fast8_t>> PII_LUDecomposition(const matrix<T>& source)
	{
		// Factors "m" matrix into A=LU where L is lower triangular and U is upper
		// triangular. The matrix is overwritten by LU with the diagonal elements
		// of L (which are unity) not stored. This must be a square n x n matrix.

		const uint_fast8_t n = source.get_n();
		assert(n == source.get_m());

		matrix<T> out(source);

		double det = 1.0;

		// Initialize the pointer vector.
		std::vector<uint_fast8_t> ri(n);
		for (int i = 0; i < n; ++i)
			ri[i] = i;

		// LU factorization.
		for (int p = 0; p < n - 1; ++p) {
			//Find pivot element.
			for (int j = p + 1; j < n; ++j) {
				if (abs(out[ri[j]][p]) > abs(out[ri[p]][p])) {
					// Switch the index for the p pivot row if necessary.
					std::swap(ri[j], ri[p]);
					det = -det;
					//ri[p] now has the index of the row to consider the pth
				}
			}
			if (out[ri[p]][p] == 0) {
				// The matrix is singular. //or not inversible by this methode untill fixed (no permutations)
				return std::tuple<bool, double, matrix<T>, std::vector<uint_fast8_t>>(false, NAN, out, ri);
			}
			// Multiply the diagonal elements.
			det *= out[ri[p]][p];

			// Form multiplier.
			for (int j = p + 1; j < n; ++j) {
				out[ri[j]][p] /= out[ri[p]][p];
				// Eliminate [p].
				for (int i = p + 1; i < n; ++i)
					out[ri[j]][i] -= out[ri[j]][p] * out[ri[p]][i];
			}
		}
		det *= out[ri[n - 1]][n - 1]; //multiply last diagonal element

		const auto _ri(ri);

		for (int i = 0; i < n; ++i) {
			if (i != ri[i]) {
				std::swap(out[i], out[ri[i]]);
				std::swap(ri[i], ri[std::find(ri.begin(), ri.end(), i) - ri.begin()]);
			}
		}

		return std::tuple<bool, double, matrix<T>, std::vector<uint_fast8_t>>
			(det != 0.0, det != 0.0 ? det : NAN, out, _ri);
	}

	template<class T>
	matrix<T> inverse(const matrix<T>& mat) {

		const uint_fast8_t m = mat.get_m(), n = mat.get_n();
		assert(m == n);
		std::tuple<const bool, const double, matrix<T>, std::vector<uint_fast8_t>> LU_decomp = PII_LUDecomposition(mat);

		const auto inv = std::get<0>(LU_decomp);
		const auto det = std::get<1>(LU_decomp);
		const auto LU = std::get<2>(LU_decomp);
		const auto ri = std::get<3>(LU_decomp);

		if (!inv) {
			std::cout << "Singular matrix\n";
			return matrix<T>(m, n, T(NAN)); //this would not work for custom T's
		}

		int i, j;

		//reconstruct L_data
		std::vector<std::valarray<T>> L_data(m, std::valarray<T>(n));
		for (j = 0; j < m; ++j) {
			for (i = 0; i <= j; ++i) {
				if (i != j) {
					L_data[j][i] = LU[j][i];
				}
				else {
					L_data[j][i] = 1;
				}
			}
		}

		//reconstruct U data
		std::vector<std::valarray<T>> U_data(m, std::valarray<T>(n));
		for (j = 0; j < m; ++j) {
			for (i = j; i < n; ++i) {
				U_data[j][i] = LU[j][i];
			}
		}

		const matrix<T> L(L_data), U(U_data);

		// L * U = A
		// L * U * P^-1 * A^-1 = I
		// P = piboted(I)

		std::vector<std::valarray<T>> inv_T_data(m, std::valarray<T>(n));
		for (i = 0; i < m; ++i) {
			std::valarray<T> Pi(n); //initilized by default to all zeros
			Pi[int(std::find(ri.begin(), ri.end(), i) - ri.begin())] = 1; // column (or row) identity vectors
			inv_T_data[i] = LU_matrix_vector_solve(L, U, Pi);
		}

		return transpose<T>(inv_T_data);
	}

	template<class T>
	std::valarray<T> LU_matrix_vector_solve(const matrix<T>& L, const matrix<T>& U, const std::valarray<T>& d) { // Solves A * y = d   with A = L * U
		const uint_fast8_t m = L.get_m(), n = L.get_n();
		assert(m == n);
		assert(U.get_m() == m && U.get_n() == n);
		assert(d.size() == m);

		//solve L*x = d
		std::valarray<T> x(d);
		for (int j = 1; j < m; ++j) {
			for (int i = 0; i < j; ++i) {
				x[j] -= L(j, i) * x[i];
			}
		}

		//solve U*y = x
		std::valarray<T> y(x);
		for (int j = m - 1; j >= 0; --j) {
			for (int i = m - 1; i > j; --i) {
				y[j] -= U(j, i) * y[i];
			}
			y[j] /= U(j, j);
		}

		return y;
	}

}
