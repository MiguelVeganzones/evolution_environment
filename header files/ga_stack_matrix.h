#pragma once
#include <iostream>
#include <array>
#include "Random.h"
/*
based on:
	msvc  std::array
	https://github.com/douglasrizzo/matrix/blob/master/include/nr3/nr3.h
*/

#ifdef NDEBUG
	#define _CHECKBOUNDS_ 1
#endif // !NDEBUG

//#ifdef _CHECKBOUNDS_
//if (i < 0 || i >= nn) {
//	throw("NRmatrix subscript out of bounds");
//}
//#endif

namespace ga_sm {

	using namespace ga_sm;

	template <class _Ty, size_t _Size>
	class matrix_const_iterator {
	public:
		using value_type = _Ty;
		using ptrdiff_t = long long;
		using pointer = const _Ty*;
		using reference = const _Ty&;

		constexpr matrix_const_iterator() noexcept : _Ptr(nullptr) {}

		constexpr explicit matrix_const_iterator(pointer _Parg, size_t _Off = 0) noexcept : _Ptr(_Parg + _Off) {}

		[[nodiscard]] constexpr reference operator*() const noexcept {
			return *_Ptr;
		}

		[[nodiscard]] constexpr pointer operator->() const noexcept {
			return _Ptr;
		}

		//++iter; retuns by reference mutated original object. Idem for the rest
		constexpr matrix_const_iterator& operator++() noexcept {
			++_Ptr;
			return *this;
		}

		//iter++; returns by value copy of the original iterator. Idem for the rest
		constexpr matrix_const_iterator operator++(int) noexcept {
			matrix_const_iterator _Tmp = *this;
			++_Ptr;
			return _Tmp;
		}

		//--iter;
		constexpr matrix_const_iterator& operator--() noexcept {
			--_Ptr;
			return *this;
		}

		//iter--;
		constexpr matrix_const_iterator operator--(int) noexcept {
			matrix_const_iterator _Tmp = *this;
			--_Ptr;
			return _Tmp;
		}

		constexpr matrix_const_iterator& operator+=(const ptrdiff_t _Off) noexcept {
			_Ptr += _Off;
			return *this;
		}

		[[nodiscard]] constexpr matrix_const_iterator operator+(const ptrdiff_t _Off) const noexcept {
			matrix_const_iterator _Tmp = *this;
			_Tmp += _Off;
			return _Tmp;
		}

		constexpr matrix_const_iterator& operator-=(const ptrdiff_t _Off) noexcept {
			_Ptr -= _Off;
			return *this;
		}

		[[nodiscard]] constexpr matrix_const_iterator operator-(const ptrdiff_t _Off) const noexcept {
			matrix_const_iterator _Tmp = *this;
			_Tmp -= _Off;
			return _Tmp;
		}

		[[nodiscard]] constexpr ptrdiff_t operator-(const matrix_const_iterator& _Rhs) const noexcept {
			return _Ptr - _Rhs._Ptr;
		}

		[[nodiscard]] constexpr reference operator[](const ptrdiff_t _Off) const noexcept {
			return _Ptr[_Off];
		}

		[[nodiscard]] constexpr bool operator==(const matrix_const_iterator& _Rhs) const noexcept {
			return _Ptr == _Rhs._Ptr;
		}

		[[nodiscard]] constexpr bool operator!=(const matrix_const_iterator& _Rhs) const noexcept {
			return !(*this == _Rhs);
		}

		[[nodiscard]] constexpr bool operator>(const matrix_const_iterator& _Rhs) const noexcept {
			return _Ptr > _Rhs._Ptr;
		}

		[[nodiscard]] constexpr bool operator<(const matrix_const_iterator& _Rhs) const noexcept {
			return _Ptr < _Rhs._Ptr;
		}

		constexpr void _Seek_to(pointer _It) noexcept {
			_Ptr = _It;
		}

		constexpr void _Unwrapped(pointer _It) const noexcept {
			return _Ptr;
		}

	private:
		pointer _Ptr;
	};

	//-------------------------------------------------------------------//

	template <class _Ty, size_t _Size>
	class matrix_iterator : public matrix_const_iterator<_Ty, _Size> {
	public:
		using _Mybase = matrix_const_iterator<_Ty, _Size>;

		using value_type = _Ty;
		using ptrdiff_t = long long;
		using pointer = _Ty*;
		using reference = _Ty&;

		constexpr matrix_iterator() noexcept {}

		constexpr explicit matrix_iterator(pointer _Parg, size_t _Off = 0) noexcept : _Mybase(_Parg, _Off) {}

		[[nodiscard]] constexpr reference operator*() const noexcept {
			return const_cast<reference>(_Mybase::operator*());
		}

		[[nodiscard]] constexpr pointer operator->() const noexcept {
			return const_cast<pointer>(_Mybase::operator->());
		}

		constexpr matrix_iterator& operator++() noexcept {
			_Mybase::operator++();
			return *this;
		}

		constexpr matrix_iterator operator++(int) noexcept {
			matrix_iterator _Tmp = *this;
			_Mybase::operator++();
			return _Tmp;
		}

		constexpr matrix_iterator& operator--() noexcept {
			_Mybase::operator--();
			return *this;
		}

		constexpr matrix_iterator operator--(int) noexcept {
			matrix_iterator _Tmp = *this;
			_Mybase::operator--();
			return _Tmp;
		}

		constexpr matrix_iterator& operator+=(const ptrdiff_t _Off) noexcept {
			_Mybase::operator+=(_Off);
			return *this;
		}

		[[nodiscard]] constexpr matrix_iterator operator+(const ptrdiff_t _Off) const noexcept {
			matrix_iterator _Tmp = *this;
			_Tmp += _Off;
			return _Tmp;
		}

		constexpr matrix_iterator& operator-=(const ptrdiff_t _Off) noexcept {
			_Mybase::operator-=(_Off);
			return *this;
		}

		[[nodiscard]] constexpr matrix_iterator operator-(const ptrdiff_t _Off) const noexcept {
			matrix_iterator _Tmp = *this;
			_Tmp -= _Off;
			return _Tmp;
		}

		[[nodiscard]] constexpr reference operator[](const ptrdiff_t _Off) const noexcept {
			return const_cast<reference>(_Mybase::operator[](_Off));
		}

		[[nodiscard]] constexpr pointer _Unwrapped() const noexcept {
			return const_cast<pointer>(_Mybase::_Unwrapped());
		}
	};

	//-------------------------------------------------------------------//

	template <class _Ty, size_t _M, size_t _N>
	class stack_matrix {
	public:
		using value_type = _Ty;
		using size_type = size_t;
		using ptrdiff_t = long long;
		using pointer = _Ty*;
		using const_pointer = const _Ty*;
		using reference = _Ty&;
		using const_reference = const _Ty&;
		using iterator = matrix_iterator<_Ty, _M* _N>;
		using const_iterator = matrix_const_iterator<_Ty, _M* _N>;
		using congruent_matrix = stack_matrix<_Ty, _M, _N>;

		size_t _Size = _M * _N;
		_Ty _Elems[_M * _N];

		/*--------------------------------------------*/

		constexpr void fill(const _Ty& _Val) {
			fill_n(iterator(_Elems), _Size, _Val);
		}

		template<auto fn, typename... Args>
		constexpr void fill(Args&&... args) {
			iterator _Curr = iterator(_Elems);
			for (ptrdiff_t i = 0; i < _Size; ++i, ++_Curr) {
				*_Curr = fn(std::forward<Args>(args)...);
			}
		}

		constexpr void fill_n(const iterator& _Dest, const ptrdiff_t _Count, const _Ty _Val) {
			#ifdef _CHECKBOUNDS_
			bool a = *_Dest < *_Elems; //Pointer before array
			bool b = *_Dest + _Count > *_Elems + _Size; //write past array limits
			if (a || b) {
				std::cout << "a: " << a <<
					"b: " << b << std::endl;
				throw("fill_n subscript out of bounds");
			}
			#endif
			iterator _Curr = _Dest;
			for (ptrdiff_t i = 0; i < _Count; ++i, ++_Curr) {
				*_Curr = _Val;
			}
		}

		[[nodiscard]] constexpr iterator begin() noexcept {
			return iterator(_Elems);
		}

		[[nodiscard]] constexpr const_iterator begin() const noexcept {
			return const_iterator(_Elems);
		}

		[[nodiscard]] constexpr iterator end() noexcept {
			return iterator(_Elems, _Size);
		}

		[[nodiscard]] constexpr const_iterator end() const noexcept {
			return const_iterator(_Elems, _Size);
		}

		[[nodiscard]] constexpr reference operator()(const size_t j, const size_t i) noexcept {
			assert(j < _M&& i < _N);
			return _Elems[j * _N + i];
		}

		[[nodiscard]] constexpr const_reference operator()(const size_t j, const size_t i) const noexcept {
			assert(j < _M&& i < _N);
			return _Elems[j * _N + i];
		}

		[[nodiscard]] constexpr stack_matrix<_Ty, 1, _N> operator[](size_t j) const {
			assert(j < _M);
			stack_matrix<_Ty, 1, _N> _Ret{0};
			const_iterator _It(_Elems, j * _N);
			for (size_t i = 0; i < _N; ++i, ++_It) {
				_Ret(0, i) = *_It;
			}
			return _Ret;
		}

		[[nodiscard]] constexpr congruent_matrix operator+(const congruent_matrix& _Other) const {
			congruent_matrix _Ret(*this);
			for (size_t i = 0; i < _Size; ++i) {
				_Ret._Elems[i] += _Other._Elems[i];
			}
			return _Ret;
		}

		[[nodiscard]] constexpr congruent_matrix operator-(const congruent_matrix& _Other) const {
			congruent_matrix _Ret(*this);
			for (size_t i = 0; i < _Size; ++i) {
				_Ret._Elems[i] -= _Other._Elems[i];
			}
			return _Ret;
		}

		[[nodiscard]] constexpr congruent_matrix operator*(const _Ty p) const {
			congruent_matrix _Ret(*this);
			for (auto& e : _Ret) e *= p;
			return _Ret;
		}

		[[nodiscard]] constexpr congruent_matrix operator*(const congruent_matrix& _Other) const {
			congruent_matrix _Ret(*this);
			for (size_t i = 0; i < _Size; ++i) {
				_Ret._Elems[i] *= _Other._Elems[i];
			}
			return _Ret;
		}

		[[nodiscard]] constexpr congruent_matrix operator/(const _Ty p) const {
			congruent_matrix _Ret(*this);
			for (auto& e : _Ret) e /= p;
			return _Ret;
		}

		/*-----------------------------------------------------------------
							###		Utility		###
		------------------------------------------------------------------*/

		[[nodiscard]] constexpr _Ty sum() const {
			_Ty sum{0};
			for (auto& e : *this) sum += e;
			return sum;
		}

		//only works for float-like types
		constexpr void rescale_L_1_1_norm(const float _Norm = 1.f) {
			assert(!std::is_integral<_Ty>::value);
			assert(_Norm != 0.f);
			const _Ty _Sum = sum();
			for (auto& e : *this) e /= _Sum;
		}

	};

	template<class _Ty, size_t _M, size_t _N>
	constexpr std::ostream& operator<<(std::ostream& os, const stack_matrix<_Ty, _M, _N>& _Mat) {
		if (!std::is_integral<_Ty>::value) {
			os << std::fixed;
			os << std::setprecision(4);
		}
		std::cout << "[";
		for (size_t j = 0; j < _M; ++j) {
			std::cout << "\n [ ";
			for (size_t i = 0; i < _N; ++i) {
				os << _Mat._Elems[j * _N + i] << " ";
			}
			os << "]";
		}
		std::cout << "\n]\n";
		os << std::defaultfloat;
		return os;
	}

	template<class _Ty, size_t _M, size_t _N>
	[[nodiscard]] constexpr std::pair<stack_matrix<_Ty, _M, _N>, stack_matrix<_Ty, _M, _N>>
		x_crossover(const stack_matrix<_Ty, _M, _N>& _Mat1, const stack_matrix<_Ty, _M, _N> _Mat2) {

		//indices to slice. a: horizontal, b: vertical
		size_t a = random::randint(0, _M);
		size_t b = random::randint(0, _N);

		//return swapped original matrices to avoid irrelevant operations
		if ((a == 0 or a == _M) and (b == 0 or b == _N)) return std::pair(_Mat2, _Mat1);

		//minimize swaps -> less area
		size_t _A1 = a * b + (_M - a) * (_N - b);
		size_t _A2 = _M * _N - _A1;

		bool _D = _A1 < _A2; //block diagonal to swap. True for main diagonal, 0 for anti diagonal 
		int count = 0;
		std::cout << "\n_D: " << _D << std::endl;

		//setup return matrices.
		stack_matrix<_Ty, _M, _N> _Ret1(_Mat1);
		stack_matrix<_Ty, _M, _N> _Ret2(_Mat2);

		//swap first block
		for (size_t j = 0; j < a; ++j) {
			for (size_t i = (_D ? 0 : b); i < (_D ? b : _N); ++i) {
				std::swap(_Ret1(j, i), _Ret2(j, i));
			}
		}

		//swap second block
		for (size_t j = a; j < _M; ++j) {
			for (size_t i = (_D ? b : 0); i < (_D ? _N : b); ++i) {
				std::swap(_Ret1(j, i), _Ret2(j, i));
			}

		}

		return std::make_pair(_Ret1, _Ret2);
	}

	template<class _Ty, size_t _M, size_t _K, size_t _N>
	[[nodiscard]] stack_matrix<_Ty, _M, _N> matrix_mul(
		const stack_matrix<_Ty, _M, _K>& _Mat1,
		const stack_matrix<_Ty, _K, _N>& _Mat2) {

		stack_matrix<_Ty, _M, _N> _Ret{0};

		for (size_t j = 0; j < _M; ++j) {
			for (size_t k = 0; k < _K; ++k) {
				for (size_t i = 0; i < _N; ++i) {
					_Ret(j, i) += _Mat1(j, k) * _Mat2(k, i);
				}
			}
		}

		return _Ret;
	}
}



