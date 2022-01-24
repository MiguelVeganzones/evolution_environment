#pragma once
#include <iostream>
/*
based on:
	msvc  std::array
	https://github.com/douglasrizzo/matrix/blob/master/include/nr3/nr3.h
*/

#define _CHECKBOUNDS_ 1
//#ifdef _CHECKBOUNDS_
//if (i < 0 || i >= nn) {
//	throw("NRmatrix subscript out of bounds");
//}
//#endif

template <class _Ty, size_t _Size>
class matrix_const_iterator {
public:
	using value_type		= _Ty;
	using ptrdiff_t			= long long;
	using pointer			= const _Ty*;
	using reference			= const _Ty&;

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

	[[nodiscard]] constexpr matrix_const_iterator operator-(const ptrdiff_t _Off) const noexcept{
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

	using value_type		= _Ty;
	using ptrdiff_t	= long long;
	using pointer			= _Ty*;
	using reference			= _Ty&;

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
	using value_type		= _Ty;
	using size_type			= size_t;
	using ptrdiff_t			= long long;
	using pointer			= _Ty*;
	using const_pointer		= const _Ty*;
	using reference			= _Ty&;
	using const_reference	= const _Ty&;
	using iterator			= matrix_iterator<_Ty, _M * _N>;
	using const_iterator	= matrix_const_iterator<_Ty, _M * _N>;

	const size_t _Size = _M * _N;
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
			std::cout << "a: " << a<<
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
};

template<class _Ty, size_t _M, size_t _N>
constexpr std::ostream& operator<<(std::ostream& os, const stack_matrix<_Ty, _M, _N>& _Mat) {
	if (!std::is_integral<_Ty>::value) {
		os << std::fixed;
		os << std::setprecision(4);
	}
	std::cout << "[";
	for (size_t i = 0; i < _M; ++i) {
		std::cout << "\n [ ";
		for (size_t j = 0; j < _N; ++j) {
			os << _Mat._Elems[i * _M + j] << " ";
		}
		os << "]";
	}
	std::cout << "\n]\n";
	os << std::defaultfloat;
	return os;
}




