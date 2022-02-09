#pragma once
#include <iostream>
#include "Random.h"
#include "nn_helper_functions.h"
#include <concepts>
/*
based on:
  msvc implementation of std::array
  and: https://github.com/douglasrizzo/matrix/blob/master/include/nr3/nr3.h
*/

#ifdef NDEBUG
#define _CHECKBOUNDS_ 1
#endif // !NDEBUG

//#ifdef _CHECKBOUNDS_
//if (i < 0 || i >= nn) {
//	throw("NRmatrix subscript out of bounds");
//}
//#endif

/*
Basic dense matrix library

Row major, dense, and efficient static matrix implementation.

It is highly based in std::array, but with two dimensional indexing.
It is a trivial type, and has standar layout (called POD type (Pre c++20)).
That means it is trivially copiable, trivially copy constructible, and others.

Most functionality is constexpr and can be exceuted in compile time. This is used
for its unit testing and can be a useful feature in some situations.

Added functionality is mainly math operations, to make this a matrix and not just a 2D array,
convenience functions, and basic functionality that can be useful in GA algorithms.

This is not a matrix intended to do math, but does have math functionality. Which means that 
algorithms like inverse or RREF might yield wrong results when matrices are not invertible or
is their determinant is too small (Note: having a small determinant is not a condition for closeness
to singularity, but can be a good heuristic). This is specially the case when using single precision floating 
point numbers, and thus, double recision should be used for math applications.
Results of this operations should be double checked if singular matrices might appear as the algorithms 
do not cover corner cases.

stack_matrix is not a particularly good name as it can also be allocated on the heap. 
It is more os a stack-enabled container.  

*/

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
    using value_type            = _Ty;
    using size_type             = size_t;
    using ptrdiff_t             = long long;
    using pointer               = _Ty*;
    using const_pointer         = const _Ty*;
    using reference             = _Ty&;
    using const_reference       = const _Ty&;
    using iterator              = matrix_iterator<_Ty, _M* _N>;
    using const_iterator        = matrix_const_iterator<_Ty, _M* _N>;
    using congruent_matrix      = stack_matrix<_Ty, _M, _N>;
    using row_matrix            = stack_matrix<_Ty, 1, _N>;
    using column_matrix         = stack_matrix<_Ty, _N, 1>;

    _Ty _Elems[_M * _N];

    /*--------------------------------------------*/

    constexpr void fill(const _Ty& _Val) {
      fill_n(iterator(_Elems), _M * _N, _Val);
    }

    template<auto fn, typename... Args>
    constexpr void fill(Args&&... _Args) {
      for (iterator _Curr = begin(); _Curr != end(); ++_Curr) {
        *_Curr = static_cast<_Ty>(fn(std::forward<Args>(_Args)...));
      }
    }

    /*
    Each element of the matrix might be:
      Modifies by a normal distribution given by [_Avg, _Stddev] with a probability of p1: // e += randnoral(avg, stddev)
      Replaced by a value given by fn scaled by _Ampl2 with a probability of p2-p1: // e = (fn(args...) + offset) * ampl
    */
    template<auto fn, auto... _Args>
      requires std::is_floating_point<_Ty>::value
    void mutate(float _Avg, float _Stddev, float p1, float p2, float _Ampl = 1, float _Offset = -0.5) {
      assert(p2 >= p1);

      float _Rand;
      for (iterator _Curr = begin(); _Curr != end(); ++_Curr) {
        _Rand = random::randfloat();
        if (_Rand < p1) {
          *_Curr += static_cast<_Ty>(random::randnormal(_Avg, _Stddev));
        }
        else if (_Rand < p2) {
          *_Curr = static_cast<_Ty>((static_cast<float>(fn(_Args...)) + _Offset) * _Ampl);
        }
      }
    }

    constexpr void fill_n(const iterator& _Dest, const ptrdiff_t _Count, const _Ty _Val) {
#ifdef _CHECKBOUNDS_
      bool a = *_Dest < *_Elems; //Pointer before array
      bool b = *_Dest + _Count > *_Elems + _M * _N; //write past array limits
      if (a || b) {
        std::cout << "a: " << a <<
          "b: " << b << std::endl;
        throw("fill_n subscript out of bounds\n");
      }
#endif

      iterator _Curr(_Dest);
      for (ptrdiff_t i = 0; i < _Count; ++i, ++_Curr) {
        *_Curr = _Val;
      }
    }

    //in place transpose operator
    constexpr void T() noexcept {
      for (size_t j = 0; j < _M - 1; ++j) {
        for (size_t i = j + 1; i < _N; ++i) {
          std::swap(this->operator()(j, i), this->operator()(i, j));
        }
      }
    }

    [[nodiscard]] constexpr iterator begin() noexcept {
      return iterator(_Elems);
    }

    [[nodiscard]] constexpr const_iterator begin() const noexcept {
      return const_iterator(_Elems);
    }

    [[nodiscard]] constexpr iterator end() noexcept {
      return iterator(_Elems, _M * _N);
    }

    [[nodiscard]] constexpr const_iterator end() const noexcept {
      return const_iterator(_Elems, _M * _N);
    }

    [[nodiscard]] inline constexpr 
    reference operator()(const size_t j, const size_t i) noexcept {
      assert(j < _M and i < _N);
      return _Elems[j * _N + i];
    }

    [[nodiscard]] inline constexpr 
    const_reference operator()(const size_t j, const size_t i) const noexcept {
      assert(j < _M and i < _N);
      return _Elems[j * _N + i];
    }

    [[nodiscard]] constexpr row_matrix operator[](size_t j) const {
      assert(j < _M);
      stack_matrix<_Ty, 1, _N> _Ret{};
      const_iterator _It(_Elems, j * _N);
      for (size_t i = 0; i < _N; ++i, ++_It) {
        _Ret(0, i) = *_It;
      }
      return _Ret;
    }

    [[nodiscard]] constexpr 
    congruent_matrix operator+(const congruent_matrix& _Other) const {
      congruent_matrix _Ret(*this);
      for (size_t i = 0; i < _M * _N; ++i) {
        _Ret._Elems[i] += _Other._Elems[i];
      }
      return _Ret;
    }

    [[nodiscard]] constexpr 
    congruent_matrix operator-(const congruent_matrix& _Other) const {
      congruent_matrix _Ret(*this);
      for (size_t i = 0; i < _M * _N; ++i) {
        _Ret._Elems[i] -= _Other._Elems[i];
      }
      return _Ret;
    }

    [[nodiscard]] constexpr congruent_matrix operator*(const _Ty p) const {
      congruent_matrix _Ret(*this);
      for (auto& e : _Ret) e *= p;
      return _Ret;
    }

    constexpr congruent_matrix operator*=(const _Ty p) noexcept {
      for (auto& e : *this) e *= p;
      return *this;
    }

    [[nodiscard]] constexpr congruent_matrix operator/(const _Ty p) const {
      congruent_matrix _Ret(*this);
      for (auto& e : _Ret) e /= p;
      return _Ret;
    }

    constexpr congruent_matrix operator/=(const _Ty p) {
      assert(p != 0);
      for (auto& e : *this) e /= p;
      return *this;
    }


    /*-----------------------------------------------------------------
              ###		Utility		###
    ------------------------------------------------------------------*/

    [[nodiscard]] constexpr _Ty sum() const noexcept {
      _Ty sum{ 0 };
      for (auto& e : *this) sum += e;
      return sum;
    }

    //only works for float-like types
    constexpr void rescale_L_1_1_norm(const float _Norm = 1.f) {
      assert(std::is_floating_point<_Ty>::value);
      assert(_Norm != 0.f);
      const _Ty _Sum = sum();
      for (auto& e : *this) e /= _Sum;
    }

  };

  template<class _Ty, size_t _M, size_t _N>
  std::ostream& operator<<(
      std::ostream& os, 
      const stack_matrix<_Ty, _M, _N>& _Mat) {

    if (!std::is_integral<_Ty>::value) {
      os << std::fixed;
      os << std::setprecision(4);
    }
    std::cout << "[";
    for (size_t j = 0; j < _M; ++j) {
      std::cout << "\n [ ";
      for (size_t i = 0; i < _N; ++i) {
        os << _Mat(j, i) << " ";
      }
      os << "]";
    }
    std::cout << "\n]\n";
    os << std::defaultfloat;
    return os;
  }


  /*
  Returns exactly equals for integral typesand nearly equals if else
  Default tolerance used is 1e-4 for non integral types
  */
  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  bool operator==(
      const stack_matrix<_Ty, _M, _N>& _Mat1,
      const stack_matrix<_Ty, _M, _N>& _Mat2) {
    return std::is_integral<_Ty>::value ? exactly_equals(_Mat1, _Mat2) : nearly_equals(_Mat1, _Mat2);
  }

  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  bool operator!=(
      const stack_matrix<_Ty, _M, _N>& _Mat1,
      const stack_matrix<_Ty, _M, _N>& _Mat2) {
    return !operator==(_Mat1, _Mat2);
  }

  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] 
  std::pair<stack_matrix<_Ty, _M, _N>, stack_matrix<_Ty, _M, _N>> x_crossover(
      const stack_matrix<_Ty, _M, _N>& _Mat1,
      const stack_matrix<_Ty, _M, _N>& _Mat2) {

    //indices to slice. a: horizontal, b: vertical
    size_t a = random::randint(0, _M);
    size_t b = random::randint(0, _N);

    //return swapped original matrices to avoid irrelevant operations
    if ((a == 0 or a == _M) and (b == 0 or b == _N)) return std::pair(_Mat2, _Mat1);

    //minimize swaps -> less area
    size_t _A1 = a * b + (_M - a) * (_N - b);
    size_t _A2 = _M * _N - _A1;

    bool _D = _A1 < _A2; //block diagonal to swap. True for main diagonal, false for anti diagonal 
    int count = 0;

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
  [[nodiscard]] constexpr 
  stack_matrix<_Ty, _M, _N> matrix_mul(
      const stack_matrix<_Ty, _M, _K>& _Mat1,
      const stack_matrix<_Ty, _K, _N>& _Mat2) {

    stack_matrix<_Ty, _M, _N> _Ret{};

    for (size_t j = 0; j < _M; ++j) {
      for (size_t k = 0; k < _K; ++k) {
        for (size_t i = 0; i < _N; ++i) {
          _Ret(j, i) += _Mat1(j, k) * _Mat2(k, i);
        }
      }
    }
    return _Ret;
  }

  /*
  efficient LU decompoition of _N by _N matrix M
  returns:
    bool: det(M) != 0 //If matrix is inversible or not
    double: det(M)
    congruent_matrix LU: L is lower triangular with unit diagonal (implicit)
               U is upped diagonal, including diagonal elements

  */
  template<class _Ty, size_t _N>
  [[nodiscard]] constexpr 
  std::tuple<bool, double, stack_matrix<float, _N, _N>, std::array<size_t, _N>>
  PII_LUDecomposition(const stack_matrix<_Ty, _N, _N>& _Src)
  {
    /*
    source:
    http://web.archive.org/web/20150701223512/http://download.intel.com/design/PentiumIII/sml/24504601.pdf

    Factors "_Source" matrix into _Out=LU where L is lower triangular and U is upper
    triangular. The matrix is overwritten by LU with the diagonal elements
    of L (which are unity) not stored. This must be a square n x n matrix.
    */

    using namespace cx_helper_func;//constexpr abs
    using float_congruent_matrix = stack_matrix<float, _N, _N>;

    float_congruent_matrix _Out{};
    for (size_t j = 0; j < _N; ++j) {
      for (size_t i = 0; i < _N; ++i) {
        _Out(j, i) = static_cast<float>(_Src(j, i));
      }
    }
    
    double _Det = 1.0;

    // Initialize the pointer vector.
    std::array<size_t, _N> _RIdx{}; //row index
    for (size_t i = 0; i < _N; ++i)
      _RIdx[i] = i;

    // LU factorization.
    for (size_t p = 0; p < _N - 1; ++p) {
      //Find pivot element.
      for (size_t j = p + 1; j < _N; ++j) {
        if (cx_abs(_Out(_RIdx[j], p)) > cx_abs(_Out(_RIdx[p], p))) {
          // Switch the index for the p pivot row if necessary.;
          std::swap(_RIdx[j], _RIdx[p]);
          _Det = -_Det;
          //_RIdx[p] now has the index of the row to consider the pth
        }
      }

      if (_Out(_RIdx[p], p) == 0) {
        // The matrix is singular. //or not inversible by this methode untill fixed (no permutations)
        return std::tuple<bool, double, float_congruent_matrix, std::array<size_t, _N>>
          (false, NAN, _Out, { 0 });
      }
      // Multiply the diagonal elements.
      _Det *= _Out(_RIdx[p], p);

      // Form multiplier.
      for (size_t j = p + 1; j < _N; ++j) {
        _Out(_RIdx[j], p) /= _Out(_RIdx[p], p);
        // Eliminate [p].
        for (int i = p + 1; i < _N; ++i) {
          _Out(_RIdx[j], i) -= _Out(_RIdx[j], p) * _Out(_RIdx[p], i);
        }
      }
    }
    _Det *= _Out(_RIdx[_N - 1], _N - 1); //multiply last diagonal element
    
    const std::array<size_t, _N> _RI(_RIdx);

    //reorder output for simplicity
    for (size_t j = 0; j < _N; ++j) {
      if (j != _RIdx[j]) {
        for (size_t i = 0; i < _N; ++i) {
          std::swap(_Out(j, i), _Out(_RIdx[j], i));
        }
        std::swap(_RIdx[j], _RIdx[std::find(_RIdx.begin(), _RIdx.end(), j) - _RIdx.begin()]);
      }
    }

    return std::tuple<bool, double, float_congruent_matrix, std::array<size_t, _N>>
      (_Det != 0.0, _Det != 0.0 ? _Det : NAN, _Out, _RI);
  }

  template<class _Ty, size_t _N>
  [[nodiscard]] constexpr double
  determinant(const stack_matrix<_Ty, _N, _N>& _Src) {
    return std::get<1>(PII_LUDecomposition(_Src));
  }

  /*
  Reduced row echelon form
  Uses Gauss-Jordan elimination with partial pivoting
  Mutates input

  Can be used to solve systems of linear equations with an aumented matrix
  or to invert a matrix M :: ( M | I ) -> ( I | M^-1 )
  */
  template<class _Ty, size_t _M, size_t _N>
    requires (std::is_floating_point<_Ty>::value and (_M <= _N))
  [[nodiscard]] constexpr 
  bool RREF(stack_matrix<_Ty, _M, _N>& _Src) {

    using namespace cx_helper_func; //constexpr abs

    std::array<size_t, _M> _RIdx{};
    for (size_t i = 0; i < _M; ++i) {
      _RIdx[i] = i;
    }

    for (size_t p = 0; p < _M; ++p) {
      for (size_t j = p + 1; j < _M; ++j) {
        if (cx_abs(_Src(_RIdx[p], p)) < cx_abs(_Src(_RIdx[j], p))) {
          std::swap(_RIdx[p], _RIdx[j]);
        }
      }

      if (_Src(_RIdx[p], p) == 0) return false; //matrix is singular

      for (size_t i = p + 1; i < _N; ++i) {
        _Src(_RIdx[p], i) /= _Src(_RIdx[p], p);
      }
      _Src(_RIdx[p], p) = 1;

      for (size_t j = 0; j < _M; ++j) {
        if (j != p) {
          for (size_t i = p + 1; i < _N; ++i) { //p+1 to avoid removing each rows' scale factor
            _Src(_RIdx[j], i) -= _Src(_RIdx[p], i) * _Src(_RIdx[j], p);
          }
          _Src(_RIdx[j], p) = 0;
        }
      }
    }

    //reorder matrix
    for (size_t j = 0; j < _M; ++j) {
      if (j != _RIdx[j]) {
        for (size_t i = 0; i < _N; ++i) {
          std::swap(_Src(j, i), _Src(_RIdx[j], i));
        }
        std::swap(_RIdx[j], _RIdx[std::find(_RIdx.begin(), _RIdx.end(), j) - _RIdx.begin()]);
      }
    }
    return true;
  }

  /*
  Inverts N*N matrix using gauss-jordan reduction with pivoting
  Not the most efficient algorithm
  */
  template<class _Ty1, class _Ty2, size_t _N>
    requires std::is_floating_point<_Ty2>::value
  [[nodiscard]] constexpr 
  bool inverse(
      const stack_matrix<_Ty1, _N, _N>& _Src,
            stack_matrix<_Ty2, _N, _N>& _Dest) {

    stack_matrix<_Ty2, _N, _N * 2> _Temp{};

    //M
    for (size_t j = 0; j < _N; ++j) {
      for (size_t i = 0; i < _N; ++i) {
        _Temp(j, i) = static_cast<_Ty2>(_Src(j, i));
      }
    }
    //I
    for (size_t j = 0; j < _N; ++j) {
      _Temp(j, j + _N) = 1;
    }

    bool inv_{ 0 };

    if (inv_ = RREF(_Temp)) {
      for (size_t j = 0; j < _N; ++j) {
        for (size_t i = 0; i < _N; ++i) {
          _Dest(j, i) = _Temp(j, i + _N);
        }
      }
    }
    return inv_;
  }

  template<class _Ty, size_t _N>
  [[nodiscard]] constexpr 
  stack_matrix<_Ty, _N, _N> identity_matrix(void) {

    stack_matrix<_Ty, _N, _N> _Ret{};
    for (size_t i = 0; i < _N; ++i) {
      _Ret(i, i) = static_cast<_Ty>(1);
    }
    return _Ret;
  }

  template<class _Ty, size_t _N>
  [[nodiscard]] constexpr 
  stack_matrix<_Ty, _N, _N> transpose(const stack_matrix<_Ty, _N, _N>& _Src) {

    stack_matrix<_Ty, _N, _N> _Ret{ _Src };
    for (size_t j = 0; j < _N - 1; ++j) {
      for (size_t i = j + 1; i < _N; ++i) {
        std::swap(_Ret(j, i), _Ret(i, j));
      }
    }
    return _Ret;
  }

  template<class _Ty2, class _Ty1, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  stack_matrix<_Ty2, _M, _N> cast_to(const stack_matrix<_Ty1, _M, _N>& _Src) {

    stack_matrix<_Ty2, _M, _N> _Ret{};

    for (size_t j = 0; j < _M; ++j) {
      for (size_t i = 0; i < _N; ++i) {
        _Ret(j, i) = static_cast<_Ty2>(_Src(j, i));
      }
    }

    return _Ret;
  }

  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  stack_matrix<_Ty, _M, _N> element_wise_mul(
      const stack_matrix<_Ty, _M, _N>& _Mat1,
      const stack_matrix<_Ty, _M, _N>& _Mat2) {

    stack_matrix<_Ty, _M, _N> _Ret{ _Mat1 };
    for (size_t j = 0; j < _M; ++j) {
      for (size_t i = 0; i < _N; ++i) {
        _Ret(j, i) *= _Mat2(j, i);
      }
    }
    return _Ret;
  }

  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  bool nearly_equals(
      const stack_matrix<_Ty, _M, _N>& _Mat1,
      const stack_matrix<_Ty, _M, _N>& _Mat2,
      const _Ty epsilon = 1e-4) {

    _Ty d{};
    for (size_t j = 0; j < _M; ++j) {
      for (size_t i = 0; i < _N; ++i) {
        d = _Mat1(j, i) - _Mat2(j, i);
        if (cx_helper_func::cx_abs(d) > epsilon) return false;
      }
    }
    return true;
  }

  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  bool exactly_equals(
      const stack_matrix<_Ty, _M, _N>& _Mat1,
      const stack_matrix<_Ty, _M, _N>& _Mat2) {

    for (size_t j = 0; j < _M; ++j) {
      for (size_t i = 0; i < _N; ++i) {
        if (_Mat1(j, i) != _Mat2(j, i)) return false;
      }
    }
    return true;
  }

  //returns L1 distance divided by the number of elements
  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  float normaliced_L1_distance(
      const stack_matrix<_Ty, _M, _N>& _Mat1,
      const stack_matrix<_Ty, _M, _N>& _Mat2) {

    _Ty _L1{};

    for (size_t j = 0; j < _M; ++j) {
      for (size_t i = 0; i < _N; ++i) {
        _L1 += abs(_Mat1 - _Mat2);
      }
    }
    return static_cast<float>(_L1) / static_cast<float>(_M + _N);
  }

  /*
  returns the element-wise type consistent average of a pack of matrices
  beware of overflow issues if matrices have large numbers or calculating the average over a big array
  */
  template<class _Ty, size_t _M, size_t _N>
  [[nodiscard]] constexpr 
  stack_matrix<_Ty, _M, _N> matrix_average(
      const std::same_as<stack_matrix<_Ty, _M, _N>> auto& ... _Mats) {

    return (_Mats + ...) / sizeof...(_Mats);
  }


} /* namespace ga_sm */





