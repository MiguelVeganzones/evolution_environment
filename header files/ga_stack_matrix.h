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

  template <class Ty, size_t Size>
  class matrix_const_iterator {
  public:
    using value_type = Ty;
    using ptrdiff_t = long long;
    using pointer = const Ty*;
    using reference = const Ty&;

    constexpr matrix_const_iterator() noexcept : m_Ptr(nullptr) {}

    constexpr explicit matrix_const_iterator(pointer Parg, size_t Offs = 0) noexcept : m_Ptr(Parg + Offs) {}

    [[nodiscard]] constexpr reference operator*() const noexcept {
      return *m_Ptr;
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept {
      return m_Ptr;
    }

    //++iter; retuns by reference mutated original object. Idem for the rest
    constexpr matrix_const_iterator& operator++() noexcept {
      ++m_Ptr;
      return *this;
    }

    //iter++; returns by value copy of the original iterator. Idem for the rest
    constexpr matrix_const_iterator operator++(int) noexcept {
      matrix_const_iterator Tmp = *this;
      ++m_Ptr;
      return Tmp;
    }

    //--iter;
    constexpr matrix_const_iterator& operator--() noexcept {
      --m_Ptr;
      return *this;
    }

    //iter--;
    constexpr matrix_const_iterator operator--(int) noexcept {
      matrix_const_iterator Tmp = *this;
      --m_Ptr;
      return Tmp;
    }

    constexpr matrix_const_iterator& operator+=(const ptrdiff_t Offs) noexcept {
      m_Ptr += Offs;
      return *this;
    }

    [[nodiscard]] constexpr matrix_const_iterator operator+(const ptrdiff_t Offs) const noexcept {
      matrix_const_iterator Tmp = *this;
      Tmp += Offs;
      return Tmp;
    }

    constexpr matrix_const_iterator& operator-=(const ptrdiff_t Offs) noexcept {
      m_Ptr -= Offs;
      return *this;
    }

    [[nodiscard]] constexpr matrix_const_iterator operator-(const ptrdiff_t Offs) const noexcept {
      matrix_const_iterator Tmp = *this;
      Tmp -= Offs;
      return Tmp;
    }

    [[nodiscard]] constexpr ptrdiff_t operator-(const matrix_const_iterator& Rhs) const noexcept {
      return m_Ptr - Rhs.m_Ptr;
    }

    [[nodiscard]] constexpr reference operator[](const ptrdiff_t Offs) const noexcept {
      return m_Ptr[Offs];
    }

    [[nodiscard]] constexpr bool operator==(const matrix_const_iterator& Rhs) const noexcept {
      return m_Ptr == Rhs.m_Ptr;
    }

    [[nodiscard]] constexpr bool operator!=(const matrix_const_iterator& Rhs) const noexcept {
      return !(*this == Rhs);
    }

    [[nodiscard]] constexpr bool operator>(const matrix_const_iterator& Rhs) const noexcept {
      return m_Ptr > Rhs.m_Ptr;
    }

    [[nodiscard]] constexpr bool operator<(const matrix_const_iterator& Rhs) const noexcept {
      return m_Ptr < Rhs.m_Ptr;
    }

    constexpr void Seek_to(pointer It) noexcept {
      m_Ptr = It;
    }

    constexpr void Unwrapped(pointer It) const noexcept {
      return m_Ptr;
    }

  private:
    pointer m_Ptr;
  };

  //-------------------------------------------------------------------//

  struct Matrix_Size {
    size_t M;
    size_t N;
    constexpr bool operator==(const Matrix_Size&) const = default;
  };

  template <class Ty, size_t Size>
  class matrix_iterator : public matrix_const_iterator<Ty, Size> {
  public:
    using Mybase = matrix_const_iterator<Ty, Size>;

    using value_type = Ty;
    using ptrdiff_t = long long;
    using pointer = Ty*;
    using reference = Ty&;

    constexpr matrix_iterator() noexcept {}

    constexpr explicit matrix_iterator(pointer Parg, size_t Offs = 0) noexcept : Mybase(Parg, Offs) {}

    [[nodiscard]] constexpr reference operator*() const noexcept {
      return const_cast<reference>(Mybase::operator*());
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept {
      return const_cast<pointer>(Mybase::operator->());
    }

    constexpr matrix_iterator& operator++() noexcept {
      Mybase::operator++();
      return *this;
    }

    constexpr matrix_iterator operator++(int) noexcept {
      matrix_iterator Tmp = *this;
      Mybase::operator++();
      return Tmp;
    }

    constexpr matrix_iterator& operator--() noexcept {
      Mybase::operator--();
      return *this;
    }

    constexpr matrix_iterator operator--(int) noexcept {
      matrix_iterator Tmp = *this;
      Mybase::operator--();
      return Tmp;
    }

    constexpr matrix_iterator& operator+=(const ptrdiff_t Offs) noexcept {
      Mybase::operator+=(Offs);
      return *this;
    }

    [[nodiscard]] constexpr matrix_iterator operator+(const ptrdiff_t Offs) const noexcept {
      matrix_iterator Tmp = *this;
      Tmp += Offs;
      return Tmp;
    }

    constexpr matrix_iterator& operator-=(const ptrdiff_t Offs) noexcept {
      Mybase::operator-=(Offs);
      return *this;
    }

    [[nodiscard]] constexpr matrix_iterator operator-(const ptrdiff_t Offs) const noexcept {
      matrix_iterator Tmp = *this;
      Tmp -= Offs;
      return Tmp;
    }

    [[nodiscard]] constexpr reference operator[](const ptrdiff_t Offs) const noexcept {
      return const_cast<reference>(Mybase::operator[](Offs));
    }

    [[nodiscard]] constexpr pointer _Unwrapped() const noexcept {
      return const_cast<pointer>(Mybase::_Unwrapped());
    }
  };

  //-------------------------------------------------------------------//

  template <class Ty, size_t M, size_t N>
  class stack_matrix {
  public:
    using value_type            = Ty;
    using size_type             = size_t;
    using ptrdiff_t             = long long;
    using pointer               = Ty*;
    using const_pointer         = const Ty*;
    using reference             = Ty&;
    using const_reference       = const Ty&;
    using iterator              = matrix_iterator<Ty, M* N>;
    using const_iterator        = matrix_const_iterator<Ty, M* N>;
    using congruent_matrix      = stack_matrix<Ty, M, N>;
    using row_matrix            = stack_matrix<Ty, 1, N>;
    using column_matrix         = stack_matrix<Ty, M, 1>;

    Ty m_Elems[M * N];

    /*--------------------------------------------*/

    constexpr void fill(const Ty& Val) {
      fill_n(iterator(m_Elems), M * N, Val);
    }

    template<auto fn, typename... Args>
    constexpr void fill(Args&&... args) {
      for (iterator Curr = begin(); Curr != end(); ++Curr) {
        *Curr = static_cast<Ty>(fn(std::forward<Args>(args)...));
      }
    }

    /*
    Each element of the matrix might be:
      Modifies by a normal distribution given by [Avg, Stddev] with a probability of p1: // e += randnoral(avg, stddev)
      Replaced by a value given by fn scaled by _Ampl2 with a probability of p2-p1: // e = (fn(args...) + offset) * ampl
    */
    template<auto fn, auto... args>
      requires std::is_floating_point<Ty>::value
    void mutate(float Avg, float Stddev, float p1, float p2, float Ampl = 1, float Offset = -0.5) {
      assert(p2 >= p1);

      float Rand;
      for (iterator Curr = begin(); Curr != end(); ++Curr) {
        Rand = random::randfloat();
        if (Rand < p1) {
          *Curr += static_cast<Ty>(random::randnormal(Avg, Stddev));
        }
        else if (Rand < p2) {
          *Curr = static_cast<Ty>((static_cast<float>(fn(args...)) + Offset) * Ampl);
        }
      }
    }

    constexpr void fill_n(const iterator& Dest, const ptrdiff_t Count, const Ty Val) {
#ifdef _CHECKBOUNDS_
      bool a = *Dest < *m_Elems; //Pointer before array
      bool b = *Dest + Count > *m_Elems + M * N; //write past array limits
      if (a || b) {
        std::cout << "a: " << a <<
          "b: " << b << std::endl;
        throw("fill_n subscript out of bounds\n");
      }
#endif

      iterator Curr(Dest);
      for (ptrdiff_t i = 0; i < Count; ++i, ++Curr) {
        *Curr = Val;
      }
    }

    //in place transpose operator
    constexpr void T() noexcept {
      for (size_t j = 0; j < M - 1; ++j) {
        for (size_t i = j + 1; i < N; ++i) {
          std::swap(this->operator()(j, i), this->operator()(i, j));
        }
      }
    }

    [[nodiscard]] constexpr iterator begin() noexcept {
      return iterator(m_Elems);
    }

    [[nodiscard]] constexpr const_iterator begin() const noexcept {
      return const_iterator(m_Elems);
    }

    [[nodiscard]] constexpr iterator end() noexcept {
      return iterator(m_Elems, M * N);
    }

    [[nodiscard]] constexpr const_iterator end() const noexcept {
      return const_iterator(m_Elems, M * N);
    }

    [[nodiscard]] inline constexpr 
    reference operator()(const size_t j, const size_t i) noexcept {
      assert(j < M and i < N);
      return m_Elems[j * N + i];
    }

    [[nodiscard]] inline constexpr 
    const_reference operator()(const size_t j, const size_t i) const noexcept {
      assert(j < M and i < N);
      return m_Elems[j * N + i];
    }

    [[nodiscard]] constexpr row_matrix operator[](size_t j) const {
      assert(j < M);
      stack_matrix<Ty, 1, N> Ret{};
      const_iterator It(m_Elems, j * N);
      for (size_t i = 0; i < N; ++i, ++It) {
        Ret(0, i) = *It;
      }
      return Ret;
    }

    [[nodiscard]] constexpr 
    congruent_matrix operator+(const congruent_matrix& Other) const {
      congruent_matrix Ret(*this);
      for (size_t i = 0; i < M * N; ++i) {
        Ret.m_Elems[i] += Other.m_Elems[i];
      }
      return Ret;
    }

    [[nodiscard]] constexpr 
    congruent_matrix operator-(const congruent_matrix& Other) const {
      congruent_matrix Ret(*this);
      for (size_t i = 0; i < M * N; ++i) {
        Ret.m_Elems[i] -= Other.m_Elems[i];
      }
      return Ret;
    }

    [[nodiscard]] constexpr congruent_matrix operator*(const Ty p) const {
      congruent_matrix Ret(*this);
      for (auto& e : Ret) e *= p;
      return Ret;
    }

    constexpr congruent_matrix operator*=(const Ty p) noexcept {
      for (auto& e : *this) e *= p;
      return *this;
    }

    [[nodiscard]] constexpr congruent_matrix operator/(const Ty p) const {
      congruent_matrix Ret(*this);
      for (auto& e : Ret) e /= p;
      return Ret;
    }

    constexpr congruent_matrix operator/=(const Ty p) {
      assert(p != 0);
      for (auto& e : *this) e /= p;
      return *this;
    }


    /*-----------------------------------------------------------------
              ###		Utility		###
    ------------------------------------------------------------------*/

    [[nodiscard]] constexpr Ty sum() const noexcept {
      Ty sum{ 0 };
      for (auto& e : *this) sum += e;
      return sum;
    }

    //only works for float-like types
    constexpr void rescale_L_1_1_norm(const float Norm = 1.f) {
      assert(std::is_floating_point<Ty>::value);
      assert(Norm != 0.f);
      const Ty Sum = sum();
      for (auto& e : *this) e /= Sum;
    }

  };

  template<class Ty, size_t M, size_t N>
  std::ostream& operator<<(
      std::ostream& os, 
      const stack_matrix<Ty, M, N>& Mat) {

    if (!std::is_integral<Ty>::value) {
      os << std::fixed;
      os << std::setprecision(4);
    }
    std::cout << "[";
    for (size_t j = 0; j < M; ++j) {
      std::cout << "\n [ ";
      for (size_t i = 0; i < N; ++i) {
        os << Mat(j, i) << " ";
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
  template<class Ty, size_t M, size_t N>
  [[nodiscard]] constexpr 
  bool operator==(
      const stack_matrix<Ty, M, N>& Mat1,
      const stack_matrix<Ty, M, N>& Mat2) {
    return std::is_integral<Ty>::value ? exactly_equals(Mat1, Mat2) : nearly_equals(Mat1, Mat2);
  }

  template<class Ty, size_t M, size_t N>
  [[nodiscard]] constexpr 
  bool operator!=(
      const stack_matrix<Ty, M, N>& Mat1,
      const stack_matrix<Ty, M, N>& Mat2) {
    return !operator==(Mat1, Mat2);
  }

  template<class Ty, size_t M, size_t N>
  [[nodiscard]] 
  std::pair<stack_matrix<Ty, M, N>, stack_matrix<Ty, M, N>> x_crossover(
      const stack_matrix<Ty, M, N>& Mat1,
      const stack_matrix<Ty, M, N>& Mat2) {

    //indices to slice. a: horizontal, b: vertical
    size_t a = random::randint(0, M);
    size_t b = random::randint(0, N);

    //return swapped original matrices to avoid irrelevant operations
    if ((a == 0 or a == M) and (b == 0 or b == N)) return std::pair(Mat2, Mat1);

    //minimize swaps -> less area
    size_t Area1 = a * b + (M - a) * (N - b);
    size_t Area2 = M * N - Area1;

    bool D = Area1 < Area2; //block diagonal to swap. True for main diagonal, false for anti diagonal 
    int count = 0;

    //setup return matrices.
    stack_matrix<Ty, M, N> Ret1(Mat1);
    stack_matrix<Ty, M, N> Ret2(Mat2);

    //swap first block
    for (size_t j = 0; j < a; ++j) {
      for (size_t i = (D ? 0 : b); i < (D ? b : N); ++i) {
        std::swap(Ret1(j, i), Ret2(j, i));
      }
    }

    //swap second block
    for (size_t j = a; j < M; ++j) {
      for (size_t i = (D ? b : 0); i < (D ? N : b); ++i) {
        std::swap(Ret1(j, i), Ret2(j, i));
      }
    }

    return std::make_pair(Ret1, Ret2);
  }

  template<class Ty, size_t M, size_t K, size_t N>
  [[nodiscard]] constexpr 
  stack_matrix<Ty, M, N> matrix_mul(
      const stack_matrix<Ty, M, K>& Mat1,
      const stack_matrix<Ty, K, N>& Mat2) {

    stack_matrix<Ty, M, N> Ret{};

    for (size_t j = 0; j < M; ++j) {
      for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < N; ++i) {
          Ret(j, i) += Mat1(j, k) * Mat2(k, i);
        }
      }
    }
    return Ret;
  }

  /*
  efficient LU decompoition of N by N matrix M
  returns:
    bool: det(M) != 0 //If matrix is inversible or not
    double: det(M)
    congruent_matrix LU: L is lower triangular with unit diagonal (implicit)
               U is upped diagonal, including diagonal elements

  */
  template<class Ty, size_t N>
  [[nodiscard]] constexpr 
  std::tuple<bool, double, stack_matrix<float, N, N>, std::array<size_t, N>>
  PII_LUDecomposition(const stack_matrix<Ty, N, N>& Src)
  {
    /*
    source:
    http://web.archive.org/web/20150701223512/http://download.intel.com/design/PentiumIII/sml/24504601.pdf

    Factors "_Source" matrix into Out=LU where L is lower triangular and U is upper
    triangular. The matrix is overwritten by LU with the diagonal elements
    of L (which are unity) not stored. This must be a square n x n matrix.
    */

    using namespace cx_helper_func;//constexpr abs
    using float_congruent_matrix = stack_matrix<float, N, N>;

    float_congruent_matrix Out{};
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Out(j, i) = static_cast<float>(Src(j, i));
      }
    }
    
    double Det = 1.0;

    // Initialize the pointer vector.
    std::array<size_t, N> RIdx{}; //row index
    for (size_t i = 0; i < N; ++i)
      RIdx[i] = i;

    // LU factorization.
    for (size_t p = 0; p < N - 1; ++p) {
      //Find pivot element.
      for (size_t j = p + 1; j < N; ++j) {
        if (cx_abs(Out(RIdx[j], p)) > cx_abs(Out(RIdx[p], p))) {
          // Switch the index for the p pivot row if necessary.;
          std::swap(RIdx[j], RIdx[p]);
          Det = -Det;
          //RIdx[p] now has the index of the row to consider the pth
        }
      }

      if (Out(RIdx[p], p) == 0) {
        // The matrix is singular. //or not inversible by this methode untill fixed (no permutations)
        return std::tuple<bool, double, float_congruent_matrix, std::array<size_t, N>>
          (false, NAN, Out, { 0 });
      }
      // Multiply the diagonal elements.
      Det *= Out(RIdx[p], p);

      // Form multiplier.
      for (size_t j = p + 1; j < N; ++j) {
        Out(RIdx[j], p) /= Out(RIdx[p], p);
        // Eliminate [p].
        for (int i = p + 1; i < N; ++i) {
          Out(RIdx[j], i) -= Out(RIdx[j], p) * Out(RIdx[p], i);
        }
      }
    }
    Det *= Out(RIdx[N - 1], N - 1); //multiply last diagonal element
    
    const std::array<size_t, N> RI(RIdx);

    //reorder output for simplicity
    for (size_t j = 0; j < N; ++j) {
      if (j != RIdx[j]) {
        for (size_t i = 0; i < N; ++i) {
          std::swap(Out(j, i), Out(RIdx[j], i));
        }
        std::swap(RIdx[j], RIdx[std::find(RIdx.begin(), RIdx.end(), j) - RIdx.begin()]);
      }
    }

    return std::tuple<bool, double, float_congruent_matrix, std::array<size_t, N>>
      (Det != 0.0, Det != 0.0 ? Det : NAN, Out, RI);
  }

  template<class Ty, size_t N>
  [[nodiscard]] constexpr double
  determinant(const stack_matrix<Ty, N, N>& Src) {
    return std::get<1>(PII_LUDecomposition(Src));
  }

  /*
  Reduced row echelon form
  Uses Gauss-Jordan elimination with partial pivoting
  Mutates input

  Can be used to solve systems of linear equations with an aumented matrix
  or to invert a matrix M :: ( M | I ) -> ( I | M^-1 )
  */
  template<class Ty, size_t M, size_t N>
    requires (std::is_floating_point<Ty>::value and (M <= N))
  [[nodiscard]] constexpr 
  bool RREF(stack_matrix<Ty, M, N>& Src) {

    using namespace cx_helper_func; //constexpr abs

    std::array<size_t, M> RIdx{};
    for (size_t i = 0; i < M; ++i) {
      RIdx[i] = i;
    }

    for (size_t p = 0; p < M; ++p) {
      for (size_t j = p + 1; j < M; ++j) {
        if (cx_abs(Src(RIdx[p], p)) < cx_abs(Src(RIdx[j], p))) {
          std::swap(RIdx[p], RIdx[j]);
        }
      }

      if (Src(RIdx[p], p) == 0) return false; //matrix is singular

      for (size_t i = p + 1; i < N; ++i) {
        Src(RIdx[p], i) /= Src(RIdx[p], p);
      }
      Src(RIdx[p], p) = 1;

      for (size_t j = 0; j < M; ++j) {
        if (j != p) {
          for (size_t i = p + 1; i < N; ++i) { //p+1 to avoid removing each rows' scale factor
            Src(RIdx[j], i) -= Src(RIdx[p], i) * Src(RIdx[j], p);
          }
          Src(RIdx[j], p) = 0;
        }
      }
    }

    //reorder matrix
    for (size_t j = 0; j < M; ++j) {
      if (j != RIdx[j]) {
        for (size_t i = 0; i < N; ++i) {
          std::swap(Src(j, i), Src(RIdx[j], i));
        }
        std::swap(RIdx[j], RIdx[std::find(RIdx.begin(), RIdx.end(), j) - RIdx.begin()]);
      }
    }
    return true;
  }

  /*
  Inverts N*N matrix using gauss-jordan reduction with pivoting
  Not the most efficient algorithm
  */
  template<class Ty1, class Ty2, size_t N>
    requires std::is_floating_point<Ty2>::value
  [[nodiscard]] constexpr 
  bool inverse(
      const stack_matrix<Ty1, N, N>& Src,
            stack_matrix<Ty2, N, N>& Dest) {

    stack_matrix<Ty2, N, N * 2> Tmp{};

    //M
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Tmp(j, i) = static_cast<Ty2>(Src(j, i));
      }
    }
    //I
    for (size_t j = 0; j < N; ++j) {
      Tmp(j, j + N) = 1;
    }

    bool Invertible{ 0 };

    if (Invertible = RREF(Tmp)) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
          Dest(j, i) = Tmp(j, i + N);
        }
      }
    }
    return Invertible;
  }

  template<class Ty, size_t N>
  [[nodiscard]] constexpr 
  stack_matrix<Ty, N, N> identity_matrix(void) {

    stack_matrix<Ty, N, N> Ret{};
    for (size_t i = 0; i < N; ++i) {
      Ret(i, i) = static_cast<Ty>(1);
    }
    return Ret;
  }

  template<class Ty, size_t N>
  [[nodiscard]] constexpr 
  stack_matrix<Ty, N, N> transpose(const stack_matrix<Ty, N, N>& Src) {

    stack_matrix<Ty, N, N> Ret{ Src };
    for (size_t j = 0; j < N - 1; ++j) {
      for (size_t i = j + 1; i < N; ++i) {
        std::swap(Ret(j, i), Ret(i, j));
      }
    }
    return Ret;
  }

  template<class Ty2, class Ty1, size_t M, size_t N>
  [[nodiscard]] constexpr 
  stack_matrix<Ty2, M, N> cast_to(const stack_matrix<Ty1, M, N>& Src) {

    stack_matrix<Ty2, M, N> Ret{};

    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Ret(j, i) = static_cast<Ty2>(Src(j, i));
      }
    }

    return Ret;
  }

  template<class Ty, size_t M, size_t N>
  [[nodiscard]] constexpr 
  stack_matrix<Ty, M, N> element_wise_mul(
      const stack_matrix<Ty, M, N>& Mat1,
      const stack_matrix<Ty, M, N>& Mat2) {

    stack_matrix<Ty, M, N> Ret{ Mat1 };
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Ret(j, i) *= Mat2(j, i);
      }
    }
    return Ret;
  }

  template<class Ty, size_t M, size_t N>
  [[nodiscard]] constexpr 
  bool nearly_equals(
      const stack_matrix<Ty, M, N>& Mat1,
      const stack_matrix<Ty, M, N>& Mat2,
      const Ty epsilon = 1e-4) {

    Ty d{};
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        d = Mat1(j, i) - Mat2(j, i);
        if (cx_helper_func::cx_abs(d) > epsilon) return false;
      }
    }
    return true;
  }

  template<class Ty, size_t M, size_t N>
  [[nodiscard]] constexpr 
  bool exactly_equals(
      const stack_matrix<Ty, M, N>& Mat1,
      const stack_matrix<Ty, M, N>& Mat2) {

    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        if (Mat1(j, i) != Mat2(j, i)) return false;
      }
    }
    return true;
  }

  //returns L1 distance divided by the number of elements
  template<class Ty, size_t M, size_t N>
  [[nodiscard]] constexpr 
  float normaliced_L1_distance(
      const stack_matrix<Ty, M, N>& Mat1,
      const stack_matrix<Ty, M, N>& Mat2) {

    Ty L1{};

    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        L1 += abs(Mat1 - Mat2);
      }
    }
    return static_cast<float>(L1) / static_cast<float>(M + N);
  }

  /*
  returns the element-wise type consistent average of a pack of matrices
  beware of overflow issues if matrices have large numbers or calculating the average over a big array
  */
  template<class Ty, size_t M, size_t N>
  [[nodiscard]] constexpr 
  stack_matrix<Ty, M, N> matrix_average(
      const std::same_as<stack_matrix<Ty, M, N>> auto& ... Mats) {

    return (Mats + ...) / sizeof...(Mats);
  }


} /* namespace ga_sm */





