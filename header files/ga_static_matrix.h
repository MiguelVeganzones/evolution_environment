#pragma once
#include <iostream>
#include "Random.h"
#include "nn_helper_functions.h"
#include <concepts>
#include <stdexcept>
#include <cstring>
#include <immintrin.h>
#include <bit>

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

Code bloat could be an issue if using multiple types of matrices (shapes and value types). This has not 
been taken into account during design.

I decided to use - T const& - in most cases because i found it easier to read due to long names 

*/

namespace ga_sm {

  template <typename T, size_t Size>
  class matrix_const_iterator {
  public:
    using value_type       = T;
    using pointer          = const T*;
    using reference        = const T&;

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

  template <typename T, size_t Size>
  class matrix_iterator : public matrix_const_iterator<T, Size> {
  public:
    using Mybase = matrix_const_iterator<T, Size>;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;

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

    [[nodiscard]] constexpr pointer Unwrapped() const noexcept {
      return const_cast<pointer>(Mybase::Unwrapped());
    }
  };

  //-------------------------------------------------------------------//

  //-------------------------------------------------------------------//

  template <typename T, size_t M, size_t N>
    requires std::is_arithmetic_v<T>
  class static_matrix {
  public:
    using value_type            = T;
    using size_type             = size_t;
    using pointer               = T*;
    using const_pointer         = const T*;
    using reference             = T&;
    using const_reference       = const T&;
    using iterator              = matrix_iterator<T, M* N>;
    using const_iterator        = matrix_const_iterator<T, M* N>;
    using congruent_matrix      = static_matrix<T, M, N>;
    using row_matrix            = static_matrix<T, 1, N>;
    using column_matrix         = static_matrix<T, M, 1>;

    T m_Elems[M * N];

    /*--------------------------------------------*/

    constexpr void fill(const T& Val) {
      fill_n(iterator(m_Elems), M * N, Val);
    }

    template<auto fn, typename... Args>
    constexpr void fill(Args&&... args) {
      for (iterator Curr = begin(); Curr != end(); ++Curr) {
        *Curr = static_cast<T>(fn(std::forward<Args>(args)...));
      }
    }

    constexpr void fill_n(const iterator& Dest, const ptrdiff_t Count, const T Val) {
#ifdef _CHECKBOUNDS_
      bool indexing_b = *Dest < *m_Elems; //Pointer before array
      bool indexing_p = *Dest + Count > *m_Elems + M * N; //write past array limits
      if (indexing_b || indexing_p) {
        std::cout << "Indexing before array: " << indexing_b <<
                     "Indexing past array: " << indexing_p << std::endl;
        std::cout << "fill_n subscript out of bounds\n";
        exit(EXIT_FAILURE);
      }
#endif

      iterator Curr(Dest);
      for (ptrdiff_t i = 0; i < Count; ++i, ++Curr) {
        *Curr = Val;
      }
    }

    /*
    Each element of the matrix might be:
      Modified by a normal distribution given by [Avg, Stddev] with a probability of p1: // e += randnormal(avg, stddev)
      Replaced by a value given by fn scaled by Ampl2 with a probability of p2-p1: // e = fn(args...)
    */
    template<auto fn, auto... args>
      requires std::is_floating_point_v<T>
    void mutate(float Avg, float Stddev, float p1, float p2) {
      assert(p2 >= p1);

      float Rand;
      for (iterator Curr = begin(); Curr != end(); ++Curr) {
        Rand = random::randfloat();
        if (Rand < p1) {
          *Curr += static_cast<T>(random::randnormal(Avg, Stddev));
        }
        else if (Rand < p2) {
          *Curr = static_cast<T>(fn(args...));
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

    [[nodiscard]] constexpr row_matrix operator[](const size_t j) const {
      assert(j < M);
      row_matrix Ret{};
      const_iterator It(m_Elems, j * N);
      for (size_t i = 0; i < N; ++i, ++It) {
        Ret(0, i) = *It;
      }
      return Ret;
    }

    [[nodiscard]] constexpr 
    congruent_matrix operator+(congruent_matrix const& Other) const {
      congruent_matrix Ret(*this);
      for (size_t i = 0; i < M * N; ++i) {
        Ret.m_Elems[i] += Other.m_Elems[i];
      }
      return Ret;
    }

    [[nodiscard]] constexpr 
    congruent_matrix operator-(congruent_matrix const& Other) const {
      congruent_matrix Ret(*this);
      for (size_t i = 0; i < M * N; ++i) {
        Ret.m_Elems[i] -= Other.m_Elems[i];
      }
      return Ret;
    }

    [[nodiscard]] constexpr congruent_matrix operator*(const T p) const {
      congruent_matrix Ret(*this);
      for (auto& e : Ret) e *= p;
      return Ret;
    }

    constexpr congruent_matrix operator*=(const T p) noexcept {
      for (auto& e : *this) e *= p;
      return *this;
    }

    [[nodiscard]] constexpr congruent_matrix operator/(const T p) const {
      congruent_matrix Ret(*this);
      for (auto& e : Ret) e /= p;
      return Ret;
    }

    constexpr congruent_matrix operator/=(const T p) {
      assert(p != 0);
      for (auto& e : *this) e /= p;
      return *this;
    }


    /*-----------------------------------------------------------------
              ###		Utility		###
    ------------------------------------------------------------------*/

    [[nodiscard]] constexpr T sum() const noexcept {
      T sum{ 0 };
      for (auto& e : *this) sum += e;
      return sum;
    }

    //only works for float-like types
    constexpr void rescale_L_1_1_norm(const float Norm = 1.f) 
        requires std::is_floating_point_v<T> {
      assert(Norm != 0.f);
      const T Sum = sum();
      const T factor = Norm / Sum;
      for (auto& e : *this) e *= factor;
    }

    //Ofstream file should be closed outside of this function
    void store(std::ofstream& out) const {
      for (size_t j = 0; j < M; ++j) {
        for (size_t i = 0; i < N; ++i) {
          out << this->operator()(j, i) << " ";
        }
        out << '\n';
      }
      out << '\n';
    }

    //Overrides matrix with matrix read fron in
    //ifstream file should be closed outside this function
    void load(std::ifstream& in) {
      for (auto& e : m_Elems) in >> e;
    }

  };

  //------------------------------------------------------------------------------------------------

  //------------------------------------------------------------------------------------------------

  template<typename T2, typename T1, size_t M, size_t N>
  [[nodiscard]] constexpr
  static_matrix<T2, M, N> cast_to(static_matrix<T1, M, N> const& Src) {

    static_matrix<T2, M, N> Ret{};
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Ret(j, i) = static_cast<T2>(Src(j, i));
      }
    }
    return Ret;
  }

  template<size_t Out_M, size_t Out_N, typename T, size_t M, size_t N>
    requires (M * N == Out_M * Out_N)
  [[nodiscard]] constexpr
  static_matrix<T, Out_M, Out_N> cast_to_shape(static_matrix<T, M, N> const& Src) {

    static_matrix<T, Out_M, Out_N> Ret{};
    std::memcpy(Ret.m_Elems, Src.m_Elems, N * M * sizeof(T));
    return Ret;
  }

  template<typename T, size_t M, size_t N>
  std::ostream& operator<<(
      std::ostream& os, 
      static_matrix<T, M, N> const& Mat) {

    if (!std::is_integral<T>::value) {
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
  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr 
  bool operator==(
      static_matrix<T, M, N> const& Mat1,
      static_matrix<T, M, N> const& Mat2) {
    return std::is_integral_v<T> ? exactly_equals(Mat1, Mat2) : nearly_equals(Mat1, Mat2);
  }

  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr 
  bool operator!=(
      static_matrix<T, M, N> const& Mat1,
      static_matrix<T, M, N> const& Mat2) {
    return !operator==(Mat1, Mat2);
  }

  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr
    bool nearly_equals(
      static_matrix<T, M, N> const& Mat1,
      static_matrix<T, M, N> const& Mat2,
      const T epsilon = 1e-4) {

    T d{};
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        d = Mat1(j, i) - Mat2(j, i);
        if (cx_helper_func::cx_abs(d) > epsilon) return false;
      }
    }
    return true;
  }

  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr
    bool exactly_equals(
      static_matrix<T, M, N> const& Mat1,
      static_matrix<T, M, N> const& Mat2) {

    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        if (Mat1(j, i) != Mat2(j, i)) return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------------- 
  //------------ GA-Specific functionallity ------------------------------------- 
  //----------------------------------------------------------------------------- 

  template<typename T, size_t M, size_t N>
  void in_place_x_crossover(
      static_matrix<T, M, N>& Mat1,
      static_matrix<T, M, N>& Mat2) {

    //indices to slice. a: horizontal, b: vertical
    size_t a = random::randint(0, M);
    size_t b = random::randint(0, N);

    //return swapped original matrices to avoid irrelevant operations
    if ((a == 0 or a == M) and (b == 0 or b == N)) {
      return;
    }

    //minimize swaps -> less area
    size_t Area1 = a * b + (M - a) * (N - b);
    size_t Area2 = M * N - Area1;

    bool D = Area1 < Area2; //block diagonal to swap. True for main diagonal, false for anti diagonal 

    //swap first block
    for (size_t j = 0; j < a; ++j) {
      for (size_t i = (D ? 0 : b); i < (D ? b : N); ++i) {
        std::swap(Mat1(j, i), Mat2(j, i));
      }
    }

    //swap second block
    for (size_t j = a; j < M; ++j) {
      for (size_t i = (D ? b : 0); i < (D ? N : b); ++i) {
        std::swap(Mat1(j, i), Mat2(j, i));
      }
    }
  }

  template<typename Matrix>
  [[nodiscard]]
  std::pair<Matrix, Matrix> x_crossover(
    Matrix const& Mat1,
    Matrix const& Mat2) {

    //setup return matrices.
    Matrix Ret1(Mat1);
    Matrix Ret2(Mat2);

    in_place_x_crossover(Ret1, Ret2);

    return std::make_pair(Ret1, Ret2);
  }

  template<typename T, size_t M, size_t K, size_t N>
  [[nodiscard]] constexpr 
  static_matrix<T, M, N> matrix_mul(
      static_matrix<T, M, K> const& Mat1,
      static_matrix<T, K, N> const& Mat2) {

    static_matrix<T, M, N> Ret{};

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
  Beneficios observados a partir de 40x40
  */
  template<size_t M, size_t N>
  [[nosidcard]] 
  static_matrix<float, M, 1> matrix_vector_mul_float_avx(
      static_matrix<float, M, N> const& mat,
      static_matrix<float, N, 1> const& vec) {

    static_matrix<float, M, 1> ret{};
    for (size_t i = 0; i < N - 7; i += 8) {
      __m256 v_piece = _mm256_loadu_ps(vec.m_Elems + i);
      for (size_t j = 0; j < M; ++j) {
        __m256 row_piece = _mm256_loadu_ps(mat.m_Elems + i + j * N);
        __m256 m256_dot_piece = _mm256_dp_ps(v_piece, row_piece, 0xf1); //compare 0xff with 0xf1
        auto float8_dot_piece = std::bit_cast<std::array<float, 8>>(m256_dot_piece);
        ret(j, 0) += float8_dot_piece[0] + float8_dot_piece[4]; //Now without UB! std::bit_cast edition
      }                                                  
    }
    //last iterations (lower 3 bits of N)
    if constexpr (N % 8 != 0) {
      constexpr size_t iters_completed = ( N & (~size_t(7)) );
      for (size_t ii = iters_completed; ii < N; ++ii) {
        for (size_t j = 0; j < M; ++j) {
          ret(j, 0) += mat(j, ii) * vec(ii, 0);
        }
      }
    }
    return ret;
  }

  //multiplies every element of a row vectot by the column of a matrix of same index as the element
  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr
  static_matrix<T, M, N> vector_expand(
      static_matrix<T, 1, N> const& Row_vector,
      static_matrix<T, M, N> const& Col_vectors) {

    auto Ret{ Col_vectors };
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Ret(j, i) *= Row_vector(0, i);
      }
    }
    return Ret;
  }

  /*
  Multiplies pairs of vectors stored in two matrices
  Useful to make multiple vector multiplications with one operation and one data structure
  */
  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr
  static_matrix<T, 1, M> multiple_dot_product(
      static_matrix<T, M, N> const& Row_vectors,
      static_matrix<T, N, M> const& Col_vectors) {

    static_matrix<T, 1, M> Ret{};
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Ret(0, j) += Row_vectors(j, i) * Col_vectors(i, j);
      }
    }
    return Ret;
  }

  //----------------------------------------------------------------------------- 
  //------------ Math related functionality  ------------------------------------- 
  //----------------------------------------------------------------------------- 

  /*
  efficient LU decompoition of N by N matrix M
  returns:
    bool: det(M) != 0 //If matrix is inversible or not
    double: det(M)
    congruent_matrix LU: L is lower triangular with unit diagonal (implicit)
               U is upped diagonal, including diagonal elements

  */
  template<typename T, size_t N>
  [[nodiscard]] constexpr 
  std::tuple<bool, double, static_matrix<float, N, N>, std::array<size_t, N>>
  PII_LUDecomposition(static_matrix<T, N, N> const& Src)
  {
    /*
    source:
    http://web.archive.org/web/20150701223512/http://download.intel.com/design/PentiumIII/sml/24504601.pdf

    Factors "_Source" matrix into Out=LU where L is lower triangular and U is upper
    triangular. The matrix is overwritten by LU with the diagonal elements
    of L (which are unity) not stored. This must be a square n x n matrix.
    */

    using cx_helper_func::cx_abs; //constexpr abs
    using float_congruent_matrix = static_matrix<float, N, N>;

    float_congruent_matrix Out = cast_to<float>(Src);
    
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
        for (size_t i = p + 1; i < N; ++i) {
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
        std::swap(RIdx[j], RIdx[static_cast<size_t>(std::find(RIdx.begin(), RIdx.end(), j) - RIdx.begin())]);
      }
    }

    return std::tuple<bool, double, float_congruent_matrix, std::array<size_t, N>>
      (Det != 0.0, Det != 0.0 ? Det : NAN, Out, RI);
  }

  template<typename T, size_t N>
  [[nodiscard]] constexpr double
  determinant(static_matrix<T, N, N> const& Src) {
    return std::get<1>(PII_LUDecomposition(Src));
  }

  /*
  Reduced row echelon form
  Uses Gauss-Jordan elimination with partial pivoting
  Mutates input

  Can be used to solve systems of linear equations with an aumented matrix
  or to invert a matrix M : ( M | I ) -> ( I | M^-1 )
  */
  template<typename T, size_t M, size_t N>
    requires (std::is_floating_point_v<T> and (M <= N))
  [[nodiscard]] constexpr 
  bool RREF(static_matrix<T, M, N>& Src) {

    using cx_helper_func::cx_abs;//constexpr abs

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
        std::swap(RIdx[j], RIdx[static_cast<size_t>(std::find(RIdx.begin(), RIdx.end(), j) - RIdx.begin())]);
      }
    }
    return true;
  }

  /*
  Inverts N*N matrix using gauss-jordan reduction with pivoting
  Not the most efficient nor stable algorithm 
  Beware of stability problems with singular or close to singularity matrices
  */
  template<typename T1, typename T2 = double, size_t N>
    requires std::is_floating_point_v<T2>
  [[nodiscard]] constexpr 
  std::tuple<bool, static_matrix<T2, N, N>>  inverse(
      static_matrix<T1, N, N> const& Src) {
    // Tmp == ( M | I )
    static_matrix<T2, N, N * 2> Tmp{};
    //M
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Tmp(j, i) = static_cast<T2>(Src(j, i));
      }
    }
    //I
    for (size_t j = 0; j < N; ++j) {
      Tmp(j, j + N) = 1;
    }
    //Tmp = ( I | M^-1 )
    const bool Invertible = RREF(Tmp);
    static_matrix<T2, N, N> Inverse{};

    if (Invertible) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
          Inverse(j, i) = Tmp(j, i + N);
        }
      }
    }
    return { Invertible, Inverse };
  }

  template<typename T, size_t N>
  [[nodiscard]] constexpr 
  static_matrix<T, N, N> identity_matrix(void) {

    static_matrix<T, N, N> Ret{};
    for (size_t i = 0; i < N; ++i) {
      Ret(i, i) = static_cast<T>(1);
    }
    return Ret;
  }

  template<typename T, size_t N>
  [[nodiscard]] constexpr 
  static_matrix<T, N, N> transpose(static_matrix<T, N, N> const& Src) {

    static_matrix<T, N, N> Ret{ Src };
    for (size_t j = 0; j < N - 1; ++j) {
      for (size_t i = j + 1; i < N; ++i) {
        std::swap(Ret(j, i), Ret(i, j));
      }
    }
    return Ret;
  }

  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr 
  static_matrix<T, M, N> element_wise_mul(
      static_matrix<T, M, N> const& Mat1,
      static_matrix<T, M, N> const& Mat2) {

    static_matrix<T, M, N> Ret{ Mat1 };
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        Ret(j, i) *= Mat2(j, i);
      }
    }
    return Ret;
  }

  //returns L1 distance divided by the number of elements
  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr 
  double normaliced_L1_distance(
      static_matrix<T, M, N> const& Mat1,
      static_matrix<T, M, N> const& Mat2) {

    using cx_helper_func::cx_abs;
    double L1{};
    for (size_t j = 0; j < M; ++j) {
      for (size_t i = 0; i < N; ++i) {
        L1 += static_cast<double>(cx_abs(Mat1(j, i) - Mat2(j, i)));
      }
    }
    return L1 / static_cast<double>(M * N);
  }

  /*
  returns the element-wise type consistent average of a pack of matrices
  beware of overflow issues if matrices have large numbers or calculating the average over a big array
  */
  template<typename T, size_t M, size_t N>
  [[nodiscard]] constexpr 
  static_matrix<T, M, N> matrix_average(
      std::same_as<static_matrix<T, M, N>> auto const& ... mats) {
    return (mats + ...) / sizeof...(mats);
  }


} /* namespace ga_sm */

