#pragma once
#include "ga_stack_matrix.h"
#include <iostream>
#include <concepts>
#include <array>

namespace ga_sm_nn {
  using namespace ga_sm;

  template<class _Ty, size_t... _Size>
    requires (((sizeof...(_Size) % 2) == 0) and 
               (sizeof...(_Size) >= 6) and 
                std::is_floating_point<_Ty>::value)
  class stack_neural_net {
  public:
    static constexpr std::array s_Size{ _Size... };
    static constexpr size_t s_Layers{ sizeof...(_Size) };

    stack_matrix<_Ty, s_Size[0], s_Size[1]> sm1{};

    stack_matrix<_Ty, s_Size[2], s_Size[3]> sm2{};

    stack_matrix<_Ty, s_Size[4], s_Size[5]> sm3{};

  };




}
