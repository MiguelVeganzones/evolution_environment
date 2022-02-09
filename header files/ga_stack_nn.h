#pragma once
#include "ga_stack_matrix.h"
#include <iostream>
#include <concepts>
#include <array>

namespace ga_sm_nn {

  template<size_t... _Size>
    requires (((sizeof...(_Size) % 2) == 0) and (sizeof...(_Size) >= 6))
  class sm_nn {
  private:
    std::array<size_t, sizeof...(_Size)> m_Size{ _Size, ... };
  };




}
