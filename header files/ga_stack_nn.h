#pragma once
#include "ga_stack_matrix.h"
#include <iostream>
#include <concepts>
#include <array>
#include <tuple>

namespace ga_sm_nn {
  using namespace ga_sm;

  //--------------------------------------------------------------------------------------//

  struct Layer_Shape {
    size_t M;
    size_t N;
    constexpr size_t size() const { return M * N; }
    constexpr bool operator==(const Layer_Shape&) const = default;
  };

  struct Layer_Structure {
    Layer_Shape Shape;
    size_t Inputs;
    size_t Outputs;
    
    constexpr bool operator==(const Layer_Structure&) const = default;
  };

  template<Layer_Shape... Shapes, size_t Size = sizeof...(Shapes)>
  constexpr [[nodiscard]]
  std::array<Layer_Structure, Size> get_structure() {
    std::array<Layer_Structure, Size> Ret{};
    std::array<Layer_Shape, Size+2> Temp{ Layer_Shape{1,1}, Shapes..., Layer_Shape{1,1} };

    for (size_t i = 1; i < Size+1; ++i) {
      Ret[i] = Layer_Structure{ Temp[i], Temp[i - 1].size() , Temp[i + 1].size()};
    }
    return Ret;
  }

  //--------------------------------------------------------------------------------------//

  template<class T, Layer_Structure Structure>
    requires std::is_floating_point<T>::value 
  class layer {
  private:
    static constexpr Layer_Shape Shape = Structure.Shape;
    static constexpr size_t Inputs = Structure.Inputs;
    static constexpr size_t Outputs = Structure.Outputs;
  public:

    //each neuron produces a column vector that splits into the input of the neurons in the next layer
    constexpr stack_matrix<T, Outputs, Shape.M* Shape.N> forward_pass(
      const stack_matrix<T, Shape.M* Shape.N, Inputs>& Input) const {

      auto weighted_input = multiple_vector_mul(Input, m_weigh_input_mat);
      return vector_expand(weighted_input, m_out_expand_mat) + m_out_offset_mat;
    }

    template<auto fn, auto... args>
    constexpr void init(const T scale = 1, const T offset = 0) {
      m_weigh_input_mat.fill<fn>(args...);
      m_out_expand_mat.fill<fn>(args...);
      m_out_offset_mat.fill<fn>(args...);
    }

    inline constexpr 
    const stack_matrix<T, Inputs, Shape.M* Shape.N>& get_weigh_input_mat() const {
      return m_weigh_input_mat;
    }

    inline constexpr 
    stack_matrix<T, Inputs, Shape.M* Shape.N>& get_weigh_input_mat() {
      return m_weigh_input_mat;
    }
    
    inline constexpr 
    const stack_matrix<T, Outputs, Shape.M* Shape.N>& get_out_expand_mat() const {
      return m_out_expand_mat;
    }

    inline constexpr 
    stack_matrix<T, Outputs, Shape.M* Shape.N>& get_out_expand_mat() {
      return m_out_expand_mat;
    }
    
    inline constexpr const 
    stack_matrix<T, Outputs, Shape.M* Shape.N>& get_out_offset_mat() const {
      return m_out_offset_mat;
    }

    inline constexpr stack_matrix<T, Outputs, Shape.M* Shape.N>& get_out_offset_mat() {
      return m_out_offset_mat;
    }

  private:
    stack_matrix<T, Inputs, Shape.M * Shape.N> m_weigh_input_mat{};
    stack_matrix<T, Outputs, Shape.M * Shape.N> m_out_expand_mat{};
    stack_matrix<T, Outputs, Shape.M * Shape.N> m_out_offset_mat{};

  };

  template<class T, Layer_Structure Structure>
  std::ostream& operator<<(
    std::ostream& os,
    const layer<T, Structure>& layer) {
    os << layer.get_weigh_input_mat() <<
      layer.get_out_expand_mat() <<
      layer.get_out_offset_mat() << "\n";
    return os;
  }

//-------------------------------------------------------------------------------
// 
//-------------------------------------------------------------------------------


  //base template
  template<typename T, size_t Inputs, Layer_Shape...>
  struct unroll;


  //specialization for last layer
  template<typename T, size_t Inputs, Layer_Shape Current_Shape>
  struct unroll<T, Inputs, Current_Shape>
  {
    layer < T, Layer_Structure{ Current_Shape, Inputs, 1 } > data{}; // one data member for this layer

    template<std::size_t index>
    auto& get()
    {
      if constexpr (index == 0){ // if its 0, return this layer
        return data;
      }
      else {
        static_assert(false && index, "requested layer index exceeds number of layers");
      }
    }

    template<auto fn, auto... args>
    void init() {
      data.init<fn, args...>();
    }
  };

  //general specialization
  template<typename T, size_t Inputs, Layer_Shape Current_Shape, Layer_Shape... Shapes>
  struct unroll<T, Inputs, Current_Shape, Shapes...>
  {
    layer < T, Layer_Structure{
      Current_Shape,
      Inputs,
      std::get<0>(std::forward_as_tuple(Shapes...)).size() } > data{}; // one data member for this layer

    unroll<T, Current_Shape.size(), Shapes...> next; //another unroll member for the rest

    //getter so that we can actually get a specific layer by index
    template<std::size_t index>
    auto& get(){
      if constexpr (index == 0) {// if its 0, return this layer
        return data;
      }
      else {
        return next.template get<index - 1>(); //if the index is not 0, ask the next layer with an updated index
      }
    }

    template<auto fn, auto... args>
    void init() {
      data.init<fn, args...>();
      next.init<fn, args...>();
    }
  };

  //--------------------------------------------------------------------------------------//

  //template<class T, Layer_Structure... Structures>
  //  requires ((sizeof...(Structures) >= 3) and std::is_floating_point<T>::value)
  //class stack_neural_net {
  //public:
  //  static constexpr size_t s_Layers{ sizeof...(Structures) };
  //  static constexpr std::array<Layer_Structure, s_Layers> s_Structures{ Structures... };

  //  std::tuple<layer<T, s_Structures>> m_Layers{};
  //};

  template<class T, Layer_Shape... Shapes>
    requires ((sizeof...(Shapes) >= 3) and std::is_floating_point<T>::value)
  class stack_neural_net {
  public:
    static constexpr size_t s_Layers{ sizeof...(Shapes) };
    static constexpr std::array<Layer_Shape, s_Layers> s_Shapes{ Shapes... };

    template<std::size_t Idx>
    auto& layer() {
      return m_Layers.template get<Idx>();
    } //you arent a real template programmer if you dont use .template  /s

    template<auto fn, auto... args>
    void init(){
      m_Layers.template init<fn, args...>();
    }

  private:
    unroll<T, 1, Shapes...> m_Layers{};
  };

  //--------------------------------------------------------------------------------------//

  //--------------------------------------------------------------------------------------//




}
