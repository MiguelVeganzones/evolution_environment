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
    inline constexpr size_t size() const { return M * N; }
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
    static constexpr Layer_Shape s_Shape = Structure.Shape;
    static constexpr size_t s_Inputs = Structure.Inputs;
    static constexpr size_t s_Outputs = Structure.Outputs;

    stack_matrix<T, s_Inputs, s_Shape.size()> m_weigh_input_mat{};
    stack_matrix<T, s_Outputs, s_Shape.size()> m_out_expand_mat{};
    stack_matrix<T, s_Outputs, s_Shape.size()> m_out_offset_mat{};

  public:

    //each neuron produces a column vector that splits into the input of the neurons in the next layer
    [[nodiscard]]constexpr 
    stack_matrix<T, s_Outputs, s_Shape.size()> forward_pass(
        const stack_matrix<T, s_Shape.size(), s_Inputs>& Input) const {
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
    const stack_matrix<T, s_Inputs, s_Shape.size()>& get_weigh_input_mat() const {
      return m_weigh_input_mat;
    }

    inline constexpr 
    stack_matrix<T, s_Inputs, s_Shape.size()>& get_weigh_input_mat() {
      return m_weigh_input_mat;
    }
    
    inline constexpr 
    const stack_matrix<T, s_Outputs, s_Shape.size()>& get_out_expand_mat() const {
      return m_out_expand_mat;
    }

    inline constexpr 
    stack_matrix<T, s_Outputs, s_Shape.size()>& get_out_expand_mat() {
      return m_out_expand_mat;
    }
    
    inline constexpr 
    const stack_matrix<T, s_Outputs, s_Shape.size()>& get_out_offset_mat() const {
      return m_out_offset_mat;
    }

    inline constexpr stack_matrix<T, s_Outputs, s_Shape.size()>& get_out_offset_mat() {
      return m_out_offset_mat;
    }

    [[nodiscard]] inline constexpr Layer_Shape get_shape() const {
      return s_Shape;
    }
    
    [[nodiscard]] inline constexpr Layer_Shape get_inputs() const {
      return s_Inputs;
    }

    [[nodiscard]] inline constexpr Layer_Shape get_outputs() const {
      return s_Outputs;
    }
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
    //-------- STATIC DATA ---------//
    static constexpr Layer_Shape s_Shape{ Current_Shape };
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ 1 };

    //-------- DATA MEMBERS -------//
    layer < T, Layer_Structure{ Current_Shape, Inputs, 1 } > m_Data{}; // one data member for this layer

    //------ MEMBER FUNCTIONS -----//
    template<std::size_t Idx>
    constexpr auto& get()
    {
      if constexpr (Idx == 0){ // if its 0, return this layer
        return m_Data;
      }
      else {
        static_assert(false && Idx, "requested layer index exceeds number of layers");
      }
    }

    template<auto fn, auto... args>
    void init() {
      m_Data.init<fn, args...>();
    }

    void print() const {
      std::cout << m_Data << "\n";
    }

    template<size_t Outputs>
      requires (Outputs == s_Outputs)
    stack_matrix<T, s_Outputs, s_Shape.size()> forward_pass(
        const stack_matrix<T, s_Shape.size(), s_Inputs>& input_data) const {
      return m_Data.forward_pass(input_data);
    }

  };

  //general specialization
  template<typename T, size_t Inputs, Layer_Shape Current_Shape, Layer_Shape... Shapes>
  struct unroll<T, Inputs, Current_Shape, Shapes...>
  {
    //-------- STATIC DATA ---------//
    static constexpr Layer_Shape s_Shape{ Current_Shape };
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ std::get<0>(std::forward_as_tuple(Shapes...)).size() };

    //-------- DATA MEMBERS -------//
    layer < T, Layer_Structure{
      s_Shape,
      s_Inputs,
      s_Outputs } > m_Data{}; // one data member for this layer

    unroll<T, Current_Shape.size(), Shapes...> m_Next; //another unroll member for the rest

    //------ MEMBER FUNCTIONS -----//
    //getter so that we can actually get a specific layer by index
    template<std::size_t Idx>
    constexpr auto& get(){
      if constexpr (Idx == 0) {// if its 0, return this layer
        return m_Data;
      }
      else {
        return m_Next.template get<Idx - 1>(); //if the index is not 0, ask the next layer with an updated index
      }
    }

    template<auto fn, auto... args>
    void init() {
      m_Data.init<fn, args...>();
      m_Next.init<fn, args...>();
    }

    void print() const {
      std::cout << m_Data << "\n";
      m_Next.print();
    }

    template<size_t Outputs>
    stack_matrix<T, 1, Outputs> forward_pass(
        const stack_matrix<T, s_Shape.size(), s_Inputs>& input_data) const {
      return m_Next.forward_pass<Outputs>(m_Data.forward_pass(input_data));
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
  private:
    static constexpr size_t s_Layers{ sizeof...(Shapes) };
    static constexpr std::array<Layer_Shape, s_Layers> s_Shapes{ Shapes... };
    static constexpr Layer_Shape s_Out_Shape{ s_Shapes[s_Layers - 1] };

    unroll<T, 1, Shapes...> m_Layers{};

  public:

    constexpr size_t parameters() const {
      size_t p = s_Shapes[0].size() + 2 * s_Shapes[s_Layers - 1].size(); //head and tail layers
      for (size_t i = 0; i < s_Layers - 1; ++i) {
        p += 3 * s_Shapes[i].size() * s_Shapes[i + 1].size();
      }
      return p;
    }
   
    template<std::size_t Idx>
    auto& layer() {
      return m_Layers.template get<Idx>();
    } //you arent a real template programmer if you dont use .template  /s

    template<auto fn, auto... args>
    void init(){
      m_Layers.template init<fn, args...>();
    }

    void print_layers() const {
      std::ios_base::sync_with_stdio(false);
      m_Layers.print();
      std::ios_base::sync_with_stdio(true);
    }

    void print_net() const {
      print_layers();
    }

    template<size_t M, size_t N>
      requires (Layer_Shape{ M, N } == s_Shapes[0])
    stack_matrix<T, 1, s_Out_Shape.size()> forward_pass(
        const stack_matrix<T, M, N>& input_data) const {
      //format input
      stack_matrix<T, s_Shapes[0].size(), 1> Temp{};
      for (size_t j = 0; j < M; ++j) {
        for (size_t i = 0; i < N; ++i) {
          Temp(j * N + i, 0) = input_data(j, i);
        }
      }
      return m_Layers.forward_pass<s_Out_Shape.size()>(Temp);
    }

  };

  //--------------------------------------------------------------------------------------//

  //--------------------------------------------------------------------------------------//




}
