#pragma once
#include "ga_static_matrix.h"
#include <iostream>
#include <concepts>
#include <array>
#include <tuple>
#include <fstream>
#include <stdexcept>

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

  //--------------------------------------------------------------------------------------//

  template<typename T, Layer_Structure Structure>
    requires std::is_floating_point<T>::value 
  class layer {
  private:
    static constexpr Layer_Shape s_Shape = Structure.Shape;
    static constexpr size_t s_Inputs = Structure.Inputs;
    static constexpr size_t s_Outputs = Structure.Outputs;

  public:
    using input_mat_shape = static_matrix<T, s_Inputs, s_Shape.size()>;
    using output_mat_shape = static_matrix<T, s_Outputs, s_Shape.size()>;

  private:
    input_mat_shape m_weigh_input_mat{};
    output_mat_shape m_out_expand_mat{};
    output_mat_shape m_out_offset_mat{};

  public:
    //each neuron produces a column vector that splits into the input of the neurons in the next layer
    [[nodiscard]]constexpr 
    output_mat_shape forward_pass(
        const static_matrix<T, s_Shape.size(), s_Inputs>& Input) const {

      const auto weighted_input = multiple_dot_product(Input, m_weigh_input_mat);
      return vector_expand(weighted_input, m_out_expand_mat) + m_out_offset_mat;
    }

    template<auto fn, auto... args>
    constexpr void init() {
      m_weigh_input_mat.fill<fn>(args...);
      m_out_expand_mat.fill<fn>(args...);
      m_out_offset_mat.fill<fn>(args...);
    }

    void store(std::ofstream& out) const {
      m_weigh_input_mat.store(out);
      m_out_expand_mat.store(out);
      m_out_offset_mat.store(out);
    }

    void load(std::ifstream& in) {
      m_weigh_input_mat.load(in);
      m_out_expand_mat.load(in);
      m_out_offset_mat.load(in);
    }

    inline constexpr 
    const input_mat_shape& get_weigh_input_mat() const {
      return m_weigh_input_mat;
    }

    inline constexpr 
    input_mat_shape& get_weigh_input_mat() {
      return m_weigh_input_mat;
    }
    
    inline constexpr 
    const output_mat_shape& get_out_expand_mat() const {
      return m_out_expand_mat;
    }

    inline constexpr 
    output_mat_shape& get_out_expand_mat() {
      return m_out_expand_mat;
    }
    
    inline constexpr 
    const output_mat_shape& get_out_offset_mat() const {
      return m_out_offset_mat;
    }

    inline constexpr 
    output_mat_shape& get_out_offset_mat() {
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

  template<typename T, Layer_Structure Structure>
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
  struct layer_unroll;


  //specialization for last layer
  template<typename T, size_t Inputs, Layer_Shape Current_Shape>
  struct layer_unroll<T, Inputs, Current_Shape>
  {
    static constexpr Layer_Shape s_Shape{ Current_Shape };
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ 1 };

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

    void store(std::ofstream& out) const {
      m_Data.store(out);
    }

    void load(std::ifstream& in) {
      m_Data.load(in);
    }

    template<size_t Outputs>
      requires (Outputs == s_Outputs)
    static_matrix<T, s_Outputs, s_Shape.size()> forward_pass(
        const static_matrix<T, s_Shape.size(), s_Inputs>& input_data) const {
      return m_Data.forward_pass(input_data);
    }

  };

  //general specialization
  template<typename T, size_t Inputs, Layer_Shape Current_Shape, Layer_Shape... Shapes>
  struct layer_unroll<T, Inputs, Current_Shape, Shapes...>
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

    layer_unroll<T, Current_Shape.size(), Shapes...> m_Next; //another layer_unroll member for the rest

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

    void store(std::ofstream& out) const {
      m_Data.store(out);
      m_Next.store(out);
    }

    void load(std::ifstream& in) {
      m_Data.load(in);
      m_Next.load(in);
    }

    template<size_t Outputs>
    static_matrix<T, 1, Outputs> forward_pass(
        const static_matrix<T, s_Shape.size(), s_Inputs>& input_data) const {
      return m_Next.forward_pass<Outputs>(m_Data.forward_pass(input_data));
    }
  };

  //--------------------------------------------------------------------------------------//

  //template<typename T, Layer_Structure... Structures>
  //  requires ((sizeof...(Structures) >= 3) and std::is_floating_point<T>::value)
  //class static_neural_net {
  //public:
  //  static constexpr size_t s_Layers{ sizeof...(Structures) };
  //  static constexpr std::array<Layer_Structure, s_Layers> s_Structures{ Structures... };

  //  std::tuple<layer<T, s_Structures>> m_Layers{};
  //};

  template<typename T, Layer_Shape... Shapes>
    requires ((sizeof...(Shapes) >= 3) and std::is_floating_point<T>::value)
  class static_neural_net {
  private:
    static constexpr size_t s_Layers{ sizeof...(Shapes) };
    static constexpr std::array<Layer_Shape, s_Layers> s_Shapes{ Shapes... };
    static constexpr Layer_Shape s_Out_Shape{ s_Shapes[s_Layers - 1] };

    layer_unroll<T, 1, Shapes...> m_Layers{};

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
      std::cout << "##########################################################################\n";
      print_layers();
      std::cout << "Net has " << parameters() << " parameters\n";
      std::cout << "--------------------------------------------------------------------------\n";
    }

    void store(const std::string& filename) const {
      std::ofstream out(filename);
      if (!out.is_open()) {
        std::cout << "Cound not create file: " + filename + "\n";
        exit(EXIT_FAILURE);
      }
      //store net shapes
      for (const auto& shape : s_Shapes) {
        out << shape.M << " " << shape.N << " ";
      }
      out << "\n\n";
      m_Layers.store(out); //Store layers recursively
      out.close();
    }

    //Overrides current net with one read from "filename"
    //Shapes of both nets must be the same
    void load(const std::string& filename) {
      std::ifstream in(filename);
      if (!in.is_open()) {
        std::cout << "Cound not open file: " + filename + "\n";
        exit(EXIT_FAILURE);
      }
      size_t m{}, n{}; //shape
      for (const auto& e : s_Shapes) {
        in >> m >> n;
        if (m != e.M or n != e.N){
          std::cout << "Cannot load this net here. Shapes must match\n";
          exit(EXIT_FAILURE);
        }
      }
      m_Layers.load(in); //load layers recursively
      in.close();
    }

    template<size_t M, size_t N>
      requires (Layer_Shape{ M, N } == s_Shapes[0])
    static_matrix<T, 1, s_Out_Shape.size()> forward_pass(
        static_matrix<T, M, N> const& input_data) const {

      static_matrix<T, s_Shapes[0].size(), 1> Temp{}; //format input
      for (size_t j = 0; j < M; ++j) {
        for (size_t i = 0; i < N; ++i) {
          Temp(j * N + i, 0) = input_data(j, i);
        }
      }
      return m_Layers.forward_pass<s_Out_Shape.size()>(Temp);
    }
  };

  //--------------------------------------------------------------------------------------//
  //               Functionality
  //--------------------------------------------------------------------------------------//



  template<typename T, Layer_Structure Structure>
  std::pair<layer<T, Structure>, layer<T, Structure>> in_place_layer_x_crossover(
      layer<T, Structure>& layer1,
      layer<T, Structure>& layer2) {
    
    in_place_x_crossover(layer1.get_weigh_input_mat(),
                         layer2.get_weigh_input_mat());

    in_place_x_crossover(layer1.get_out_expand_mat(),
                         layer2.get_out_expand_mat());
    
    in_place_x_crossover(layer1.get_out_offset_mat(),
                         layer2.get_out_offset_mat());

    return std::make_pair(layer1, layer2);
  }

  //neural net x_crossover

  template<typename T, Layer_Shape... Shapes>
  std::pair< static_neural_net<T, Shapes...>, static_neural_net<T, Shapes...>> x_crossover(
      static_neural_net<T, Shapes...> const& net1, 
      static_neural_net<T, Shapes...> const& net2) {

    static_neural_net<T, Shapes...> ret_net1{ net1 };
    static_neural_net<T, Shapes...> ret_net2{ net2 };
    //Recursivelly crossover layers
    in_place_net_x_crossover<0>(ret_net1, ret_net2);
    return std::make_pair(ret_net1, ret_net2);
  }

  template<size_t I, typename T, Layer_Shape... Shapes>
    requires (I != sizeof...(Shapes) - 1)
  void in_place_net_x_crossover(
      static_neural_net<T, Shapes...>& temp_net1,
      static_neural_net<T, Shapes...>& temp_net2) {
    in_place_layer_x_crossover(temp_net1.layer<I>(), temp_net2.layer<I>());
    in_place_net_x_crossover<I + 1>(temp_net1, temp_net2);
  }

  template<size_t I, typename T, Layer_Shape... Shapes>
    requires (I == sizeof...(Shapes) - 1)
  void in_place_net_x_crossover(
      static_neural_net<T, Shapes...>& temp_net1,
      static_neural_net<T, Shapes...>& temp_net2) {
    in_place_layer_x_crossover(temp_net1.layer<I>(), temp_net2.layer<I>());
  }


}
