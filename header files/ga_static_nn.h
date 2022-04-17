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
    size_t Inputs;
    size_t Outputs;
    
    constexpr bool operator==(const Layer_Structure&) const = default;
  };

  //--------------------------------------------------------------------------------------//

  template<typename T, Layer_Structure Structure>
    requires std::is_floating_point<T>::value 
  class layer {
  private:
    static constexpr size_t s_Inputs = Structure.Inputs;
    static constexpr size_t s_Outputs = Structure.Outputs;

  public:
    using layer_weights_shape = static_matrix<T, s_Outputs, s_Inputs>;
    using output_vector_shape = static_matrix<T, s_Outputs, 1>;
    using input_vector_shape = static_matrix<T, s_Inputs, 1>;

  private:
    layer_weights_shape m_layer_weights_mat{};
    output_vector_shape m_offset_vector{};


  public:
    //each neuron produces a column vector that splits into the input of the neurons in the next layer
    [[nodiscard]]constexpr 
    output_vector_shape forward_pass(
        input_vector_shape const& Input) const {

      return matrix_mul(m_layer_weights_mat, Input) + m_offset_vector; 
    }

    template<auto fn, auto... args>
    constexpr void init() {
      m_layer_weights_mat.fill<fn>(args...);
      m_offset_vector.fill<fn>(args...);
    }

    void store(std::ofstream& out) const {
      m_layer_weights_mat.store(out);
      m_offset_vector.store(out);
    }

    void load(std::ifstream& in) {
      m_layer_weights_mat.load(in);
      m_offset_vector.load(in);
    }

    inline constexpr 
    const layer_weights_shape& get_layer_weights_mat() const {
      return m_layer_weights_mat;
    }

    inline constexpr 
    layer_weights_shape& get_layer_weights_mat() {
      return m_layer_weights_mat;
    }
    
    inline constexpr 
    const output_vector_shape& get_offset_vector() const {
      return m_offset_vector;
    }

    inline constexpr 
    output_vector_shape& get_offset_vector() {
      return m_offset_vector;
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

    os << layer.get_layer_weights_mat() <<
          layer.get_offset_vector() << "\n";
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
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ Current_Shape.size() };

    layer <T, Layer_Structure{ Inputs, Current_Shape.size() }> m_Data{}; // one data member for this layer

    //------ MEMBER FUNCTIONS -----//
    template<size_t Idx>
    constexpr auto& get() {
      if constexpr (Idx == 0){ // if its 0, return this layer
        return m_Data;
      }
      else {
        static_assert(false && Idx, "requested layer index exceeds number of layers");
      }
    }

    template<size_t Idx>
    constexpr auto const& const_get() const {
      if constexpr (Idx == 0) { // if its 0, return this layer
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

    template<size_t Net_Outputs>
      requires (Net_Outputs == s_Outputs)
    static_matrix<T, s_Outputs, 1> forward_pass(
        static_matrix<T, s_Inputs, 1> const& input_data) const {
      return m_Data.forward_pass(input_data);
    }

  };

  //general specialization
  template<typename T, size_t Inputs, Layer_Shape Current_Shape, Layer_Shape... Shapes>
  struct layer_unroll<T, Inputs, Current_Shape, Shapes...>
  {
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ Current_Shape.size() };

    layer <T, Layer_Structure{ s_Inputs,s_Outputs }> m_Data{}; // one data member for this layer
    layer_unroll<T, Current_Shape.size(), Shapes...> m_Next; //another layer_unroll member for the rest

    //------ MEMBER FUNCTIONS -----//
    //getter so that we can actually get a specific layer by index
    template<size_t Idx>
    constexpr auto& get(){
      if constexpr (Idx == 0) {// if its 0, return this layer
        return m_Data;
      }
      else {
        return m_Next.template get<Idx - 1>(); //if the index is not 0, ask the next layer with an updated index
      }
    }

    template<size_t Idx>
    constexpr auto const& const_get() const {
      if constexpr (Idx == 0) {// if its 0, return this layer
        return m_Data;
      }
      else {
        return m_Next.template const_get<Idx - 1>(); //if the index is not 0, ask the next layer with an updated index
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

    template<size_t Net_Outputs>
    static_matrix<T, Net_Outputs, 1> forward_pass(
        static_matrix<T, s_Inputs, 1> const& input_data) const {
      return m_Next.forward_pass<Net_Outputs>(m_Data.forward_pass(input_data));
    }
  };

  //--------------------------------------------------------------------------------------//

  template<typename T, Layer_Shape... Shapes>
    requires ((sizeof...(Shapes) >= 3) and std::is_floating_point<T>::value)
  class static_neural_net {
  private:
    static constexpr size_t s_Layers{ sizeof...(Shapes) };
    static constexpr std::array<Layer_Shape, s_Layers> s_Shapes{ Shapes... };
    static constexpr Layer_Shape s_Input_Shape{ s_Shapes[0] };
    static constexpr Layer_Shape s_Output_Shape{ s_Shapes[s_Layers - 1] };
    static constexpr size_t s_Out_M = s_Output_Shape.M;
    static constexpr size_t s_Out_N = s_Output_Shape.N;

    layer_unroll<T, s_Input_Shape.size(), Shapes...> m_Layers{};
  public:

    constexpr size_t parameter_count() const {
      size_t p = (s_Input_Shape.size() + 1) * s_Input_Shape.size();
      for (size_t i = 1; i < s_Layers; ++i) {
        p += (s_Shapes[i - 1].size() + 1) * s_Shapes[i].size();
      }
      return p;
    }
   
    template<std::size_t Idx>
    auto& layer() {
      return m_Layers.template get<Idx>();
    } //you arent a real template programmer if you dont use .template  /s

    template<std::size_t Idx>
    auto const& const_layer() const {
      return m_Layers.template const_get<Idx>();
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
      std::cout << "Net has " << parameter_count() << " parameters\n";
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
      requires (Layer_Shape{ M, N } == s_Input_Shape)
    static_matrix<T, s_Out_M, s_Out_N> forward_pass(
        static_matrix<T, M, N> const& input_data) const {
      const auto Temp = cast_to_shape<s_Input_Shape.size(), 1>(input_data); //format input
      const static_matrix<T, s_Output_Shape.size(), 1> inferred = m_Layers.forward_pass<s_Output_Shape.size()>(Temp);
      return cast_to_shape<s_Out_M, s_Out_N>(inferred); //format output
    }
  };

  //--------------------------------------------------------------------------------------//
  //               Utility
  //--------------------------------------------------------------------------------------//


  //--------------------------------------------------------------------------------------//
  //neural net x_crossover

  template<typename T, Layer_Shape... Shapes>
  std::pair< static_neural_net<T, Shapes...>, static_neural_net<T, Shapes...>> x_crossover(
      static_neural_net<T, Shapes...> const& net1,
      static_neural_net<T, Shapes...> const& net2) {

    static_neural_net<T, Shapes...> ret_net1{ net1 };
    static_neural_net<T, Shapes...> ret_net2{ net2 };
    //Recursivelly crossover layers, starting with layer 0
    in_place_net_x_crossover<0>(ret_net1, ret_net2);
    return std::make_pair(ret_net1, ret_net2);
  }

  template<typename T, Layer_Structure Structure>
  std::pair<layer<T, Structure>, layer<T, Structure>> in_place_layer_x_crossover(
      layer<T, Structure>& layer1,
      layer<T, Structure>& layer2) {
    
    in_place_x_crossover(layer1.get_layer_weights_mat(),
                         layer2.get_layer_weights_mat());
    
    in_place_x_crossover(layer1.get_offset_vector(),
                         layer2.get_offset_vector());

    return std::make_pair(layer1, layer2);
  }

  template<size_t I, typename T, Layer_Shape... Shapes>
    requires (I < sizeof...(Shapes))
  void in_place_net_x_crossover(
      static_neural_net<T, Shapes...>& net1,
      static_neural_net<T, Shapes...>& net2) {
    if constexpr (I == sizeof...(Shapes) - 1) {
      in_place_layer_x_crossover(net1.layer<I>(), net2.layer<I>());
    }
    else {
      in_place_layer_x_crossover(net1.layer<I>(), net2.layer<I>());
      in_place_net_x_crossover<I + 1>(net1, net2);
    }
  }
  //......................................................................................//


  //--------------------------------------------------------------------------------------//
  // population variability

  //Returns matrix of distances between elements
  //Diagonal is set to 0
  template<size_t N, typename T, Layer_Shape... Shapes>
    requires (N > 1)
  [[nodiscard]]
  static_matrix<double, N, N> population_variability(
      std::array<static_neural_net<T, Shapes...>, N> const& net_arr) {

    static_matrix<double, N, N> L11_distance_matrix{};
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = j + 1; i < N; ++i) {
        double distance = L11_net_distance<0>(net_arr[j], net_arr[i]);
        L11_distance_matrix(j, i) = distance;
        L11_distance_matrix(i, j) = distance;
      }
    }
    return L11_distance_matrix;
  }

  template<typename T, Layer_Structure Structure>
  [[nodiscard]]
  double L11_layer_distance(
      layer<T, Structure> const& layer1,
      layer<T, Structure> const& layer2) {
    double distance = normaliced_L1_distance(layer1.get_layer_weights_mat(), layer2.get_layer_weights_mat())
                    + normaliced_L1_distance(layer1.get_offset_vector(), layer2.get_offset_vector());
    return distance;
  }

  template<size_t I, typename T, Layer_Shape... Shapes>
    requires (I < sizeof...(Shapes))
  [[nodiscard]]
  double L11_net_distance(
    static_neural_net<T, Shapes...> const& net1,
    static_neural_net<T, Shapes...> const& net2) {
    if constexpr (I == sizeof...(Shapes) - 1) {
      return L11_layer_distance(net1.const_layer<I>(), net2.const_layer<I>());
    }
    else {
      return L11_layer_distance(net1.const_layer<I>(), net2.const_layer<I>()) + L11_net_distance<I + 1>(net1, net2);
    }
  }

  //......................................................................................//


}
