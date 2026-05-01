#pragma once
#include "linear_layer.hpp"

template <typename T>
class NeuralNetwork{

	private:
		LinearLayer<T> l1;
		LinearLayer<T> l2;

		MathVec<T> cache1;

	public:
		NeuralNetwork(){}
    		NeuralNetwork(size_t in_dim, size_t hidden_dim, size_t out_dim)
        		: l1(in_dim, hidden_dim, true),   //ReLU activation true
			  l2(hidden_dim, out_dim, false) {} //sigmoid output

	//-RANDOM INITIALIZATION
	void random_init(T scale = T(1)) {
   		l1.random_init(scale);
    		l2.random_init(scale);
	}

	//-FORWARD PASS
	MathVec<T> forward(const MathVec<T> &x){

		cache1 = l1.linear(x);

		MathVec<T> out = l2.linear(cache1);

		return out;
	}

	//-BACKWARD PASS
	MathVec<T> backward(const MathVec<T> &grad_out){

		MathVec<T> g2 = l2.backward(grad_out);
		MathVec<T> g1 = l1.backward(g2);

		return g1;
	}

	//-UPDATE STEP
	void step(T lr){

		l1.step(lr);
		l2.step(lr);
	}
	//-ZERO OUT GRAD
	void zero_grad() {
    		l1.zero_grad();
    		l2.zero_grad();
	}

	//-PRINT PARAMETERS
	void print_params(){
		std::cout << "Layer 1:\n";
		l1.print_params();
		std::cout << "Layer 2:\n";
		l2.print_params();
	}

};
