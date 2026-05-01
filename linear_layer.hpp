#pragma once
#include "mat_lib.hpp"
#include <iostream>

template <typename T>
class LinearLayer{

	private:
		MathMat<T> weights;
		MathVec<T> bias;
		MathVec<T> last_input;
		MathVec<T> last_z;
		MathMat<T> grad_W;
		MathVec<T> grad_b;
		bool relu_on;

	public:
		LinearLayer(){}
    		LinearLayer(size_t in_dim, size_t out_dim, bool relu=true)
        		: weights(out_dim, in_dim),
			  bias(out_dim),
			  grad_W(out_dim, in_dim),
			  grad_b(out_dim),
			  relu_on(relu) {}

	//-RANDOM INITIALIZATION OF WEIGHTS AND BIAS
	void random_init(T scale= T(1)) {
   		weights.random_init(scale);
    		bias.random_init(scale);
	}

	//-FORWARD PASS
	MathVec<T> linear(const MathVec<T> &x){
		last_input = x;
		last_z = weights * x + bias;
		return relu_on ? ReLU(last_z):sigmoid(last_z);
	}

	MathVec<T> ReLU(const MathVec<T> &x) const{
		MathVec<T> res(x.size());
		for(size_t i = 0; i < x.size(); ++i){
			res[i] = (x[i] > 0) ? x[i] : 0;
		}
		return res;
	}


        MathVec<T> sigmoid(const MathVec<T> &x) const{
                MathVec<T> res(x.size());
                for(size_t i = 0; i < x.size(); ++i){
                        res[i] = 1.0/(1.0+std::exp(-x[i]));
                }
                return res;
        }




	//-BACKWARD PASS
	MathVec<T> backward(const MathVec<T> &grad_out){
		//grad_out.print();
		
		MathVec<T> grad = relu_on 
					? ReLU_mask(last_z)%grad_out
					: grad_out%sigmoid_derivative(last_z);
		grad_b += grad;
		// gradient outer product: grad_out ^ last_input
		grad_W += grad^last_input;
		// W**T*grad_out			
		return weights.trans() * grad;
	}


        MathVec<T> ReLU_mask(const MathVec<T> &z) const {
                MathVec<T> mask(z.size());
                for(size_t i = 0; i < z.size(); ++i){
                        mask[i] = (z[i] > 0) ? 1 : 0;
                }
                return mask;
        }

        MathVec<T> sigmoid_derivative(const MathVec<T> &z) const {
                MathVec<T> sig = sigmoid(z);
    		MathVec<T> one_minus_sig(sig.size());

    		for (size_t i = 0; i < sig.size(); ++i)
        		one_minus_sig[i] = 1 - sig[i];

                return sig%one_minus_sig;
        }





	//-STEP UPDATE
	void step(T lr){

		for(size_t i = 0; i < weights.rows(); ++i){
			for(size_t j = 0; j < weights.cols(); ++j){
				weights(i,j) -= lr * grad_W(i,j);
			}
		}

		for(size_t i = 0; i < bias.size(); ++i){
			bias[i] -= lr * grad_b[i];
		}
	}

	//-ZEROS OUT GRADIENTS
	void zero_grad() {
    		// zero weight gradients
    		for (size_t i = 0; i < grad_W.rows(); ++i) {
        		for (size_t j = 0; j < grad_W.cols(); ++j) {
            			grad_W(i,j) = 0.0;
        		}
    		}
    		// zero bias gradients
    		for (size_t i = 0; i < grad_b.size(); ++i) {
        		grad_b[i] = 0.0;
    		}
	}


	//-PRINT OUT WEIGHTS AND BIAS-PRIVATE VARIABLES
	void print_params() const {

    		for (size_t i = 0; i < weights.rows(); ++i) {
        		for (size_t j = 0; j < weights.cols(); ++j) {
            			std::cout << weights(i,j) << " ";
        		}
        		std::cout << "\n";
    		}

    		std::cout << "----\n";

    		for (size_t i = 0; i < bias.size(); ++i) {
        		std::cout << bias[i] << "\n";
    		}
	}

};//end LinearLayer
