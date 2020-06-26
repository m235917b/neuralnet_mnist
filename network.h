#include "matrix.h"

//for comments/documentary see network.cpp / documentary.txt

class Network
{
public:
    Network();
    Network(const matrix<int>& layers, int flag, double dist);
    Network(const Network& owner);
    Network(Network&& owner) noexcept;
    Network& operator=(const Network& that);
    Network& operator=(Network&& that);
    friend std::ostream& operator<<(std::ostream& out, const Network& n);
    Network clone();
    ~Network();

    void set(int position, const matrix<double>& mat);
    const matrix<double>& get(int position) const;
    void set_bias(int position, const matrix<double>& mat);
    const matrix<double>& get_bias(int position) const;

    matrix<double> forward(matrix<double> mat);
    matrix<double> forward_batch(const matrix<double>& input);
    matrix<double> forward_to(int layer, matrix<double> mat);
    void set_zero();

    void mutate(int range);
    double train_grad(double learning_rate, double change_rate, const matrix<double>& input, const matrix<double>& output);
    double train_grad_double(double learning_rate, double change_rate, const matrix<double>& input, const matrix<double>& output);
    double train_rand(double learning_rate, const matrix<double>& input, const matrix<double>& output);

    void sgd(double learning_rate, double normalization_factor, int mini_batch_size,
             int epochs, const matrix<double>& input, const matrix<double>& output,
             const matrix<double>& test_data, const matrix<double>& test_data_out);
    void gradient_descent(double learning_rate, double normalization_factor, int training_set_size,
                          const matrix<double>& input, const matrix<double>& output);
    void backpropagate(const matrix<double>& input, const matrix<double>& output, Network& gradients);

private:
    int depth;
    matrix<int> layers;
    matrix<double> *net_ptr;
    matrix<double> *bias_ptr;

    matrix<int> shuffle(int range, int size);
    matrix<double> activation_prime(const matrix<double>& m);
    matrix<double> activation(const matrix<double>& m);
    double cost_function(const matrix<double>& act_out, const matrix<double>& des_out);

    double quadratic(const matrix<double>& act_out, const matrix<double>& des_out);
    double cross_entropy(const matrix<double>& act_out, const matrix<double>& des_out);
};

inline std::ostream& operator<<(std::ostream& out, const Network& n)
{
    for(int i = 0; i < n.depth; ++i)
    {
        out << n.net_ptr[i];
    }

    return out;
}
