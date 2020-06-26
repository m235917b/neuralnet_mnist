#include <iostream>
#include <math.h>
#include <vector>
#include "network.h"

/*
This class creates simple feed forward neural Networks. They are represented by a set of matrices from the "matrix" class that is needed additionally.
Each matrix represents one layer (only if there are no hidden layers, both input and output layer are represented by one matrix, since the input layer has no weights).
The way it works is, that the input data is given in a (vertical) vector ([number of input neurons] x 1 - matrix) which is then multiplied by the first matrix. The entries of the resulting vector
are then passed through a non-linear activation function (because otherwise the Network would just be able to solve trivial linear problems, since mtrices represent only linear functions) and the results
are saved in the original place of this vector. Then the resulting vector is multiplied with the second matrix and so on... The resulting vector of the last multiplication represents the output values.
This is just a mathematical representation, that the inputs of every neuron are weighted and then summed together to then be passed through an activation function, which models the function of a neuron.
Therefore one matrix represents one layer, one row of a matrix represents the weights/inputs of one neuron in that layer and one column represents the weights/inputs of every neuron from just one former neuron.

This class offers many different learning algorithms, including random ones for the implementation of genetic algorithms
*/

//constructors

Network::Network() : depth(0), net_ptr(nullptr), bias_ptr(nullptr)
{

}

Network::Network(const matrix<int>& layers, int flag, double dist) : depth(layers.width() - 1), layers(layers),
    net_ptr(new matrix<double>[depth]), bias_ptr(new matrix<double>[depth])
{
    switch(flag)
    {
    case 0:
        for(int i = depth - 1; i >= 0; --i)
        {
            net_ptr[i] = matrix<double>::null(layers(0, i + 1), layers(0, i));
            bias_ptr[i] = matrix<double>::null(layers(0, i + 1), 1);
        }
        break;

    case 1:
        for(int i = depth - 1; i >= 0; --i)
        {
            net_ptr[i] = matrix<double>::random(layers(0, i + 1), layers(0, i), dist);
            bias_ptr[i] = matrix<double>::random(layers(0, i + 1), 1, 1);
        }
        break;
    }
}

//copy constructor; needed, because otherwise copies of an instance of "Network" would only contain a copy of the pointer net_ptr an thus pointing to the same adress as the original object
Network::Network(const Network& owner) : depth(owner.depth), layers(owner.layers),
    net_ptr(new matrix<double>[depth]), bias_ptr(new matrix<double>[depth])
{
    //copy entrys from original net_ptr and bias_ptr to the new allocated memory space for this copy
    std::copy(owner.net_ptr, owner.net_ptr + owner.depth, net_ptr);
    std::copy(owner.bias_ptr, owner.bias_ptr + owner.depth, bias_ptr);
}

//move constructor
Network::Network(Network&& owner) noexcept : Network()
{
    std::swap(depth, owner.depth);
    std::swap(layers, owner.layers);
    std::swap(net_ptr, owner.net_ptr);
    std::swap(bias_ptr, owner.bias_ptr);
}

//copy assignment operator; same as copy constructor but for copies created by assignments (e.g. Network1 = Network2)
Network& Network::operator=(const Network& that)
{
    //if this copy is not the original
    if (this != &that)
    {
        delete[] net_ptr;
        delete[] bias_ptr;

        depth = that.depth;
        layers = that.layers;

        net_ptr = new matrix<double>[depth];
        bias_ptr = new matrix<double>[depth];

        std::copy(that.net_ptr, that.net_ptr + that.depth, net_ptr);
        std::copy(that.bias_ptr, that.bias_ptr + that.depth, bias_ptr);
    }

    return *this;
}

//move operator
Network& Network::operator=(Network&& that)
{
    std::swap(depth, that.depth);
    std::swap(layers, that.layers);
    std::swap(net_ptr, that.net_ptr);
    std::swap(bias_ptr, that.bias_ptr);

    return *this;
}

Network Network::clone()
{
    return *this;
}

//destructor
Network::~Network()
{
    //free memory space from Network content
    delete[] net_ptr;
    delete[] bias_ptr;
}

//set one layer by passing a representing matrix mat; position is the position of the layer (0 is the first layer)
void Network::set(int position, const matrix<double> &mat)
{
    //check if position is valid
    if(position >= depth || position < 0)
    {
        return;
    }

    //assign matrix to correct layer and save it into the Network
    net_ptr[position] = mat;
}

//returns the representing matrix of layer number "position"
const matrix<double>& Network::get(int position) const
{
    //check if position is valid
    if(position >= depth || position < 0)
    {
        return net_ptr[0];
    }

    //return matrix
    return net_ptr[position];
}

//set one bias for a layer by passing a representing matrix mat; position is the position of the layer (0 is the firs layer)
void Network::set_bias(int position, const matrix<double>& mat)
{
    //check if position is valid
    if(position > depth || position < 0)
    {
        return;
    }

    //assign matrix to correct layer and save it into the Network
    bias_ptr[position] = mat;
}

//returns the representing bias matrix of layer number "position"
const matrix<double>& Network::get_bias(int position) const
{
    //check if position is valid
    if(position > depth || position < 0)
    {
        return bias_ptr[0];
    }

    //return matrix
    return bias_ptr[position];
}

void Network::set_zero()
{
    for(int i = depth - 1; i >= 0; --i)
    {
        net_ptr[i].set_zero();
        bias_ptr[i].set_zero();
    }
}

//takes an input vector (vector of input values) and passes it through the Network. Returns a vertical vector with the calculated output values.
matrix<double> Network::forward(matrix<double> mat)
{
    //check if mat has correct dimensions (height = number of input neurons; width = 1)
    if(mat.height() != net_ptr[0].width() || mat.width() != 1)
    {
        return mat;
    }

    //multiply mat by i-th layer-matrix, add biases and calculate the activation of that
    for(int i = 0; i < depth; ++i)
    {
        mat = activation((net_ptr[i] * mat) + bias_ptr[i]);
    }

    //return result of the last multiplication
    return mat;
}

//same as forward() but with a matrix of input vectors
matrix<double> Network::forward_batch(const matrix<double>& input)
{
    matrix<double> out = matrix<double>(layers(0, depth), input.width());

    for(int i = input.width() - 1; i >= 0; --i)
    {
        out.set_column(i, forward(input.column(i)));
    }

    return out;
}

//calculates the output of a certain layer
matrix<double> Network::forward_to(int layer, matrix<double> mat)
{
    //check if mat has correct dimensions (height = number of input neurons; width = 1)
    if(mat.height() != net_ptr[0].width() || mat.width() != 1)
    {
        return mat;
    }

    //multiply mat by i-th layer-matrix, add biases and calculate the activation if that
    for(int i = 0; i < layer; ++i)
    {
        mat = activation((net_ptr[i] * mat) + bias_ptr[i]);
    }

    //return result of the last multiplication
    return mat;
}

//changes weights of the Network randomly; "range" is the order by how much the change should be (range = 0 changes weights by a value between -1 and 1)
void Network::mutate(int range)
{
    //variables to save height and width of processed matrix
    int h, w;

    //run through every layer
    for(int i = depth - 1; i >= 0; --i)
    {
        //get height of i-th layer-matrix
        h = net_ptr[i].height();
        //get width of i-th layer-matrix
        w = net_ptr[i].width();

        net_ptr[i] = net_ptr[i] + (matrix<double>::random(h, w, 1) * pow(10, range));
        bias_ptr[i] = bias_ptr[i] + (matrix<double>::random(h, 1, 1) * pow(10, range));
    }
}

/*
train Network by approximating the error gradient (from error function f: weight-space -> error-space) via the difference quotients and then change Network in the oppositew direction to minimalize error;
this changes the Network just a little step into the right direction, so this method must be called repeatedly for the Network to converge against an error of 0;
this is why it returns the new error value, to determine how often it must be called to fulfill a certain specification;
the method should be called repeatedly until the returned error falls below a certain desirable value
learning_rate specifies how fast the Network changes but also how inaccurate (the faster the more inaccurate); must be a positive vale
change_rate specifies how fine the change of each difference quotient should be (has the same effects as learning_rate); must be a positive value
"input" is a matrix that contains the whole training set of inputs, where each column represents a single input vector; must have same height as number of input neurons
"output" contains the associated desired output vectors, where each column of "output" represents a output vector; must have same height as number of output neurons;
converges slower but safer than trainGraddouble
*/
double Network::train_grad(double learning_rate, double change_rate, const matrix<double>& input, const matrix<double>& output)
{
    //variables for saving the matrix dimensions
    int height, width, width_io;

    //copy this Network
    Network copy = *this;

    //save width of "input"/"output" to calculate the outputs
    width_io = input.width();
    //create a matrix with same dimensions as "output" to save those outputs
    matrix<double> act_out = matrix<double>(output.height(), output.width());

    //variable to save the squared error of the Network with one changed weight
    double changed_error1 = 0;
    double changed_error2 = 0;
    //variable to save the difference quotient
    double diff_quot;

    //run through each matrix in this Network
    for(int i = depth - 1; i >= 0; --i)
    {
        //save dimensions of this matrix
        height = net_ptr[i].height();
        width = net_ptr[i].width();

        //run through each row and column (each entry) of the matrix
        for(int x = height - 1; x >= 0; --x)
        {
            for(int y = width - 1; y >= 0; --y)
            {
                //add change_rate to the weight that is processed in this loop run
                net_ptr[i](x, y) += change_rate;

                //calculate each output vector by forwarding the input vector and save result in act_out
                for(int y = width_io - 1; y >= 0; --y)
                {
                    act_out.set_column(y, forward(input.column(y)));
                }

                //calculate the squared error with outputs by changed weight
                changed_error1 = cost_function(act_out, output);

                //subtract change_rate from the weight twice (so it is subtracted one time from thr original value) that is processed in this loop run
                net_ptr[i](x, y) -= 2 * change_rate;

                //calculate each output vector by forwarding the input vector and save result in act_out
                for(int y = width_io - 1; y >= 0; --y)
                {
                    act_out.set_column(y, forward(input.column(y)));
                }

                //calculate the squared error with outputs by changed weight
                changed_error2 = cost_function(act_out, output);

                //undo the change of the weight, to calculate next difference quotient at the same point (to not change the point of differentiation while differentiating)
                net_ptr[i](x, y) += change_rate;

                //calculate difference quotient (not the differential quotient)
                diff_quot = (changed_error1 - changed_error2) / (2.0 * change_rate);

                //save the negative of diff_quot times learning_rate in the according weight-entry of the copy of this Network;
                //the resulting matrices in "copy" will be added to the matrices of this Network to change the entire Network into the opposite direction of the error gradient
                copy.net_ptr[i](x, y) = -1.0 * diff_quot * learning_rate;
            }

            //do the same for the bias

            bias_ptr[i](x, 0) += change_rate;

            for(int y = width_io - 1; y >= 0; --y)
            {
                act_out.set_column(y, forward(input.column(y)));
            }

            changed_error1 = cost_function(act_out, output);

            bias_ptr[i](x, 0) -= 2 * change_rate;

            for(int y = width_io - 1; y >= 0; --y)
            {
                act_out.set_column(y, forward(input.column(y)));
            }

            changed_error2 = cost_function(act_out, output);

            bias_ptr[i](x, 0) += change_rate;

            diff_quot = (changed_error1 - changed_error2) / (2.0 * change_rate);

            copy.bias_ptr[i](x, 0) = -1.0 * diff_quot * learning_rate;
        }
    }

    //now the error gradient is calculated and the negative value times learning_rate of its components is saved in the matrices of "copy";
    //each value for each weight in the according position in those matrices, so now we only need to add those matrices to the matrices of this Network
    //to change it into the direction of steepest error minimalization and thus getting better results with the changed Network

    //add the matrices of "copy" to the matrices of this Network to train/improove this Network
    for(int i = depth - 1; i >= 0; --i)
    {
        net_ptr[i] += copy.net_ptr[i];
        bias_ptr[i] += copy.bias_ptr[i];
    }

    //recalculate the outputs with changed Network to get new squared error
    for(int y = width_io - 1; y >= 0; --y)
    {
        act_out.set_column(y, forward(input.column(y)));
    }

    //return the final squared error of the changed/improoved Network
    return cost_function(act_out, output);
}

//Same as trainGrad, but this time every weight is changed when its difference quotient is calculated, bevore the difference quotient for the next weight is calculated; better with activation function
double Network::train_grad_double(double learning_rate, double change_rate, const matrix<double>& input, const matrix<double>& output)
{
    //variables for saving the matrix dimensions
    int height, width, width_io;

    //save width of "input"/"output" to calculate the outputs
    width_io = input.width();
    //create a matrix with same dimensions as "output" to save those outputs
    matrix<double> act_out = matrix<double>(output.height(), output.width());

    //variable to save the squared error of the Network with one changed weight
    double changed_error1 = 0;
    double changed_error2 = 0;
    //variable to save the difference quotient
    double diff_quot;

    //run through each matrix in this Network
    for(int i = depth - 1; i >= 0; --i)
    {
        //save dimensions of this matrix
        height = net_ptr[i].height();
        width = net_ptr[i].width();

        //run through each row and column (each entry) of the matrix
        for(int x = height - 1; x >= 0; --x)
        {
            for(int y = width - 1; y >= 0; --y)
            {
                //add change_rate to the weight that is processed in this loop run
                net_ptr[i](x, y) += change_rate;

                //calculate each output vector by forwarding the input vector and save result in act_out
                for(int y = width_io - 1; y >= 0; --y)
                {
                    act_out.set_column(y, forward(input.column(y)));
                }

                //calculate the squared error with outputs by changed weight
                changed_error1 = cost_function(act_out, output);

                //subtract change_rate from the weight twice (so it is subtracted one time from thr original value) that is processed in this loop run
                net_ptr[i](x, y) -= 2 * change_rate;

                for(int y = width_io - 1; y >= 0; --y)
                {
                    act_out.set_column(y, forward(input.column(y)));
                }

                //calculate the squared error with outputs by changed weight
                changed_error2 = cost_function(act_out, output);

                //undo the change of the weight, to recalculating weight from original value
                net_ptr[i](x, y) += change_rate;

                //calculate difference quotient (not the differential quotient)
                diff_quot = (changed_error1 - changed_error2) / (2.0 * change_rate);

                //change weight into the opposite direction of the error gradient times learning_rate to converge against a lower
                //error value an thus getting better results with the changed Network
                net_ptr[i](x, y) -= diff_quot * learning_rate;
            }

            //do the same for the bias

            bias_ptr[i](x, 0) += change_rate;

            for(int y = width_io - 1; y >= 0; --y)
            {
                act_out.set_column(y, forward(input.column(y)));
            }

            changed_error1 = cost_function(act_out, output);

            bias_ptr[i](x, 0) -= 2 * change_rate;

            for(int y = width_io - 1; y >= 0; --y)
            {
                act_out.set_column(y, forward(input.column(y)));
            }

            changed_error2 = cost_function(act_out, output);

            bias_ptr[i](x, 0) += change_rate;

            diff_quot = (changed_error1 - changed_error2) / (2.0 * change_rate);

            bias_ptr[i](x, 0) -= diff_quot * learning_rate;
        }
    }

    //recalculate the outputs with changed Network to get new squared error
    for(int y = width_io - 1; y >= 0; --y)
    {
        act_out.set_column(y, forward(input.column(y)));
    }

    //return the final squared error of the changed/improoved Network
    return cost_function(act_out, output);
}

double Network::train_rand(double mutate_rate, const matrix<double>& input, const matrix<double>& output)
{
    double error = 0, changed_error = 0;

    //copy this Network
    Network copy = *this;

    //calculate the squared error with outputs before mutated network
    error = cost_function(forward_batch(input), output);

    //mutate the network
    mutate(mutate_rate);

    //calculate the squared error with outputs by mutated network
    changed_error = cost_function(forward_batch(input), output);

    if(error > changed_error)
    {
        return changed_error;
    }

    //reverse changes
    for(int i = depth - 1; i >= 0; --i)
    {
        net_ptr[i] = copy.net_ptr[i];
        bias_ptr[i] = copy.bias_ptr[i];
    }

    return error;
}

/*
Train the neural network using mini-batch stochastic gradient descent.
normalization_factor is only needed, if L2 normalization is used in gradient_descent()
NOTE: different learning rates may be needed for different cost functions!
*/
void Network::sgd(double learning_rate, double normalization_factor, int mini_batch_size,
                  int epochs, const matrix<double>& input, const matrix<double>& output,
                  const matrix<double>& test_data, const matrix<double>& test_data_out)
{
    //create vector with random selected indices for mini-batch
    matrix<int> mini_batch_indices(1, input.width());
    //create matrix for mini-batch input
    matrix<double> mini_batch_in = matrix<double>(input.height(), mini_batch_size);
    //create matrix for mini-batch output
    matrix<double> mini_batch_out = matrix<double>(output.height(), mini_batch_size);

    int ctr, batch_ctr;

    //train each training-epoch
    for(int i = epochs - 1; i >= 0; --i)
    {
        //create vector with random selected indices for mini-batch
        mini_batch_indices = shuffle(input.width(), input.width());

        ctr = 0;
        batch_ctr = 0;

        //#pragma omp parallel for
        for(batch_ctr = mini_batch_indices.width() - 1; batch_ctr >= 0; --batch_ctr)
        {
            mini_batch_in.set_column(ctr, input.column(mini_batch_indices[batch_ctr]));
            mini_batch_out.set_column(ctr, output.column(mini_batch_indices[batch_ctr]));

            //create mini-batch
            if(++ctr == mini_batch_size)
            {
                ctr = 0;
                //train network on mini-batch
                gradient_descent(learning_rate, normalization_factor, input.width(), mini_batch_in, mini_batch_out);
            }
        }

        //calculate perormance
        matrix<double> output;
        int correct_sets = 0;
        double max_output = 0;

        for(int j = test_data.width() - 1; j >= 0; --j)
        {
            output = forward(test_data.column(j));

            //get neuron with max output value
            for(int k = output.height() - 1; k >= 0; --k)
            {
                max_output = output(k, 0) > output(max_output, 0) ? k : max_output;
            }

            if(max_output == test_data_out[j])
            {
                ++correct_sets;
            }
        }

        std::cout << "Epoch " << epochs - i << ": " << correct_sets << " of " << test_data.width() << "\n";
    }
}

/*
train network via gradient descent for one set of input and output vectors;
normalization_factor is only needed, if L2 normalization is used (uncomment the according line below)
training_set_size is also just needed for L2 normalization
*/
void Network::gradient_descent(double learning_rate, double normalization_factor, int training_set_size,
                               const matrix<double>& input, const matrix<double>& output)
{
    //create an instance of Network to save the gradients for each wheight
    Network gradients = Network(layers, 0, 0);

    //create matrix to store weighted input for one layer
    matrix<double> wi;
    //create vector for storing weighted inputs for each layer as a matrix
    matrix<double> z[depth];
    //create matrix to store output for one layer
    matrix<double> out;
    //create vector for storing outputs for each layer
    matrix<double> outputs[depth + 1];
    //create vector for storing derivatives of the cost function of z for each layer as a matrix
    matrix<double> delta[depth];

    //run through the batch
    for(int i = input.width() - 1; i >= 0; --i)
    {
        //calculate gradients for each weight for a single input-output pair in the batch and add them to the current sum

        out = input.column(i);
        outputs[0] = out;

        //compute weighted inputs and store them in z
        for(int j = 0; j < depth; ++j)
        {
            wi = (net_ptr[j] * out) + bias_ptr[j];
            z[j] = wi;
            out = activation(wi);
            outputs[j + 1] = out;
        }

        //compute derivative of cost function of z for output layer L and store it in delta
        /*
        We look at the function C(out_j(z_j)) where C is the cost function, out_j is the output of the j-th neuron
        and z_j is the wheighted input for the j-th neuron. According to the chain-rule, its derivative is
        C'(out_j(z_j)) * out_j'(z_j). The derivative of C of out_j is simply out_j - des_out_j, where des_out_j
        is the desired output for the j-th neuron. Since out_j(z_j) = sigma(z_j), the derivative of out_j of z_j
        is simply activation_prime(z_j), where activation_prime is the derivative of the activation activation function.

        This line computes the derivatives for each neuron and stores it in a vertical vector
        */

        //choose from the following lines and uncomment one depending on what cost function you want to use

        //quadratic cost function;
        //delta[0] = (outputs[depth] - output.column(i)) % activation_prime(z[depth - 1]);
        //cross entropy function
        delta[0] = activation(z[depth - 1]) - output.column(i);

        //backpropagate the error through each layer
        /*
        According to the chain rule, the derivate of C of z_jl, where z_jl is the weighted input of the j-th neuron of the l-th layer
        is the sum over k of the derivative of C of z_kl+1 times z_kl+1'(z_jl), where z_kl+1 is the weighted input of the k-th neuron
        in the l+1-th layer. This is because z_jl goes through the j-th neuron in layer l and is then distributed to the weighted inputs
        of the neurons in layer l+1.
        For l=L-1 z_kl+1 becomes z_j from above and thus d:=C'(out_kl+1(z_kl+1)) is simply the k-th entry of the vector calculated above,
        since z_jl is again a function of the original input, and thus z_kl+1(z_il) is the same for every i.

        z_kl+1 per definition is the sum of the weights of neuron kl+1 times the output of the neurons of layer l and thus the sum over
        j of w_kj * sigma(z_jl) where w_kl is the weight for neuron jl of neuron kl+1. Therefore the derivative of z_kl+1 of z_jl is
        w_kj * activation_prime(z_jl).

        If we define d_jl+1 as the derivative of C of z_jl+1 the derivative of C of z_jl becomes the sum over k of
        d_jl+1 * w_kj * activation_prime(z_jl).

        For l=L-1 this is the sum over k of the j-th entry of delta(delta.size()-1) * w_jk * activation_prime(z_jl).

        This propagates through the layers l with d_jl+1 = j-th entry of delta(l).

        If we now calculate the derivative entrywise as a vector for each neuron in layer l this formula generalizes to the form
        (net_ptr[l] * d_l+1) % activation_prime(z_l), where d_l is the derivative vector for layer l+1 and z_l is the vector
        of weighted inputs for every neuron in layer l and % is the hadamard product.
        */
        for(int j = depth - 2; j >= 0; --j)
        {
            delta[depth - 1 - j] = (net_ptr[j + 1].transpose() * delta[depth - 2 - j]) % activation_prime(z[j]);
        }

        //update the weights
        for(int j = depth - 1; j >= 0; --j)
        {
            gradients.net_ptr[j] += delta[depth - 1 - j] * outputs[j].transpose();
            gradients.bias_ptr[j] += delta[depth - 1 - j];
        }
    }

    for(int i = depth - 1; i >= 0; --i)
    {
        /*for L2 normalization uncomment / comment the following line for perormance reasons;
        	you can also just set the parameter to 0, but this may impact speed*/
        net_ptr[i] *= 1 - (learning_rate * normalization_factor / training_set_size);
        net_ptr[i] -= gradients.net_ptr[i] * (learning_rate / input.width());
        bias_ptr[i] -= gradients.bias_ptr[i] * (learning_rate / input.width());
    }

    //return cost_function(forward_batch(input), output);
}

//same as Network backpropagate(), but with reference to a gradient network to make algorithm faster (see Netwotk backpropagate())
void Network::backpropagate(const matrix<double>& input, const matrix<double>& output, Network& gradients)
{
    matrix<double> wi;
    matrix<double> z[depth];
    matrix<double> out = input;
    matrix<double> outputs[depth + 1];
    outputs[0] = out;

    for(int i = depth - 1; i >= 0; --i)
    {
        wi = (net_ptr[i] * out) + bias_ptr[i];
        z[i] = wi;
        out = activation(wi);
        outputs[i + 1] = out;
    }

    matrix<double> delta[depth];

    delta[0] = (outputs[depth] - output) % activation_prime(z[depth - 1]);

    for(int i = depth - 2; i >= 0; --i)
    {
        delta[depth - 1 - i] = (net_ptr[i + 1].transpose() * delta[depth - 2 - i]) % activation_prime(z[i]);
    }

    for(int i = depth - 1; i >= 0; --i)
    {
        //add new deltas to old ones, so gradient_descent() can calculate the mean of them
        gradients.net_ptr[i] += delta[depth - 1 - i] * outputs[i].transpose();
        gradients.bias_ptr[i] += delta[depth - 1 - i];
    }
}

//creates [size] unique random integers between 0 and range - 1
matrix<int> Network::shuffle(int range, int size)
{
    //output vector
    matrix<int> out(1, size);

    //create a vector with all numbers between 0 and range - 1
    std::vector<int> vec;
    for(int i = range - 1; i >= 0; --i)
    {
        vec.push_back(i);
    }

    //create an integer to store the random index
    int index;

    //select the numbers
    for(int i = size - 1; i >= 0; --i)
    {
        //generate a random index
        index = rand() % vec.size();
        //save number with that index in out
        out[i] = vec.at(index);
        //delete used element
        vec.erase(vec.begin() + index);
    }

    return out;
}

//outputs a vector containing the outputs of the derivative of the activation activation function
//of each entry in m
matrix<double> Network::activation_prime(const matrix<double>& m)
{
    //create output matrix
    matrix<double> out = matrix<double>(m.height(), 1);

    for(int i = m.height() - 1; i >= 0; --i)
    {
        out(i, 0) = exp(-m(i, 0)) / ( (1 + exp(-m(i, 0))) * (1 + exp(-m(i, 0))) );
    }

    return out;
}

//creates a vector containing the activation of each entry of m
matrix<double> Network::activation(const matrix<double>& m)
{
    //create output matrix
    matrix<double> out = matrix<double>(m.height(), 1);

    //set the according entry in the output-vector to the activation of this entry
    for(int i = m.height() - 1; i >= 0; --i)
    {
        out(i, 0) = 1 / ( 1 + exp(-m(i, 0)) );
    }

    return out;
}

//cost function for backpropagation
//act_out is the output vector of the Network
//des_out is the desired output vector
//calculates the error for the Network
double Network::cost_function(const matrix<double>& act_out, const matrix<double>& des_out)
{
    //uncomment the line of the cost finction you want to use
    //quadratic
    //return quadratic(act_out, des_out);
    //cross entropy
    return cross_entropy(act_out, des_out);
}

inline double Network::quadratic(const matrix<double>& act_out, const matrix<double>& des_out)
{
    //variable for error value
    double err = 0;

    //matrix for calculations
    matrix<double> temp = (des_out - act_out) % (des_out - act_out);

    int h = temp.height(), w = temp.width();

    for(int i = w - 1; i >= 0; --i)
    {
        err += temp.column(i).sum() / (2 * h);
    }

    //return error value
    return err / w;
}

inline double Network::cross_entropy(const matrix<double>& act_out, const matrix<double>& des_out)
{
    //variable for error value
    double err = 0;

    int h = act_out.height(), w = act_out.width();

    for(int i = w - 1; i >= 0; --i)
    {
        for(int j = h - 1; j >= 0; --j)
        {
            err += des_out(j, i) * log(act_out(j, i)) + (1 - des_out(j, i)) * log(1 - act_out(j, i));
        }
    }

    //return error value
    return -err / w;
}
