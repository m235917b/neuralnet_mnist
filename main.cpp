#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <omp.h>
#include "network.h"

using namespace std;

char input;

void mnist()
{
    matrix<double> input_mnist_train = matrix<double>(784, 50000);
    matrix<double> output_mnist_train = matrix<double>(10, 50000);

    matrix<double> input_mnist_test = matrix<double>(784, 10000);
    matrix<double> output_mnist_test = matrix<double>(1, 10000);

    //mnist training and validation

    matrix<int> net_setup_mnist(1, 3);
    net_setup_mnist(0, 0) = 784;
    net_setup_mnist(0, 1) = 30;
    net_setup_mnist(0, 2) = 10;

    Network net_mnist = Network(net_setup_mnist, 1, 1/sqrt(50000));

    //load training data arrays

    int ctr = 0, data_ctr = 0;

    //ifstream file("training_data.txt");
    ifstream file("test.txt");
    string str;
    while(getline(file, str))
    {
        if(ctr < 784)
        {
            input_mnist_train(ctr, data_ctr) = (double) stod(str);
        }
        else
        {
            output_mnist_train(ctr - 784, data_ctr) = (double) stod(str);
        }

        if(++ctr == 794)
        {
            ctr = 0;
            ++data_ctr;

            cout << data_ctr << "\n";
        }
    }
    file.close();
    ctr = 0;
    data_ctr = 0;

    //load test data arrays

    file.open("test_data.txt");
    while(getline(file, str))
    {
        if(ctr < 784)
        {
            input_mnist_test(ctr, data_ctr) = (double) stod(str);
        }
        else
        {
            output_mnist_test(0, data_ctr) = (double) stod(str);
        }

        if(++ctr == 785)
        {
            ctr = 0;
            ++data_ctr;

            cout << data_ctr << "\n";
        }
    }
    file.close();

    //train the network

    clock_t begin_time = clock();
    //double begin_time = omp_get_wtime();

    net_mnist.sgd(0.5, 5, 10, 1, input_mnist_train, output_mnist_train, input_mnist_test, output_mnist_test);

    /*for(int i = 99999; i >= 0; --i)
    {
    	cout << 99999 - i << "    " << net_mnist.train_rand(-1, input_mnist_train, output_mnist_train) << "    ";

    	//calculate perormance
    	matrix<double> output;
    	int correct_sets = 0;
    	double max_output = 0;

    	for(int j = input_mnist_test.width() - 1; j >= 0; --j)
    	{
    		output = net_mnist.forward(input_mnist_test.column(j));

    		//get neuron with max output value
    		for(int k = output.height() - 1; k >= 0; --k)
    			max_output = output(k, 0) > output(max_output, 0) ? k : max_output;

    		if(max_output == output_mnist_test[j])
    			++correct_sets;
    	}

    	std::cout << correct_sets << " of " << input_mnist_test.width() << "\n";

    }*/

    std::cout << double(clock() - begin_time) /  CLOCKS_PER_SEC << "\n";
    //std::cout << omp_get_wtime() - begin_time << "\n";

    ofstream writer("network.txt");
    writer << "784\n30\n10\n";
    for(int i = 0; i < 2; ++i)
    {
        matrix<double> weights = net_mnist.get(i);
        for(int w = 0; w < weights.width(); ++w)
            for(int h = 0; h < weights.height(); ++h)
                writer << weights(h, w) << "\n";
    }
    writer.close();

    matrix<double> img = matrix<double>(784, 1);
    matrix<double> output;

    while(input != 'q')
    {
        file.open("test1.txt");
        for(int i = 0; i < 784; ++i)
        {
            getline(file, str);
            img.set(i, 0, stod(str));
        }
        file.close();

        output = net_mnist.forward(img);

        int max_output = 0;

        //get neuron with max output value
        for(int i = output.height() - 1; i >= 0; --i)
            max_output = output.get(i, 0) > output.get(max_output, 0) ? i : max_output;

        cout << max_output << "\n";

        cin >> input;
    }
}

/*void test_net()
{
	matrix<double> input_m = matrix<double>(3,4);
	matrix<double> output_m = matrix<double>(3,4);

	std::vector<int> net_setup;
	net_setup.push_back(3);
	net_setup.push_back(5);
	net_setup.push_back(3);

	Network net = Network(net_setup);

	cout << net.to_string();

	input_m.set(0, 0, 0);
	input_m.set(1, 0, 0);
	input_m.set(2, 0, 0);
	input_m.set(0, 1, 1);
	input_m.set(1, 1, 0);
	input_m.set(2, 1, 0);
	input_m.set(0, 2, 0);
	input_m.set(1, 2, 1);
	input_m.set(2, 2, 0);
	input_m.set(0, 3, 1);
	input_m.set(1, 3, 1);
	input_m.set(2, 3, 0);

	cout << input_m.to_string();

	output_m.set(0, 0, 0);
	output_m.set(1, 0, 0);
	output_m.set(2, 0, 0);
	output_m.set(0, 1, 1);
	output_m.set(1, 1, 0);
	output_m.set(2, 1, 1);
	output_m.set(0, 2, 0);
	output_m.set(1, 2, 0);
	output_m.set(2, 2, 1);
	output_m.set(0, 3, 0);
	output_m.set(1, 3, 1);
	output_m.set(2, 3, 0);

	cout << output_m.to_string();

	for(int x = 0; x < 4; x++)
	{
		cout << net.forward(input_m.column(x)).to_string();
	}

	for(int i = 0; i < 10000; i++)
	{
		//Changes weights only after gradient is calculated; converges slower but safer
		//cout << "Error: " << net.train_grad(0.5, 0.1, input_m, output_m) << "\n";
		//Changes every weight bevore claculating difference quotient for next; better with Sigmoid function
		//cout << "Error: " << net.train_grad_double(0.5, 0.1, input_m, output_m) << "\n";
		//trains network by random mutations
		//cout << "Error: " << net.train_rand(-1, input_m, output_m) << "\n";
	}

	net.sgd(2, 4, 10000, input_m, output_m, input_m, output_m);

	cout << net.to_string();

	for(int x = 0; x < 4; x++)
	{
		cout << net.forward(input_m.column(x)).to_string();
	}
}*/

int main()
{
    mnist();
    //test_net();

    /*double a[100000];
    double b[100000];

    for(int i = 0; i < 100000; i++)
    	a[i] = i;

    //for(int i = 0; i < 100; i++)
    	//cout << b[i] << "\n";

    for(int j = 0; j < 500000; j++)
    {
    	//copy(a, a + 5, b);

    	for(int i = 0; i < 100000; i++)
    	{
    		b[i] = a[i];
    	}
    }*/

    //for(int i = 0; i < 100; i++)
    //cout << b[i] << "\n";

    /*matrix<double> m1 = matrix<double>::random(1000,1000);
    matrix<double> m2(matrix<double>::random(1000,1000));
    matrix<double> m3;

    for(int i = 0; i < 10; i++)
    	m3 = m1 * m2;*/

    return 0;
}
