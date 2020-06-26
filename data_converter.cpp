#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    ifstream data("test_data.txt");
    ofstream file("test2.txt");

    string str;
    double value;
    int ctr = 0;

    while(getline(data, str))
    {
        value = stod(str);

        if(++ctr < 785)
        {
            if(value < 0.5)
            {
                file << "0\n";
            }
            else
            {
                file << "1\n";
            }
        }
        else
        {
            file << (int)value << "\n";
            ctr = 0;
        }
    }

    file.close();
    data.close();

    return 0;
}
