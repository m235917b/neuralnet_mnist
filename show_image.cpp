#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    ifstream image("test.txt");

    int ctr = 0;

    string str;
    double value;

    int input;

    cin >> input;

    for(int i = 0; i < input*794; i++)
        getline(image, str);

    for(int i = 0; i < 784; i++)
    {
        getline(image, str);
        value = stod(str);

        if(value == 0)
        {
            cout << "0";
        }
        if(value == 1)
        {
            cout << "1";
        }

        ctr++;

        if(ctr == 28)
        {
            cout << "\n";
            ctr = 0;
        }
    }

    ctr = 0;
    getline(image, str);
    while(str == "0")
    {
        getline(image, str);
        ctr++;
    }

    cout << "\n\n" << ctr << "\n";

    image.close();

    return 0;
}
