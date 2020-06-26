#include <iostream>
#include <fstream>

using namespace std;

unsigned char* readBMP(const char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    for(i = 0; i < size; i += 3)
    {
        unsigned char tmp = data[i];
        data[i] = data[i+2];
        data[i+2] = tmp;
    }

    return data;
}

int main()
{

    unsigned char* data = readBMP("test.bmp");

    ofstream file("test1.txt");

    for(int i = 27; i >= 0; i--)
    {
        for(int j = 0; j < 28; j++)
        {
            /*cout << +data[3 * (i * 28 + j)] << "\n";
            cout << +data[3 * (i * 28 + j) + 1] << "\n";
            cout << +data[3 * (i * 28 + j) + 2] << "\n\n";*/
            if(+data[3 * (i * 28 + j)] > 0)
            {
                file << 0 << "\n";
            }
            else
            {
                file << 1 << "\n";
            }
        }
    }

    file.close();

    ifstream image("test1.txt");

    int ctr = 0;

    string str;
    double value;

    while(getline(image, str))
    {
        value = stod(str);

        if(value < 0.5)
        {
            cout << "0";
        }
        else
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

    image.close();

    return 0;
}
