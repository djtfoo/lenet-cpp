#ifndef MAP_H
#define MAP_H

#include <stdio.h>
#include <string>
#include <sstream>

template<class T>
class Map {
private:
    int _length;
    T** _values;

public:
    // default constructor for dynamic allocation
    Map() : _length(0), _values(nullptr) {}

    Map(int len) : _length(len), _values(nullptr) {
        init(len);
    }
    virtual ~Map() {
        if (_values != nullptr) {
            for (int i = 0; i < _length; ++i) {
                delete[] _values[i];
            }
            delete[] _values;
        }
    }

    void init(int len) {
        _length = len;
        _values = new T * [len];
        for (int i = 0; i < len; ++i) {
            _values[i] = new T[len];
        }
    }

    void set_cell(T val, int i, int j) {
        _values[i][j] = val;
    }

    void print() {

        for (int i = 0; i < _length; ++i) {
            for (int j = 0; j < _length; ++j) {
                printf("%d ", _values[i][j]);
            }
            printf("\n");
        }
    }

    std::string to_string() {

        std::ostringstream ss;
        for (int i = 0; i < _length; ++i) {
            for (int j = 0; j < _length; ++j) {
                //printf("%.2f ", (float)_values[i][j]);
                ss << _values[i][j];
                if (j < _length - 1)
                    ss << " ";
            }
            //printf("\n");
            if (i < _length - 1)
                ss << "\n";
        }

        return ss.str();
    }

    friend class Lenet5;
    //friend int max_pool(FeatureMap* inputMap, int i_start, int j_start);
};

//typedef Map<char> ImageMap; // 8-bit (0-255) for each image pixel
typedef Map<float> FeatureMap;  // outputs of convolution and FC are floats

#endif