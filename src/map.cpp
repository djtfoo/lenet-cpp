#include <stdio.h>
#include "map.h"

template <class T>
Map<T>::Map(int len) : _length(len), _values(nullptr) {
    init(len);
}

template <class T>
Map<T>::~Map() {
    if (_values != nullptr) {
        for (int i = 0; i < _length; ++i) {
            delete[] _values[i];
        }
        delete[] _values;
    }
}

template <class T>
void Map<T>::init(int len) {
    _length = len;
    _values = new T * [len];
    for (int i = 0; i < len; ++i) {
        _values[i] = new T[len];
    }
}

template <class T>
void Map<T>::set_cell(T val, int i, int j) {
    _values[i][j] = val;
}

template <class T>
void Map<T>::print() {
    for (int i = 0; i < _length; ++i) {
        for (int j = 0; j < _length; ++j) {
            printf("%d ", _values[i][j]);
        }
        printf("\n");
    }
}