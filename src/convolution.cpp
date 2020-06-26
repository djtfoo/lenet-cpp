#include "map.h"
#include "kernel.h"
#include "lenet5.h"


// friend functions of Map

/*int convolution(int map_input[CONV][CONV], int weights[CONV][CONV], int bias) {

    int convResult = 0;
    for (int i = 0; i < CONV; i++) {
        for (int j = 0; j < CONV; j++) {
            convResult += map_input[i][j] * weights[i][j];
        }
    }

    return convResult + bias;
}

int max_pool(int map_input[POOL][POOL]) {
    
    int max = map_input[0][0];
    for (int i = 0; i < POOL; i++) {
        for (int j = 0; j < POOL; j++) {
            if (map_input[i][j] > max)
                max = map_input[i][j];
        }
    }

    return max;
}*/