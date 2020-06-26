#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <vector>
#include <time.h>

#include "map.h"
#include "imagemap.h"
#include "kernel.h"
#include "lenet5.h"
#include "fcparams.h"


#define IN_LEN  32  // (28x28 with padding)
#define C1_LEN  28
#define S2_LEN  14
#define C3_LEN  10
#define S4_LEN  5
#define C5_LEN  1
#define F6_LEN  84
#define OUT_LEN 10

#define C1_MAPS 6
#define C3_MAPS 16
#define C5_MAPS 120

#define CONV 5
#define POOL 2

#define MAXCHAR 1600    // 1570 characters per row

// read files
bool load_image(ImageMap* image, const char* filename);

// run program
void run_test_lenet5(); // testing
void run_lenet5_dataset();  // read dataset and run lenet-5 on each data

// read dataset
bool read_dataset(std::vector<ImageMap*>& images, const char* filename);

int main() {

    // seed RNG
    srand(time(NULL));

    // run
    //run_test_lenet5();
    run_lenet5_dataset();

    return 0;
}


void run_lenet5_dataset() {

    // instantiate images dataset
    std::vector<ImageMap*> images;  // vector of 32x32 images
    read_dataset(images, "test_dataset.csv");   // read dataset

    // instantiate Lenet-5 neural network
    Lenet5 lenet5;

    // for each image
    for (int i = 0; i < images.size(); ++i) {

        // print image
        printf("Image:\n");
        images[i]->print();
        printf("\n");

        // START COUNTING TIME
        clock_t begin = clock();

        // run inference
        int digit = lenet5.run_inference(images[i]);

        // STOP COUNTING TIME
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

        // Print results
        printf("Predicted Digit: %d\n\n", digit);
        printf("time_spent: %.4f seconds\n", time_spent);
    }

    // delete images after running
    for (int i = 0; i < images.size(); ++i) {
        delete images[i];
    }
}

void run_test_lenet5() {

    ImageMap image(IN_LEN);
    char filename[] = "./test_img.txt";
    if (!load_image(&image, filename))
    {
        printf("Failed to load image\n");
        return;
    }

    // print image
    printf("Image:\n");
    image.print();
    printf("\n");
    //std::cout << image.to_string() << std::endl;

    // instantiate Lenet-5 neural network
    Lenet5 lenet5;

    // START COUNTING TIME
    clock_t begin = clock();

    // run inference
    int digit = lenet5.run_inference(&image);

    // STOP COUNTING TIME
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    // Print results
    printf("Predicted Digit: %d\n\n", digit);
    printf("time_spent: %.4f seconds\n", time_spent);
}

bool read_dataset(std::vector<ImageMap*>& images, const char* filename) {

    FILE* fp;
    errno_t err;
    char str[MAXCHAR];

    // open file
    if ((err = fopen_s(&fp, filename, "r")) != 0) { // file opened unsuccessfully
        fprintf(stderr, "cannot open file '%s'\n", filename);
        return false;
    }

    // read file
    while (fgets(str, MAXCHAR, fp) != NULL) {   // each row is a 28x28 image

        //printf("%s\n", str);

        // initialize new image first
        ImageMap* newImage = new ImageMap(IN_LEN);
        images.push_back(newImage);
        // add zero padding first
        for (int i = 0; i < 32; ++i) {
            // rows
            newImage->set_cell(0, 0, i);    // row 0
            newImage->set_cell(0, 1, i);    // row 1
            newImage->set_cell(0, 30, i);   // row 30
            newImage->set_cell(0, 31, i);   // row 31

            // columns
            if (i > 1 && i < 30) {
                newImage->set_cell(0, i, 0);    // column 0
                newImage->set_cell(0, i, 1);    // column 1
                newImage->set_cell(0, i, 30);   // column 30
                newImage->set_cell(0, i, 31);   // column 31
            }
        }

        char* token, * next_token;
        // retrieve first token - label
        token = strtok_s(str, ",", &next_token);
        newImage->set_label(token[0]);
        // retrieve next token - first cell
        token = strtok_s(NULL, ",", &next_token);
        // loop through the string to add and extract all other tokens
        int row = 0, col = 0;
        while (token != NULL) {
            // convert string to int
            int iVal = std::stoi(token);
            // set respective cell in ImageMap
            newImage->set_cell(iVal, row + 2, col + 2);
            token = strtok_s(NULL, ",", &next_token);
            // update count
            col++;
            if (col >= 28) {
                row++;
                col = 0;
            }
        }
    }
    fclose(fp);

    return true;
}

bool load_image(ImageMap* image, const char* filename) {

    FILE* fp;
    errno_t err;
    char str[MAXCHAR];

    // open file
    if ((err = fopen_s(&fp, filename, "r")) != 0) { // file opened unsuccessfully
        fprintf(stderr, "cannot open file '%s'\n", filename);
        return false;
    }

    // read file
    int i = 0;
    while (fgets(str, MAXCHAR, fp) != NULL) {
        char* token, * next_token;
        token = strtok_s(str, ",", &next_token);
        // loop through the string to extract all other tokens
        int j = 0;
        while (token != NULL) {
            // convert string to int
            int iVal = std::stoi(token);
            //printf( "%d ", iVal ); // print each token
            // set respective cell in ImageMap
            image->set_cell(iVal, i, j);
            token = strtok_s(NULL, ",", &next_token);
            j++;
        }
        i++;
    }
    fclose(fp);

    return true;
}
