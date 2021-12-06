#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define MAX_BUF 1000

void readCSV(float* nums, int& numFloats, std::string filename) {
    std::ifstream is(filename);

    if (is.fail()) {
        std::cout << "Unable to open dna file.  Exiting " << std::endl;
        exit(1);
    }

    std::string allChars = "";
    std::string line;
    while(is.good()) {
        std::getline(is, line);

        // Remove trailing newline, carriage returns
        while(line[line.length()-1] == '\n' || line[line.length()-1] == '\r' || line[line.length()-1] == ',') {
            line.pop_back();
        }

        // Populate array of data
        std::string element = "";
        for(char c : line) {
            if(c == ' ' || c == ',') {
                if(element != "") {
                    nums[numFloats++] = std::stof(element);
                    element = "";
                }
                continue;
            }

            element += c;
        }

        if (element != "") {
            nums[numFloats++] = std::stof(element);
        }
    }
}

__global__ void cuda_sqrt(float* dnums, int numFloats) {
    int shift = gridDim.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = offset; i < numFloats; i += shift) {
        dnums[i] = sqrt(dnums[i]);
    }
}

int main() {
    float nums[MAX_BUF];
    int numFloats = 0;

    readCSV(nums, numFloats, "input.csv");

    // Now we have the csv properly parsed, we do the parallel sqrt computation
    float* dnums;
    cudaMalloc((void**) &dnums, numFloats * sizeof(float));
    cudaMemcpy(dnums, nums, numFloats * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = 2;
    int numThreads = 4;

    cuda_sqrt<<<numBlocks, numThreads>>>(dnums, numFloats);

    cudaMemcpy(nums, dnums, numFloats * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream os("output.csv");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    for(int i = 0; i < numFloats - 1; ++i) {
        os << nums[i] << std::endl;
    }
    // Last element shouldn't have comma space after it
    os << nums[numFloats-1];

    os.close();

}