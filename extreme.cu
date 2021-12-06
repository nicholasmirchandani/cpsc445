#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define MAX_BUF 200000

void readCSV(float* nums, int& numFloats, int& floatsPerRow, std::string filename) {
    std::ifstream is(filename);

    if (is.fail()) {
        std::cout << "Unable to open dna file.  Exiting " << std::endl;
        exit(1);
    }

    std::string allChars = "";
    std::string line;
    while(is.good()) {
        std::getline(is, line);
        std::cout << "Line read: " << line << std::endl;

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
    
        if(floatsPerRow == -1) {
            floatsPerRow = numFloats;
            std::cout << "Floats per row set!" << std::endl;
        }
    }
    std::cout << "is is not good :(" << std::endl;
}

__global__ void find_extremes(float* dnums, int numFloats, int floatsPerRow, bool* disExtreme) {
    int shift = gridDim.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    bool isMin = true;
    bool isMax = true;
    for(int i = offset; i < numFloats; i += shift) {
        // For 8 neighbors, where i - floatsPerRow - 1 is top left and i + floatsPerRow + 1 is bottom right
        for(int j = -1; j < 2; ++j) {
            for(int k = -1; k < 2; ++k) {
                if(j == 0 && k == 0) {
                    // Skip element in its own loop
                    continue;
                }

                isMin = isMin && (dnums[i] < dnums[i + j * floatsPerRow + k * 1]); // Neighbor :)
                isMax = isMax && (dnums[i] > dnums[i + j * floatsPerRow + k * 1]);
            }
        }
        
        disExtreme[i] = isMin || isMax;
    }
}

int main() {
    std::cout << "MAIN!" << std::endl;
    float nums[MAX_BUF];
    bool isExtreme[MAX_BUF];
    int numFloats = 0;
    int floatsPerRow = -1;

    system("head input.csv");

    readCSV(nums, numFloats, floatsPerRow, "input.csv");

    // Now we have the csv properly parsed, we do the parallel sqrt computation
    float* dnums;
    bool* disExtreme;
    cudaMalloc((void**) &dnums, numFloats * sizeof(float));
    cudaMemcpy(dnums, nums, numFloats * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &disExtreme, numFloats * sizeof(bool));

    int numBlocks = 2;
    int numThreads = 4;

    std::cout << "Find extremes!" << std::endl;
    find_extremes<<<numBlocks, numThreads>>>(dnums, numFloats, floatsPerRow, disExtreme);

    cudaMemcpy(isExtreme, disExtreme, numFloats * sizeof(bool), cudaMemcpyDeviceToHost);

    std::ofstream os("output.csv");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }


    std::cout << "RESULTS: " << std::endl << std::endl;
    for(int i = 0; i < numFloats - 1; ++i) {
        int rowNum = i / floatsPerRow;
        int colNum = i % floatsPerRow;
        if(isExtreme[i]) {
            std::cout << "(" << colNum << "," << rowNum << ")" << std::endl;
        }
    }

    os.close();

}
