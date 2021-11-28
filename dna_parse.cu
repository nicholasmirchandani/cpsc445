#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#define MAX_BUF 10000

char returnNucleotide(int num) {
    switch(num) {
        case 0:
            return 'A';
        case 1:
            return 'C';
        case 2:
            return 'G';
        case 3:
            return 'T';
        default:
            return '?';
    }
}

__global__ void count_step_1(char* dn, int* dsums, int numChars) {
    int shift = gridDim.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 64; ++i) {
        dsums[i + 64 * offset] = 0;
    }
    
    // Count every trigram
    const int A_KEY = 0;
    const int C_KEY = 1;
    const int T_KEY = 2;
    const int G_KEY = 3;
    
    int curKeys[3];

    numChars = (numChars / shift) + ((numChars % shift) == 0 ? 0 : 1);
    offset *= numChars;

    for(int i = offset; i < numChars; i += 3) {
        curKeys[0] = 0;
        curKeys[1] = 1;
        curKeys[2] = 2;
        for(int j = i; j < i + 3; ++j) {
            switch(dn[j]) {
                case 'A':
                    curKeys[j] = A_KEY;
                    break;
                case 'C':
                    curKeys[j] = C_KEY;
                    break;
                case 'T':
                    curKeys[j] = T_KEY;
                    break;
                case 'G':
                    curKeys[j] = G_KEY;
                    break;
            }
        }
        int targetIndex = curKeys[2] * 16 + curKeys[1] * 4 + curKeys[0];
        dsums[targetIndex + 64 * offset] += 1;
    }

    __syncthreads();
    __shared__ int tmpsums[4*64]; // 4*64 for 4 threads per block

    for(int i = 0; i < 64; ++i) {
        tmpsums[i + 64 * threadIdx.x] = dsums[i + 64 * offset];
    }

    __syncthreads();

    for(int delta = 1; delta < blockDim.x; delta *= 2) {
        if((threadIdx.x + delta) < blockDim.x) {
            for(int i = 0; i < 64; ++i) {
                tmpsums[i + 64 * threadIdx.x] = tmpsums[i + 64 * threadIdx.x] + tmpsums[i + 64 * (delta + threadIdx.x)];
            }
        }
    }
    __syncthreads();

    // Combine all threads within the same block; Currently issue with shared memory not being used properly

    offset = blockDim.x * blockIdx.x;

    for(int i = 0; i < 64; ++i) {
        dsums[i + 64 * offset] = tmpsums[i];
    }
}

__global__ void count_step_2(int* dsums, int W) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x; // NOTE: blockDim should be 1 and blockIdx should be 0
    
    __shared__ int tmpsums[64 * 4]; // 64 * 4 for 4 threads per block

    for(int i = 0; i < 64; ++i) {
        tmpsums[i + (64 * threadIdx.x)] = dsums[i + 64 * W * offset];
    }

    __syncthreads();

    
    for(int delta = 1; delta < blockDim.x; delta *= 2) {
        if((threadIdx.x + delta) < blockDim.x) {
            for(int i = 0; i < 64; ++i) {
                tmpsums[i + 64 * threadIdx.x] = tmpsums[i + 64 * threadIdx.x] + tmpsums[i + 64 * (i + delta)];
            }
        }
        __syncthreads();
    }
    
    for(int i = 0; i < 64; ++i) {
        dsums[i] = tmpsums[i];
    }

}

int main() {

    char* n = new char[MAX_BUF];
    int* sums = new int[64];
    for(int i = 0; i < 64; ++i) {
        sums[i] = 0;
    }

    std::ifstream is("dna.txt");

    if (is.fail()) {
    std::cout << "Unable to open dna file.  Exiting " << std::endl;
    exit(1);
    }

    std::string allChars = "";
    std::string line;
    while(is.good()) {
    std::getline(is, line);

    // Remove trailing newline and carriage returns
    while(line[line.length()-1] == '\n' || line[line.length()-1] == '\r') {
        line.pop_back();
    }

    allChars.append(line);
    }

    is.close();

    strcpy(n, allChars.c_str());
    for(int i = strlen(n); i < MAX_BUF; ++i) {
    // Ensures 0s are padded after string as needed
    n[i] = '\0';
    }

    char* dn;
    int* dsums;
    int numBlocks = 4;
    int numThreads = 4;
    cudaMalloc((void**) &dn, MAX_BUF * sizeof(char));
    cudaMalloc((void**) &dsums, 64 * numThreads * numBlocks * sizeof(int));
    cudaMemcpy(dn, n, MAX_BUF * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dsums, sums, 64 * numThreads * numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Step 1!" << std::endl;
    count_step_1<<<numBlocks, numThreads>>>(dn, dsums, strlen(n));
    cudaDeviceSynchronize();

    std::cout << "Step 2!" << std::endl;
    count_step_2<<<1, numBlocks>>>(dsums, numThreads);
    cudaDeviceSynchronize();

    cudaMemcpy(sums, dsums, 64 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dn);
    cudaFree(dsums);

    std::ofstream os("output.txt");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            for(int k = 0; k < 4; ++k) {
                // if(sums[i + 4 * j + 16 * k] != 0) {
                    os << returnNucleotide(i) << returnNucleotide(j) << returnNucleotide(k) << " " << sums[i + 4 * j + 16 * k] << std::endl;
                // }
            }
        }
    }

    os.close();
    delete(n);
    delete(sums);
    return 0;
}
