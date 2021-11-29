#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#define MAX_BUF 10000

__global__ void genes_calc(char* dn, int* dstates, int numChars) {
    int shift = gridDim.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    numChars = (numChars / shift) + ((numChars % shift) == 0 ? 0 : 1);
    while(numChars % 3 != 0) {
        ++numChars;
    }
    offset *= numChars;

    for(int i = offset; i + 2 < numChars + offset; i += 3) {
        int index = i/3;
        if(dn[i] == 'A' && dn[i+1] == 'T' && dn[i+2] == 'G') {
            dstates[index] = 1;
        }

        if(dn[i] == 'T' && dn[i+1] == 'A' && dn[i+2] == 'G') {
            dstates[index] = 2;
        }

        if(dn[i] == 'T' && dn[i+1] == 'A' && dn[i+2] == 'A') {
            dstates[index] = 2;
        }

        if(dn[i] == 'T' && dn[i+1] == 'G' && dn[i+2] == 'A') {
            dstates[index] = 2;
        }
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
                tmpsums[i + 64 * threadIdx.x] = tmpsums[i + 64 * threadIdx.x] + tmpsums[i + 64 * (threadIdx.x + delta)];
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
    int* states = new int[MAX_BUF];
    for(int i = 0; i < MAX_BUF; ++i) {
        states[i] = 0;
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
    int* dstates;
    int numBlocks = 4;
    int numThreads = 4;
    cudaMalloc((void**) &dn, MAX_BUF * sizeof(char));
    cudaMalloc((void**) &dstates, MAX_BUF * numThreads * numBlocks * sizeof(int));
    cudaMemcpy(dn, n, MAX_BUF * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dstates, states, MAX_BUF * numThreads * numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    genes_calc<<<numBlocks, numThreads>>>(dn, dstates, strlen(n));
    cudaDeviceSynchronize();

    cudaMemcpy(states, dstates, MAX_BUF * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dn);
    cudaFree(dstates);

    bool inGene = false;
    int startIndex = -1;
    std::ofstream os("output.txt");
    
    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
        exit(1);
    }

    for(int i = 0; i < MAX_BUF; ++i) {
        if(states[i] == 1 && !inGene) {
            inGene = true;
            startIndex = i;
        } else if (states[i] == 2 && inGene) {
            inGene = false;
            os << startIndex << " " << i << std::endl;
        }
    }


    os.close();
    delete(n);
    delete(states);
    return 0;
}
