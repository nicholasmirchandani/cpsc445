#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#define MAX_BUF 10000

__global__ void count_step_1(char* dn, int* dsums, int numChars) {
    int shift = gridDim.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 4; ++i) {
        dsums[i + 4 * offset] = 0;
    }
    
    for(int i = offset; i < numChars; i += shift) {
        switch(dn[i]) {
            case 'A':
                dsums[0 + 4 * offset] += 1;
                break;
            case 'C':
                dsums[1 + 4 * offset] += 1;
                break;
            case 'T':
                dsums[2 + 4 * offset] += 1;
                break;
            case 'G':
                dsums[3 + 4 * offset] += 1;
                break;
        }
    }

    __syncthreads();
    __shared__ int tmpsums[16]; // 16 for 4 threads per block

    tmpsums[0 + (4 * threadIdx.x)] = dsums[0 + 4 * offset];    
    tmpsums[1 + (4 * threadIdx.x)] = dsums[1 + 4 * offset];
    tmpsums[2 + (4 * threadIdx.x)] = dsums[2 + 4 * offset];
    tmpsums[3 + (4 * threadIdx.x)] = dsums[3 + 4 * offset];    

    __syncthreads();

    int i = threadIdx.x;
    for(int delta = 1; delta < blockDim.x; delta *= 2) {
        if((i + delta) < blockDim.x) {
            tmpsums[0 + 4 * i] = tmpsums[0 + 4 * i] + tmpsums[0 + 4 * (i + delta)];
            tmpsums[1 + 4 * i] = tmpsums[1 + 4 * i] + tmpsums[1 + 4 * (i + delta)];
            tmpsums[2 + 4 * i] = tmpsums[2 + 4 * i] + tmpsums[2 + 4 * (i + delta)];
            tmpsums[3 + 4 * i] = tmpsums[3 + 4 * i] + tmpsums[3 + 4 * (i + delta)];
        }
        __syncthreads();
    }

    __syncthreads();

    // Combine all threads within the same block; Currently issue with shared memory not being used properly

    offset = blockDim.x * blockIdx.x;
    dsums[0 + 4 * offset] = tmpsums[0];    
    dsums[1 + 4 * offset] = tmpsums[1];
    dsums[2 + 4 * offset] = tmpsums[2];
    dsums[3 + 4 * offset] = tmpsums[3];    
}

__global__ void count_step_2(int* dsums, int W) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x; // NOTE: blockDim should be 1 and blockIdx should be 0
    
    __shared__ int tmpsums[16]; // 16 for 4 threads per block

    tmpsums[0 + (4 * threadIdx.x)] = dsums[0 + 4 * W * offset];    
    tmpsums[1 + (4 * threadIdx.x)] = dsums[1 + 4 * W * offset];
    tmpsums[2 + (4 * threadIdx.x)] = dsums[2 + 4 * W * offset];
    tmpsums[3 + (4 * threadIdx.x)] = dsums[3 + 4 * W * offset];    

    __syncthreads();

    int i = threadIdx.x;
    for(int delta = 1; delta < blockDim.x; delta *= 2) {
        if((i + delta) < blockDim.x) {
            tmpsums[0 + 4 * i] = tmpsums[0 + 4 * i] + tmpsums[0 + 4 * (i + delta)];
            tmpsums[1 + 4 * i] = tmpsums[1 + 4 * i] + tmpsums[1 + 4 * (i + delta)];
            tmpsums[2 + 4 * i] = tmpsums[2 + 4 * i] + tmpsums[2 + 4 * (i + delta)];
            tmpsums[3 + 4 * i] = tmpsums[3 + 4 * i] + tmpsums[3 + 4 * (i + delta)];
        }
        __syncthreads();
    }
    
    dsums[0] = tmpsums[0];    
    dsums[1] = tmpsums[1];
    dsums[2] = tmpsums[2];
    dsums[3] = tmpsums[3];   

}

int main() {

    char* n = new char[MAX_BUF];
    int* sums = new int[4];
    for(int i = 0; i < 4; ++i) {
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
    cudaMalloc((void**) &dsums, 4 * numThreads * numBlocks * sizeof(int));
    cudaMemcpy(dn, n, MAX_BUF * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dsums, sums, 4 * numThreads * numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Step 1!" << std::endl;
    count_step_1<<<numBlocks, numThreads>>>(dn, dsums, strlen(n));
    cudaDeviceSynchronize();

    std::cout << "Step 2!" << std::endl;
    count_step_2<<<1, numBlocks>>>(dsums, numThreads);
    cudaDeviceSynchronize();

    cudaMemcpy(sums, dsums, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dn);
    cudaFree(dsums);

    std::ofstream os("output.txt");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    os << "A " << sums[0] << std::endl;
    os << "T " << sums[2] << std::endl;    
    os << "G " << sums[3] << std::endl;      
    os << "C " << sums[1] << std::endl;

    os.close();
    delete(n);
    delete(sums);
    return 0;
}
