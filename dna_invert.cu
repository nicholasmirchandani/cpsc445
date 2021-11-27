#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#define MAX_BUF 10000

__global__ void invert(char* dn, int numChars) {
    int shift = gridDim.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = offset; i < numChars; i += shift) {
        switch(dn[i]) {
            case 'A':
                dn[i] = 'T';
                break;
            case 'C':
                dn[i] = 'G';
                break;
            case 'T':
                dn[i] = 'A';
                break;
            case 'G':
                dn[i] = 'C';
                break;
        }
    }
}

int main() {

    char* n = new char[MAX_BUF];

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
    cudaMalloc((void **)&dn, MAX_BUF * sizeof(char));
    cudaMemcpy(dn, n, MAX_BUF * sizeof(char), cudaMemcpyHostToDevice);

    int numBlocks = 4;
    int numThreads = 4;
    invert<<<numBlocks, numThreads>>>(dn, strlen(n));
    cudaDeviceSynchronize();

    cudaMemcpy(n, dn, MAX_BUF * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(dn);
    std::ofstream os("output.txt");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    os << n;

    os.close();
    delete(n);
    return 0;
}
