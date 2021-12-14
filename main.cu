#include <iostream>
#include <chrono>

#define NUM_POLYGONS 100

bool pointInTri(float* p, float p1x, float p1y);
bool checkTesselatedTris(float* p1, float p1_num_verts, float* p2, float p2_num_verts);
bool checkOverlap(float* p1, float p1_num_verts, float* p2, float p2_num_verts);

__device__ bool pointInTri_device(float* p, float p1x, float p1y);
__device__ bool checkTesselatedTris_device(float* p1, float p1_num_verts, float* p2, float p2_num_verts);
__device__ bool checkOverlap_device(float* p1, float p1_num_verts, float* p2, float p2_num_verts);


__global__ void parallel_checkOverlap(float* dps, int* dp_counts, int num_polys) {
    int shift = gridDim.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = offset; i < num_polys; i += shift) {
        for(int j = i + 1; j < num_polys; ++j) {
            bool result = checkOverlap_device(&dps[i * 10 * 2], dp_counts[i], &dps[j * 10 * 2], dp_counts[j]);
        }
    }
}

int main() {
    // TODO: Call delete and free and cudafree
    srand(time(0));
    // 3 Polygons
    float* polygons = new float[NUM_POLYGONS * 10 * 2];
    int* polygonCounts = new int[NUM_POLYGONS];
    for (int i = 0; i < NUM_POLYGONS; ++i) {
        // Assuming all polygons have 3-10 vertices, chosen at random
        int randNum = (rand() % 8) + 3;
        int numVertices = randNum;
        polygonCounts[i] = numVertices;
        for(int j = 0; j < numVertices; ++j) { 
            polygons[i * 10 * 2 + j * 2 + 0] = rand()/(float) RAND_MAX;
            polygons[i * 10 * 2 + j * 2 + 1] = rand()/(float) RAND_MAX;
        }
    }


    auto start = std::chrono::steady_clock::now();
    // Iteration through triangles inspired by selection sort.  All pairs of triangles need to be checked for overlapping, so no further optimization can be made
    for(int i = 0; i < NUM_POLYGONS; ++i) {
        for(int j = i + 1; j < NUM_POLYGONS; ++j) {
            bool result = checkOverlap(&polygons[i * 10 * 2], polygonCounts[i], &polygons[j * 10 * 2], polygonCounts[j]);
            // std::cout << "RESULT: " << result << std::endl;
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time to complete serial implementation: " <<  diff.count() << "\n";

    start = std::chrono::steady_clock::now();
    // TODO: Parallel implementation
    
    float* dps;
    int* dp_counts;
    cudaMalloc((void**)&dps, NUM_POLYGONS * 10 * 2 * sizeof(float)); // Since max Vertices is 10 we're allocating that :)
    //dps[polyNum * 10 * 2 + vertexNum * 2 + offset(0/1) ]
    cudaMalloc((void**)&dp_counts, NUM_POLYGONS * sizeof(int));

    cudaMemcpy(dps, polygons, NUM_POLYGONS * 10 * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dp_counts, polygonCounts, NUM_POLYGONS * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = 1;
    int numThreads = 1;
    parallel_checkOverlap<<<numBlocks, numThreads>>>(dps, dp_counts, NUM_POLYGONS);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "Time to complete parallel implementation: " <<  diff.count() << "\n";

}

bool pointInTri(float* p, float p1x, float p1y) {
    bool inTriangle = true;
    
    // If point not within triangle, then 3 eqs bounding triangle will not be true, so inTriangle will be false
    for(int i = 0; i < 3; ++i) {
        if (p[((i+1) % 3) * 2 + 0] - p[i * 2 + 0] == 0) {
            bool isRight = p[((i+2) % 3 * 2) + 0] > p[i * 2 + 0];
            inTriangle = inTriangle && (isRight ? p1x >= p[i * 2 + 0] : p1x <= p[i * 2 + 0]); 
        } else {
            float m = (p[((i+1) % 3) * 2 + 1] - p[i * 2 + 1])/(p[((i+1) % 3) * 2 + 0] - p[i * 2 + 0]);
            float b = -m * p[i * 2 + 0] + p[i * 2 + 1];

            bool isGreater = p[((i+2) % 3) * 2 + 1] > (m * p[((i+2) % 3) * 2 + 0] + b);
            inTriangle = inTriangle && (isGreater ? p1y >= (m * p1x + b) : p1y <= (m * p1x + b));
        }
    }

    return inTriangle;
}

bool checkTesselatedTris(float* p1, float p1_num_verts, float* p2, float p2_num_verts) {
    int v0 = 0;
    int v1 = 1;
    int v2 = p1_num_verts - 1;
    bool collision = false;
    float* tempTri = new float[6];

    while(v2-v1 >= 1) {
        tempTri[0 * 2 + 0] = p1[v0 * 2 + 0];
        tempTri[0 * 2 + 1] = p1[v0 * 2 + 1];
        tempTri[1 * 2 + 0] = p1[v1 * 2 + 0];
        tempTri[1 * 2 + 1] = p1[v1 * 2 + 1];
        tempTri[2 * 2 + 0] = p1[v2 * 2 + 0];
        tempTri[2 * 2 + 1] = p1[v2 * 2 + 1];

        for(int j = 0; j < p2_num_verts; ++j) {
            collision = collision || pointInTri(tempTri, p2[j * 2 + 0], p2[j * 2 + 1]);
        }
        v0 = v1;
        v1 = v2;
        v2 = v0 - 1; // v0 now is the previous state of v1

        if (v2-v1 < 1) {
            break;
        }

        tempTri[0 * 2 + 0] = p1[v0 * 2 + 0];
        tempTri[0 * 2 + 1] = p1[v0 * 2 + 1];
        tempTri[1 * 2 + 0] = p1[v1 * 2 + 0];
        tempTri[1 * 2 + 1] = p1[v1 * 2 + 1];
        tempTri[2 * 2 + 0] = p1[v2 * 2 + 0];
        tempTri[2 * 2 + 1] = p1[v2 * 2 + 1];

        for(int j = 0; j < p2_num_verts; ++j) {
            collision = collision || pointInTri(tempTri, p2[j * 2 + 0], p2[j * 2 + 1]);
        }

        v0 = v1;
        v1 = v2;
        v2 = v0 + 1;
    }

    return collision;
}

bool checkOverlap(float* p1, float p1_num_verts, float* p2, float p2_num_verts) {
    // TODO: Check if is line instead of polygon

    // Using custom algorithm I wrote, assuming polygon does not self-intersect and has non-identical vertices, and they're not a line either
    bool collision = false;
    // p1 tesselated into triangles, checking all points of p2 in each tri
    collision = collision || checkTesselatedTris(p1, p1_num_verts, p2, p2_num_verts);
    // p2 tesseleated into triangles, checking all points of p1 in each tri
    collision = collision || checkTesselatedTris(p2, p2_num_verts, p1, p1_num_verts);

    return collision;
}

__device__ bool pointInTri_device(float* p, float p1x, float p1y) {
    bool inTriangle = true;
    
    // If point not within triangle, then 3 eqs bounding triangle will not be true, so inTriangle will be false
    for(int i = 0; i < 3; ++i) {
        if (p[((i+1) % 3) * 2 + 0] - p[i * 2 + 0] == 0) {
            bool isRight = p[((i+2) % 3 * 2) + 0] > p[i * 2 + 0];
            inTriangle = inTriangle && (isRight ? p1x >= p[i * 2 + 0] : p1x <= p[i * 2 + 0]); 
        } else {
            float m = (p[((i+1) % 3) * 2 + 1] - p[i * 2 + 1])/(p[((i+1) % 3) * 2 + 0] - p[i * 2 + 0]);
            float b = -m * p[i * 2 + 0] + p[i * 2 + 1];

            bool isGreater = p[((i+2) % 3) * 2 + 1] > (m * p[((i+2) % 3) * 2 + 0] + b);
            inTriangle = inTriangle && (isGreater ? p1y >= (m * p1x + b) : p1y <= (m * p1x + b));
        }
    }

    return inTriangle;
}

__device__ bool checkTesselatedTris_device(float* p1, float p1_num_verts, float* p2, float p2_num_verts) {
    int v0 = 0;
    int v1 = 1;
    int v2 = p1_num_verts - 1;
    bool collision = false;
    float* tempTri = new float[6];

    while(v2-v1 >= 1) {
        tempTri[0 * 2 + 0] = p1[v0 * 2 + 0];
        tempTri[0 * 2 + 1] = p1[v0 * 2 + 1];
        tempTri[1 * 2 + 0] = p1[v1 * 2 + 0];
        tempTri[1 * 2 + 1] = p1[v1 * 2 + 1];
        tempTri[2 * 2 + 0] = p1[v2 * 2 + 0];
        tempTri[2 * 2 + 1] = p1[v2 * 2 + 1];

        for(int j = 0; j < p2_num_verts; ++j) {
            collision = collision || pointInTri_device(tempTri, p2[j * 2 + 0], p2[j * 2 + 1]);
        }
        v0 = v1;
        v1 = v2;
        v2 = v0 - 1; // v0 now is the previous state of v1

        if (v2-v1 < 1) {
            break;
        }

        tempTri[0 * 2 + 0] = p1[v0 * 2 + 0];
        tempTri[0 * 2 + 1] = p1[v0 * 2 + 1];
        tempTri[1 * 2 + 0] = p1[v1 * 2 + 0];
        tempTri[1 * 2 + 1] = p1[v1 * 2 + 1];
        tempTri[2 * 2 + 0] = p1[v2 * 2 + 0];
        tempTri[2 * 2 + 1] = p1[v2 * 2 + 1];

        for(int j = 0; j < p2_num_verts; ++j) {
            collision = collision || pointInTri_device(tempTri, p2[j * 2 + 0], p2[j * 2 + 1]);
        }

        v0 = v1;
        v1 = v2;
        v2 = v0 + 1;
    }

    return collision;
}

__device__ bool checkOverlap_device(float* p1, float p1_num_verts, float* p2, float p2_num_verts) {
    // TODO: Check if is line instead of polygon

    // Using custom algorithm I wrote, assuming polygon does not self-intersect and has non-identical vertices, and they're not a line either
    bool collision = false;
    // p1 tesselated into triangles, checking all points of p2 in each tri
    collision = collision || checkTesselatedTris_device(p1, p1_num_verts, p2, p2_num_verts);
    // p2 tesseleated into triangles, checking all points of p1 in each tri
    collision = collision || checkTesselatedTris_device(p2, p2_num_verts, p1, p1_num_verts);

    return collision;
}