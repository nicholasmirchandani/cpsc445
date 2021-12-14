#include <iostream>
#include <chrono>

#define NUM_POLYGONS 1000

bool pointInTri(float** p, float p1x, float p1y);
bool checkTesselatedTris(float** p1, float p1_num_verts, float** p2, float p2_num_verts);
bool checkOverlap(float** p1, float p1_num_verts, float** p2, float p2_num_verts);

int main() {
    srand(time(0));
    // 3 Polygons
    float*** polygons = new float**[NUM_POLYGONS];
    int* polygonCounts = new int[NUM_POLYGONS];
    for (int i = 0; i < NUM_POLYGONS; ++i) {
        // Assuming all polygons have 3-10 vertices, chosen at random
        int randNum = (rand() % 8) + 3;
        int numVertices = randNum;
        polygonCounts[i] = numVertices;
        polygons[i] = new float*[numVertices];
        for(int j = 0; j < numVertices; ++j) { 
            polygons[i][j] = new float[2];

            // TODO: How to randomly generate points?
            polygons[i][j][0] = rand()/(float) RAND_MAX;
            polygons[i][j][1] = rand()/(float) RAND_MAX;
        }
    }


    auto start = std::chrono::steady_clock::now();
    // Iteration through triangles inspired by selection sort.  All pairs of triangles need to be checked for overlapping, so no further optimization can be made
    for(int i = 0; i < NUM_POLYGONS; ++i) {
        for(int j = i + 1; j < NUM_POLYGONS; ++j) {
            bool result =  checkOverlap(polygons[i], polygonCounts[i], polygons[j], polygonCounts[j]);
            // std::cout << "RESULT: " << result << std::endl;
        }
    }
    auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Time to complete serial implementation: " <<  diff.count() << "\n";
}

bool pointInTri(float** p, float p1x, float p1y) {
    bool inTriangle = true;
    
    // If point not within triangle, then 3 eqs bounding triangle will not be true, so inTriangle will be false
    for(int i = 0; i < 3; ++i) {
        if (p[(i+1) % 3][0] - p[i][0] == 0) {
            bool isRight = p[(i+2) % 3][0] > p[i][0];
            inTriangle = inTriangle && (isRight ? p1x >= p[i][0] : p1x <= p[i][0]); 
        } else {
            float m = (p[(i+1) % 3][1] - p[i][1])/(p[(i+1) % 3][0] - p[i][0]);
            float b = -m * p[i][0] + p[i][1];

            bool isGreater = p[(i+2) % 3][1] > (m * p[(i+2) % 3][0] + b);
            inTriangle = inTriangle && (isGreater ? p1y >= (m * p1x + b) : p1y <= (m * p1x + b));
        }
    }

    return inTriangle;
}

bool checkTesselatedTris(float** p1, float p1_num_verts, float** p2, float p2_num_verts) {
    int v0 = 0;
    int v1 = 1;
    int v2 = p1_num_verts - 1;
    bool collision = false;
    float** tempTri = new float* [3];
    for(int i = 0; i < 3; ++i) {
        tempTri[i] = new float[2];
    }

    while(v2-v1 >= 1) {
        tempTri[0][0] = p1[v0][0];
        tempTri[0][1] = p1[v0][1];
        tempTri[1][0] = p1[v1][0];
        tempTri[1][1] = p1[v1][1];
        tempTri[2][0] = p1[v2][0];
        tempTri[2][1] = p1[v2][1];

        for(int j = 0; j < p2_num_verts; ++j) {
            collision = collision || pointInTri(tempTri, p2[j][0], p2[j][1]);
        }
        v0 = v1;
        v1 = v2;
        v2 = v0 - 1; // v0 now is the previous state of v1

        if (v2-v1 < 1) {
            break;
        }

        tempTri[0][0] = p1[v0][0];
        tempTri[0][1] = p1[v0][1];
        tempTri[1][0] = p1[v1][0];
        tempTri[1][1] = p1[v1][1];
        tempTri[2][0] = p1[v2][0];
        tempTri[2][1] = p1[v2][1];

        for(int j = 0; j < p2_num_verts; ++j) {
            collision = collision || pointInTri(tempTri, p2[j][0], p2[j][1]);
        }

        v0 = v1;
        v1 = v2;
        v2 = v0 + 1;
    }

    return collision;
}

bool checkOverlap(float** p1, float p1_num_verts, float** p2, float p2_num_verts) {
    /*std::cout << "Checking polygons:  [ ";
    for(int i = 0; i < p1_num_verts; ++i) {
        std::cout << "( " << p1[i][0] << ", " << p1[i][1] << "), ";
    }
    std::cout << "] and [ ";
    for(int i = 0; i < p2_num_verts; ++i) {
        std::cout << "( " << p2[i][0] << ", " << p2[i][1] << "), ";
    }
    std::cout << "] " << std::endl;*/


    // TODO: Check if is line instead of polygon

    // Using custom algorithm I wrote, assuming polygon does not self-intersect and has non-identical vertices, and they're not a line either
    bool collision = false;
    // p1 tesselated into triangles, checking all points of p2 in each tri
    collision = collision || checkTesselatedTris(p1, p1_num_verts, p2, p2_num_verts);
    // p2 tesseleated into triangles, checking all points of p1 in each tri
    collision = collision || checkTesselatedTris(p2, p2_num_verts, p1, p1_num_verts);

    return collision;
}