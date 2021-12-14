#include <iostream>

#define NUM_POLYGONS 3

bool checkOverlap(int a1, int a2, int a3, int b1, int b2, int b3) {
    std::cout << "Checking triangles: [ " << a1 << ", " << a2 << ", " << a3 << " ] and [ " << b1 << ", " << b2 << ", " << b3 << " ]" << std::endl;
    return false;
}

int main() {
    std::cout << "Hello World!" << std::endl;
    // 3 Polygons
    int** polygons = new int*[NUM_POLYGONS];
    for (int i = 0; i < NUM_POLYGONS; ++i) {
        // Assuming all polygons are triangles
        polygons[i] = new int[3];
        polygons[i][0] = i + 1;
        polygons[i][1] = i + 2;
        polygons[i][2] = 3 * i;
    }

    // Iteration through triangles inspired by selection sort.  All pairs of triangles need to be checked for overlapping, so no further optimization can be made
    for(int i = 0; i < NUM_POLYGONS; ++i) {
        for(int j = i + 1; j < NUM_POLYGONS; ++j) {
            checkOverlap(polygons[i][0], polygons[i][1], polygons[i][2], polygons[j][0], polygons[j][1], polygons[j][2]);            
        }
    }
}
