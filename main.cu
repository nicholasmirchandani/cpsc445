#include <iostream>

#define NUM_POLYGONS 3

bool pointInTri(float** p, float p1x, float p1y) {
    // NOTE: numVerts no longer needed since assuming triangles at this point
    std::cout << "Checking the point of (" << p1x << ", " << p1y << ") " << std::endl;
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

bool checkOverlap(float** p1, float p1_num_verts, float** p2, float p2_num_verts) {
    std::cout << "Checking polygons:  [ ";
    for(int i = 0; i < p1_num_verts; ++i) {
        std::cout << "( " << p1[i][0] << ", " << p1[i][1] << "), ";
    }
    std::cout << "] and [ ";
    for(int i = 0; i < p2_num_verts; ++i) {
        std::cout << "( " << p2[i][0] << ", " << p2[i][1] << "), ";
    }
    std::cout << "] " << std::endl;

    if(p1_num_verts != 3) {
        std::cout << "P1 not have 3 vertices!" << std::endl; // TODO: Tesselate
    }

    if(p2_num_verts != 3) {
        std::cout << "P2 not have 3 vertices" << std::endl; // TODO: Tesselate
    }

    // Separating axis theorem on all line segments, assuming triangle

    // If any vertex is within the other triangle, triangles intersect (only true with coplanar triangles in two dimensions)
    bool collision = false;
    collision = collision || pointInTri(p2, p1[0][0], p1[0][1]);
    collision = collision || pointInTri(p2, p1[1][0], p1[1][1]);
    collision = collision || pointInTri(p2, p1[2][0], p1[2][1]);
    collision = collision || pointInTri(p1, p2[0][0], p2[0][1]);
    collision = collision || pointInTri(p1, p2[1][0], p2[1][1]);
    collision = collision || pointInTri(p1, p2[2][0], p2[2][1]);

    return collision;
}

int main() {
    std::cout << "Hello World!" << std::endl;
    // 3 Polygons
    float*** polygons = new float**[NUM_POLYGONS];
    for (int i = 0; i < NUM_POLYGONS; ++i) {
        // Assuming all polygons are triangles; 3's here are numVertices
        polygons[i] = new float*[3];
        for(int j = 0; j < 3; ++j) {
            polygons[i][j] = new float[2];
        }
        polygons[i][0][0] = i + 1;
        polygons[i][1][0] = i + 2;
        polygons[i][2][0] = 3 * i;
        polygons[i][0][1] = i - 1;
        polygons[i][1][1] = i - 2;
        polygons[i][2][1] = 13 - i;
    }

    polygons[0][0][0] = 0;
    polygons[0][0][1] = 0;
    polygons[0][1][0] = 2;
    polygons[0][1][1] = 0;
    polygons[0][2][0] = -3;
    polygons[0][2][1] = 2;

    polygons[1][0][0] = 0;
    polygons[1][0][1] = 1;
    polygons[1][1][0] = 2;
    polygons[1][1][1] = 2;
    polygons[1][2][0] = 2;
    polygons[1][2][1] = -2;

    // Iteration through triangles inspired by selection sort.  All pairs of triangles need to be checked for overlapping, so no further optimization can be made
    for(int i = 0; i < NUM_POLYGONS; ++i) {
        for(int j = i + 1; j < NUM_POLYGONS; ++j) {
            std::cout << "RESULT: " << checkOverlap(polygons[i], 3, polygons[j], 3) << std::endl;      
        }
    }
}
