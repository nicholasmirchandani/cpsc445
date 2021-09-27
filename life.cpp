/*
Name: Nicholas Mirchandani
ID: 2317024
Email: nmirchandani@chapman.edu
Course: CPSC445-01
Assignment 1: Game of Life
life.cpp contains all classes
*/

#include <iostream>
#include <fstream>
#include <thread>
#include "math.h"
#include <vector>
#include <cstring>

// The Grid class is used to abstract away the concept of a Grid for easy double buffering/shadow paging with pointers.
class Grid {
    public:
        bool** cells;
        int rows;
        int cols;
        Grid(int rows, int cols);
        ~Grid();
};

Grid::Grid(int rows, int cols) {
    // Design: cells[row][col] = appropriate cell
    this->rows = rows;
    this->cols = cols;
    cells = new bool*[rows];
    for(int i = 0; i < rows; ++i) {
        cells[i] = new bool[cols];
    }
}

Grid::~Grid() {
    for(int i = 0; i < rows; ++i) {
        delete(cells[i]);
    }
    delete(cells);
}

std::ostream& operator<< (std::ostream& os, const Grid& g) {
    for(int i = 0; i < g.rows; ++i) {
        for(int j = 0; j < g.cols; ++j) {
            os << (g.cells[i][j] ? "1" : "0");
        }
        os << std::endl;
    }
    return os;
}

// calcFuture counts the number of neighbors of a given cell in the current grid and updates the future grid accordingly.
void calcFuture(Grid* current, Grid* future, int row, int col) {
    char numNeighbors = 0; // Using char for 1 byte storage
    for(int i = row-1; i <= row + 1; ++i) {
        if(i < 0 || i >= current->rows) {
            continue;
        }
        for(int j = col-1; j <= col+1; ++j) {
            if(j < 0 || j >= current->cols || (i == row && j == col)) {
                continue;
            }

            if(current->cells[i][j]) {
                ++numNeighbors;
            }
        }
    }
    
    switch(numNeighbors) {
        case 2:
            future->cells[row][col] = current->cells[row][col];
            break;
        case 3:
            future->cells[row][col] = true;
            break;
        default:
            // Default case covers 0, 1, and 3+ neighbors
            future->cells[row][col] = false;
            break;
    }
}

bool isSame; // Used as a global variable shared across all threads to end execution early if none of the cells are different from generation to generation

// threadTask calculates the future of the given cellsToCompute starting from startRow, startCol.  
// Is safe from isues since threads will not be modifying the same memory, as each cell is a different offset within future->cells
void threadTask(Grid* current, Grid* future, int cellsToCompute, int startRow, int startCol) {
    int row = startRow;
    int col = startCol;
    for(int i = 0; i < cellsToCompute; ++i) {
        calcFuture(current, future, row, col);
        if(current->cells[row][col] != future->cells[row][col]) {
            isSame = false;
        }
        col = (col + 1) % current->cols;
        row = row + (col == 0 ? 1 : 0);
    }
}

// main is the driver, parsing the inputfile, invoking the threads, and dumping to the outputfile with all the required error checking
int main(int argc, char** argv) {
    if(argc < 5) {
        std::cout << "Proper Usage: %s <inputfile> <outputfile> <numsteps> <numthreads>" << std::endl;
        exit(1);
    }

    std::ifstream is(argv[1]);
    if (is.fail()) {
        std::cout << "Unable to open input file.  Exiting " << std::endl;
        exit(1);
    }

    int rows = 0;
    int cols = 0;
    std::string line;
    while(is.good()) {
        std::getline(is, line);
        if(line.length() == 0) {
            continue;
        }
        if(cols != 0) {
            // If not first loop and line length is different, file is invalid
            if (line.length() != cols) {
                std::cout << "Invalid file [COLUMN_LENGTH].  Exiting" << std::endl;
                exit(1);
            }
        }
        cols = line.length();
        ++rows;
    }

    is.clear();
    is.seekg(0, is.beg);

    // Making the grids pointers so we can use shadow paging as an optimization with double buffering, to prevent copying everything back every update
    Grid* current = new Grid(rows, cols);
    Grid* future = new Grid(rows, cols);

    {
        int row = 0;
        while(is.good()) {
            int col = 0;
            std::getline(is, line);
            if(line.length() == 0) {
                continue; // Extra returns in file are fine
            }   
            for(char c : line) {
                if (c == '1') {
                    current->cells[row][col] = true;
                } else if (c == '0') {
                    current->cells[row][col] = false;
                } else {
                    // If chars in file are not 0 or 1, file is invalid.
                    std::cout << "Invalid File [UNKNOWN_CHAR].  Exiting" << std::endl;
                    exit(1);
                }
                ++col;
            }
            ++row;
        }
    }

    is.close();

    int steps = atoi(argv[3]);
    if (steps < 0) {
        std::cout << "Nonsensical number of steps [INVALID_ARGUMENT].  Exiting" << std::endl;
        exit(1);
    } else if (steps == 0 && (argv[3][0] != '0' || strlen(argv[3]) > 1)) {
        std::cout << "Steps invalid [INVALID_ARGUMENT].  Exiting" << std::endl;
        exit(1);
    }

    int numThreads = atoi(argv[4]);
    if (numThreads <= 0) {
        std::cout << "Nonsensical number of threads or bad thread input [INVALID_ARGUMENT].  Exiting" << std::endl;
        exit(1);
    }

    for(int i = 0; i < steps; ++i) {
        isSame = true;
        
        // Update loop
        std::vector<std::thread*> threads;
        int numCells = current->rows * current->cols;

        for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
            int cellsToCompute = (numCells / numThreads) + ((threadNum < numCells % numThreads) ? 1 : 0);
            int startCell = threadNum * (numCells / numThreads) + std::min(threadNum, numCells % numThreads);
            int startRow = startCell / current->cols;
            int startCol = startCell % current->cols;
            threads.push_back(new std::thread(threadTask, current, future, cellsToCompute, startRow, startCol));
        }

        for (int j = 0; j < numThreads; ++j) {
            (*threads[j]).join();
            delete threads[j];
        }

        threads.resize(0);

        // Short circuit evaluation computed via the parallel threads synced thanks to join
        if(isSame) {
            break;
        }

        void* temp = future;
        future = current;
        current = (Grid*) temp;
    }

    std::ofstream os(argv[2]);
    if (os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
        exit(1);
    }
    os << *current;
}