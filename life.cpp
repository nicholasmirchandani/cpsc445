#include <iostream>
#include <fstream>
#include <thread>
#include "math.h"
#include <vector>

class Cell {
    public:
        bool isAlive;
        short numNeighbors;
        Cell();
};

Cell::Cell() {
    isAlive = false;
    numNeighbors = -1;  // Using negative numbers to indicate not yet calculated
}

class Grid {
    public:
        Cell** cells;
        int rows;
        int cols;
        Grid(int rows, int cols);
        ~Grid();
};

Grid::Grid(int rows, int cols) {
    // Design: cells[row][col] = appropriate cell
    this->rows = rows;
    this->cols = cols;
    cells = new Cell*[rows];
    for(int i = 0; i < rows; ++i) {
        cells[i] = new Cell[cols];
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
            os << (g.cells[i][j].isAlive ? "1" : "0");
        }
        os << std::endl;
    }
    return os;
}

void calcNeighbors(Grid* g, int row, int col) {
    short numNeighbors = 0;
    for(int i = row-1; i <= row + 1; ++i) {
        if(i < 0 || i >= g->rows) {
            continue;
        }
        for(int j = col-1; j <= col+1; ++j) {
            if(j < 0 || j >= g->cols || (i == row && j == col)) {
                continue;
            }

            if(g->cells[i][j].isAlive) {
                ++numNeighbors;
            }
        }
    }
    g->cells[row][col].numNeighbors = numNeighbors;
}

void calcFuture(Grid* current, Grid* future, int row, int col) {
    switch(current->cells[row][col].numNeighbors) {
        case -1:
            // Definitely not needed in final code, but will have for now just in case I do something dumb
            std::cout << "ERROR: Uninitialized neighbor" << std::endl;
            break;
        case 2:
            future->cells[row][col].isAlive = current->cells[row][col].isAlive;
            break;
        case 3:
            future->cells[row][col].isAlive = true;
            break;
        default:
            // Default case covers 0, 1, and 3+ neighbors
            future->cells[row][col].isAlive = false;
            break;
    }
}

bool isSame;

void cellTask(Grid* current, Grid* future, int cellsToCompute, int startRow, int startCol) {
    int row = startRow;
    int col = startCol;
    for(int i = 0; i < cellsToCompute; ++i) {
        calcNeighbors(current, row, col);
        calcFuture(current, future, row, col);
        if(current->cells[row][col].isAlive != future->cells[row][col].isAlive) {
            isSame = false;
        }
        col = (col + 1) % current->cols;
        row = row + (col == 0 ? 1 : 0);
    }
}


int main(int argc, char** argv) {
    if(argc < 5) {
        std::cout << "Proper Usage: %s <inputfile> <outputfile> <numsteps> <numthreads>" << std::endl;
        exit(1);
    }

    std::ifstream is(argv[1]);
    int rows = 0;
    int cols = 0;
    std::string line;
    while(is.good()) {
        std::getline(is, line);
        ++rows;
    }

    cols = line.length();

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
            for(char c : line) {
                if (c == '1') {
                    current->cells[row][col].isAlive = true;
                } else {
                    current->cells[row][col].isAlive = false;
                }
                ++col;
            }
            ++row;
        }
    }

    is.close();

    int steps = atoi(argv[3]);
    int numThreads = atoi(argv[4]);

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
            threads.push_back(new std::thread(cellTask, current, future, cellsToCompute, startRow, startCol));
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
    os << *current;
}
