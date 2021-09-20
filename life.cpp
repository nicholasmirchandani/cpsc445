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
    numNeighbors = -1;  // Using negative numbers to indicate not yetcalculated
}

// TODO: Is this required?  I don't think so, since != is all that's required when comparing grids.
bool operator== (const Cell& c1, const Cell& c2) {
    return c1.isAlive == c2.isAlive;
}

bool operator!= (const Cell& c1, const Cell& c2) {
    // Whether or not the cells are alive is what matters to us when comparing them.
    return c1.isAlive != c2.isAlive;
}

class Grid {
    public:
        Cell** cells;
        int cols;
        int rows;
        Grid(int cols, int rows);
        ~Grid();
};

Grid::Grid(int cols, int rows) {
    // Design: cells[col][row] = appropriate cell
    this->cols = cols;
    this->rows = rows;
    cells = new Cell*[cols];
    for(int i = 0; i < cols; ++i) {
        cells[i] = new Cell[rows];
    }
}

Grid::~Grid() {
    for(int i = 0; i < cols; ++i) {
        delete(cells[i]);
    }
    delete(cells);
}

// Checks if the grids are the same, used for short circuit evaluation of GoL for efficiency
bool operator== (const Grid& g1, const Grid& g2) {
    if(g1.cols != g2.cols || g1.rows != g2.rows) {
        // Not necessary in our implementation since all grids will have the same size, but useful as a sanity check
        return false;
    }
    for(int i = 0; i < g1.cols; ++i) {
        for(int j = 0; j < g1.rows; ++j) {
            if(g1.cells[i][j] != g2.cells[i][j]) {
                return false;
            }
        }
    }
    return true;
}

std::ostream& operator<< (std::ostream& os, const Grid& g) {
    for(int i = 0; i < g.cols; ++i) {
        for(int j = 0; j < g.rows; ++j) {
            os << (g.cells[i][j].isAlive ? "1" : "0");
        }
        os << std::endl;
    }
    return os;
}

void calcNeighbors(Grid* g, int col, int row) {
    short numNeighbors = 0;
    for(int i = col-1; i <= col + 1; ++i) {
        if(i < 0 || i >= g->cols) {
            continue;
        }
        for(int j = row-1; j <= row+1; ++j) {
            if(j < 0 || j >= g->rows || (i == col && j == row)) {
                continue;
            }

            if(g->cells[i][j].isAlive) {
                ++numNeighbors;
            }
        }
    }
    g->cells[col][row].numNeighbors = numNeighbors;
}

void calcAllNeighbors(Grid* g) {
    // TODO: Implement calculating neighbors of each cell
    for(int i = 0; i < g->cols; ++i) {
        for(int j = 0; j < g->rows; ++j) {
            calcNeighbors(g, i, j);
        }
    }
}

void calcNeighborsTask(Grid* g, int cellsToCompute, int startCol, int startRow) {
    int col = startCol;
    int row = startRow;
    for(int i = 0; i < cellsToCompute; ++i) {
        std::cout << "Grid Size: " << g->cols << " " << g->rows << std::endl;
        calcNeighbors(g, row, col);
        row = (row + 1) % g->cols;
        col = col + (row == 0 ? 1 : 0);
    }
}

int main(int argc, char** argv) {
    if(argc < 5) {
        std::cout << "Proper Usage: %s <inputfile> <outputfile> <numsteps> <numthreads>" << std::endl;
        exit(1);
    }

    std::ifstream is(argv[1]);
    int cols = 0;
    int rows = 0;
    std::string line;
    while(is.good()) {
        std::getline(is, line);
        ++cols;
    }

    rows = line.length();

    is.clear();
    is.seekg(0, is.beg);

    // Making the grids pointers so we can use shadow paging as an optimization, to prevent copying everything back every update
    Grid* current = new Grid(cols, rows);
    Grid* future = new Grid(cols, rows);

    {
        int col = 0;
        while(is.good()) {
            int row = 0;
            std::getline(is, line);
            for(char c : line) {
                if (c == '1') {
                    current->cells[col][row].isAlive = true;
                } else {
                    current->cells[col][row].isAlive = false;
                }
                ++row;
            }
            ++col;
        }
    }

    is.close();

    int steps = atoi(argv[3]);
    int numThreads = atoi(argv[4]);

    for(int i = 0; i < steps; ++i) {
        // Update loop

        // TODO: Parallelize these!
        std::vector<std::thread*> threads;
        int numCells = current->cols * current->rows;
        for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
            int cellsToCompute = (numCells / numThreads) + ((threadNum < numCells % numThreads) ? 1 : 0);
            std::cout << "Cells per thread: " << cellsToCompute << "\t";
            int startCell = threadNum * (numCells / numThreads) + std::min(threadNum, numCells % numThreads);
            std::cout << "Starting cell num: " << startCell << "\t";
            int startCol = startCell / current->cols;
            int startRow = startCell % current->cols;
            std::cout << "startCol: " << startCol << "\t";
            std::cout << "startRow: " << startRow << std::endl;
            threads.push_back(new std::thread(calcNeighborsTask, current, cellsToCompute, startCol, startRow));
        }

        for (int j = 0; j < numThreads; ++j) {
            (*threads[j]).join();
            delete threads[j];
        }

        threads.resize(0);

        // calcAllNeighbors(current);
        for(int j = 0; j < current->cols; ++j) {
            for(int k = 0; k < current->rows; ++k) {
                // Implementation of the rules in a switch statement
                switch(current->cells[j][k].numNeighbors) {
                    case -1:
                        // Definitely not needed in final code, but will have for now just in case I do something dumb
                        std::cout << "ERROR: Uninitialized neighbor" << std::endl;
                        break;
                    case 2:
                        future->cells[j][k].isAlive = current->cells[j][k].isAlive;
                        break;
                    case 3:
                        future->cells[j][k].isAlive = true;
                        break;
                    default:
                        // Default case covers 0, 1, and 3+ neighbors
                        future->cells[j][k].isAlive = false;
                        break;
                }
            }
        }
        // End TODO for parallelization
        void* temp = future;
        future = current;
        current = (Grid*) temp;
    }

    std::ofstream os(argv[2]);
    os << *current;
}
