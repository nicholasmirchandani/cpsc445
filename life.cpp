#include <iostream>

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

bool operator== (const Cell& c1, const Cell& c2) {
    // TODO: Implement the comparison to tell if the cells are the same, used in comparing the grids
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
    // Design: cells[col][row] = appropriate cell
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

bool operator== (const Grid& g1, const Grid& g2) {
    // TODO: Implement the comparison to tell if the grids are the same, used for short circuit evaluation of GoL for efficiency
}

int main() {
    std::cout << "Hello World!" << std::endl;
}
