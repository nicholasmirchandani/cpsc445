#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <keywords-file> <text-file> <output-file>" << std::endl;
        exit(1);
    }

    std::cout << "Keywords file: " << argv[1] << std::endl;
    std::cout << "Text file: " << argv[2] << std::endl;
    std::cout << "Output file: " << argv[3] << std::endl;
}