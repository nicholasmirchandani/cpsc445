#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <keywords-file> <text-file> <output-file> <numthreads>" << std::endl;
        exit(1);
    }

    std::cout << "Keywords file: " << argv[1] << std::endl;
    std::cout << "Text file: " << argv[2] << std::endl;
    std::cout << "Output file: " << argv[3] << std::endl;
    std::cout << "NumThreads: " << argv[4] << std::endl;

    std::ifstream keywords_is(argv[1]);
    if (keywords_is.fail()) {
        std::cout << "Unable to open keywords file.  Exiting " << std::endl;
        exit(1);
    }

    keywords_is.close();

    std::ifstream text_is(argv[1]);
    if (text_is.fail()) {
        std::cout << "Unable to open text file.  Exiting " << std::endl;
        exit(1);
    }

    text_is.close();

    std::ofstream os(argv[3]);
    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    os << "This is a test output" << std::endl;

    os.close();
}