#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <keywords-file> <text-file> <output-file> <numthreads>" << std::endl;
        exit(1);
    }

    std::vector<std::string> keywords_vec;
    std::ifstream keywords_is(argv[1]);
    if (keywords_is.fail()) {
        std::cout << "Unable to open keywords file.  Exiting " << std::endl;
        exit(1);
    }

    std::string line;
    while(keywords_is.good()) {
        std::getline(keywords_is, line);
        // Remove trailing newline and carriage returns
        while(line[line.length()-1] == '\n' || line[line.length()-1] == '\r') {
            line.pop_back();
        }
        if(line.length() != 0) {
            keywords_vec.push_back(line);
        }
    }
    keywords_is.close();

    // Sorts vector because that's how it is in the code example
    std::sort(keywords_vec.begin(), keywords_vec.end());

    int* keywords_count = new int[keywords_vec.size()];
    for(int i = 0; i < keywords_vec.size(); ++i) {
        keywords_count[i] = 0;
    }


    std::ifstream text_is(argv[2]);
    if (text_is.fail()) {
        std::cout << "Unable to open text file.  Exiting " << std::endl;
        exit(1);
    }

    std::string word = "";
    while(text_is.good()) {
        std::getline(text_is, line);
        for(char c : line) {
            if(c == '.' || c == ' ') {
                // If vector contains word, then increment it
                for(int i = 0; i < keywords_vec.size(); ++i) {
                    // std::cout << "Keyword comparing: " << keywords_vec[i] << " Of length: " << keywords_vec[i].length() << std::endl;
                    if(keywords_vec[i].compare(word) == 0) {
                    //    std::cout << "Matches keyword " << std::endl;
                        ++keywords_count[i];
                    }
                }
                word = "";
            } else {
                word.push_back(c);
            }
        }
    }
    // Also check last word if doesn't end on space
    // If vector contains word, then increment it
    for(int i = 0; i < keywords_vec.size(); ++i) {
        if(keywords_vec[i] == word) {
            ++keywords_count[i];
        }
    }

    text_is.close();



    std::ofstream os(argv[3]);
    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    for(int i = 0; i < keywords_vec.size(); ++i) {
        os << keywords_vec[i] << " " << keywords_count[i] << std::endl;
    }

    os.close();
}