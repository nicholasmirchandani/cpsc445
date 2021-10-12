#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <mutex>

std::vector<std::string> allLines; // Storing all lines in global buffer to be read by all threads
std::vector<std::string> keywords_vec;
std::mutex counts_mutex;

void threadTask(int threadNum, int numThreads, int* keywords_count) {
    std::string line;
    std::string word = "";
    // Line k needs to be processed by k % p, where p is the total number of threads.  So 6 lines, 3 threads, would be allocated 0 1 2 0 1 2, which is accomplished by this for loop
    for(int i = threadNum; i < allLines.size(); i += numThreads) {
        line = allLines[i];
        for(char c : line) {
            if(c == '.' || c == ' ' || c == '\r' || c == '\n') {
                // If vector contains word, then increment it
                for(int i = 0; i < keywords_vec.size(); ++i) {
                    if(keywords_vec[i].compare(word) == 0) {
                        counts_mutex.lock();
                        ++keywords_count[i];
                        counts_mutex.unlock();
                    }
                }
                word = "";
            } else {
                word.push_back(c);
            }
        }

        // Also check last word if doesn't end on space, assuming newline isn't present
        // If vector contains word, then increment it
        for(int i = 0; i < keywords_vec.size(); ++i) {
            if(keywords_vec[i].compare(word) == 0) {
                counts_mutex.lock();
                ++keywords_count[i];
                counts_mutex.unlock();
                word = ""; // ASSUMPTION: Keyword is not repeated
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <keywords-file> <text-file> <output-file> <numthreads>" << std::endl;
        exit(1);
    }

    std::ifstream keywords_is(argv[1]);
    if (keywords_is.fail()) {
        std::cout << "Unable to open keywords file.  Exiting " << std::endl;
        exit(1);
    }

    std::string line;

    keywords_vec.resize(0); // Clearing buffer just in case

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

    allLines.resize(0); // Clearing buffer, just in case
    while(text_is.good()) {
        std::getline(text_is, line);
        if(line.length() == 1) { // If the line is just the newline, ignore it
            continue;
        }
        allLines.push_back(line);
    }

    text_is.clear();
    text_is.seekg(0, text_is.beg);

    int numThreads = atoi(argv[4]);

    // Update loop
    std::vector<std::thread*> threads;
    for (int threadNum = 0; threadNum < numThreads; ++threadNum) {
        threads.push_back(new std::thread(threadTask, threadNum, numThreads, keywords_count));
    }

    for (int j = 0; j < numThreads; ++j) {
        (*threads[j]).join();
        delete threads[j];
    }


    /*std::string word = "";
    while(text_is.good()) {
        std::getline(text_is, line);
        for(char c : line) {
            if(c == '.' || c == ' ' || c == '\r' || c == '\n') {
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

        // Also check last word if doesn't end on space, assuming newline isn't present
        // If vector contains word, then increment it
        for(int i = 0; i < keywords_vec.size(); ++i) {
            if(keywords_vec[i].compare(word) == 0) {
                ++keywords_count[i];
                word = ""; // ASSUMPTION: Keyword is not repeated
            }
        }
    }*/

    text_is.close(); // TODO: Move this up



    std::ofstream os(argv[3]);
    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    for(int i = 0; i < keywords_vec.size(); ++i) {
        os << keywords_vec[i] << " " << keywords_count[i] << std::endl;
    }

    os.close();
}