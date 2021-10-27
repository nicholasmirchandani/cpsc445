#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

using namespace std;

void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

char returnNucleotide(int num) {
    switch(num) {
        case 0:
            return 'A';
        case 1:
            return 'C';
        case 2:
            return 'G';
        case 3:
            return 'T';
        default:
            return '?';
    }
}
#define MAX_BUF 100000

int main (int argc, char *argv[]) {
  int rank;
  int p;

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  // example code
  char n[MAX_BUF];
  char n_final[MAX_BUF];
  char* recv_buf = new char[MAX_BUF / p];
  int numCharsToSend = 0;
  if(rank == 0) {
      std::ifstream is("dna.txt");
      if (is.fail()) {
        std::cout << "Unable to open dna file.  Exiting " << std::endl;
        exit(1);
      }

      std::string allChars = "";
      std::string line;
      while(is.good()) {
        std::getline(is, line);

        // Remove trailing newline and carriage returns
        while(line[line.length()-1] == '\n' || line[line.length()-1] == '\r') {
            line.pop_back();
        }

        allChars.append(line);
      }

      is.close();

      strcpy(n, allChars.c_str());
      for(int i = strlen(n); i < MAX_BUF; ++i) {
      // Ensures 0s are padded after string as needed
        n[i] = '\0';
      }
      numCharsToSend = (allChars.length() / p) + ((strlen(n) % p) == 0 ? 0 : 1);
      // Ensure that numCharsToSend is divisible by 3
      while (numCharsToSend % 3 != 0) {
          ++numCharsToSend;
      }
      for(int i = 1; i < p; ++i) {
        MPI_Send(&numCharsToSend, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
  } else {
    MPI_Status status;
    MPI_Recv(&numCharsToSend, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }
  // Scatter chunks of the string
  check_error(MPI_Scatter(n, numCharsToSend, MPI_CHAR, recv_buf, numCharsToSend, MPI_CHAR, 0, MPI_COMM_WORLD));


  // Count every trigram
  const int A_KEY = 0;
  const int C_KEY = 1;
  const int G_KEY = 2;
  const int T_KEY = 3;
  int counts[4 * 4 * 4]; // Using counts[(first_key) + 4 * (second_key) + 16 * (third_key)]
  for(int i = 0; i < 64; ++i) {
      counts[i] = 0;
  }
  int counts_final[4 * 4 * 4];

  int curKeys[3];
  bool stopNow = false;
  for(int i = 0; i < numCharsToSend; i += 3) {
    int j;
    for(j = i; j < i + 3; ++j) {
        if (stopNow) { // Don't add extra counts if we've hit a null character
            break;
        }
        switch(recv_buf[j]) {
            case 'A':
                curKeys[j] = A_KEY;
                break;
            case 'C':
                curKeys[j] = C_KEY;
                break;
            case 'G':
                curKeys[j] = G_KEY;
                break;
            case 'T':
                curKeys[j] = T_KEY;
                break;
            default:
                stopNow = true;
                break;
        }
    }

    if (stopNow) { // Don't add extra counts if we've hit a null character
        break;
    }

    if (j <= numCharsToSend) {
        int targetIndex = curKeys[i] + 4 * curKeys[i+1] + 16 * curKeys[i+2];
        counts[targetIndex] = counts[targetIndex] + 1;
    }
  }

  std::cout << "DEBUG: first char is " << (uint) recv_buf[0] << "  DEBUG: Count AAA " << counts[0] << std::endl;
  delete(recv_buf);

  // Sum the counts to output them
  check_error(MPI_Reduce(&counts, &counts_final, 64, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));


  // Reduce counts of As, Cs, Ts, and Gs into process of rank 0, independently as 4 separate reduce calls
  if (rank==0) {
    ofstream os("output.txt");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            for(int k = 0; k < 4; ++k) {
                if(counts_final[i + 4 * j + 16 * k] != 0) {
                    os << returnNucleotide(i) << returnNucleotide(j) << returnNucleotide(k) << " " << counts_final[i + 4 * j + 16 * k] << std::endl;
                }
            }
        }
    }

    os.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}