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

#define MAX_BUF 10000

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
      for(int i = 1; i < p; ++i) {
        MPI_Send(&numCharsToSend, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
  } else {
    MPI_Status status;
    MPI_Recv(&numCharsToSend, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }
  // Scatter chunks of the string
  check_error(MPI_Scatter(n, numCharsToSend, MPI_CHAR, recv_buf, numCharsToSend, MPI_CHAR, 0, MPI_COMM_WORLD));

  // Flip everything within the recv_buf
  for(int i = 0; i < numCharsToSend; ++i) {
    if(recv_buf[i] == '\0') {
      // Break out of loop on null terminating character, since we don't know the length of the string
      break;
    }
    switch(recv_buf[i]) {
      case 'A':
        recv_buf[i] = 'T';
        break;
      case 'C':
        recv_buf[i] = 'G';
        break;
      case 'T':
        recv_buf[i] = 'A';
        break;
      case 'G':
        recv_buf[i] = 'C';
        break;
      default:
        std::cout << "Unknown character detected. Character is ' " << recv_buf[i] << " ' Skipping" << std::endl;
        break;
    }
  }

  // Gather everything within the recv bufs and output
  check_error(MPI_Gather(recv_buf, numCharsToSend, MPI_CHAR, n_final, numCharsToSend, MPI_CHAR, 0, MPI_COMM_WORLD));

  delete[] (recv_buf);

  // Reduce counts of As, Cs, Ts, and Gs into process of rank 0, independently as 4 separate reduce calls
  if (rank==0) {
    ofstream os("output.txt");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    os << n_final;

    os.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}