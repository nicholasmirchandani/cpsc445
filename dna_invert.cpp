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
  char* recv_buf = new char[MAX_BUF / p];
  int numCharsToSend;
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
      numCharsToSend = allChars.length();
  }
  // TODO: Scatter chunks of the string instead of an int
  check_error(MPI_Scatter(n, numCharsToSend, MPI_CHAR, recv_buf, numCharsToSend, MPI_CHAR, 0, MPI_COMM_WORLD));

  // TODO: Flip everything within the recv_buf

  delete(recv_buf);

  // TODO: Gather everything from recv_bufs and output

  if (rank==0) {
    ofstream os("output.txt");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    os << "Some Output" << std::endl;

    os.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}