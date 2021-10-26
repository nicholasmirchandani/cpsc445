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

  // TODO: Count number of As, Cs, Ts, and Gs and store them separately

  // TODO: Reduce counts of As, Cs, Ts, and Gs into process of rank 0, independently as 4 separate reduce calls
  int A_Count = 0;
  int C_Count = 0;
  int T_Count = 0;
  int G_Count = 0;
  int A_Count_loc = 0;
  int C_Count_loc = 0;
  int T_Count_loc = 0;
  int G_Count_loc = 0;

  for(int i = 0; i < numCharsToSend; ++i) { // NOTE: Assuming 10 is number of chars
    switch(recv_buf[i]) {
      case 'A':
        ++A_Count_loc;
        break;
      case 'C':
        ++C_Count_loc;
        break;
      case 'T':
        ++T_Count_loc;
        break;
      case 'G':
        ++G_Count_loc;
        break;
      default:
        std::cout << "Unknown character detected. Character is ' " << recv_buf[i] << " ' Skipping" << std::endl;
    }
  }

  delete(recv_buf);



  check_error(MPI_Reduce(&A_Count_loc, &A_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  check_error(MPI_Reduce(&C_Count_loc, &C_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  check_error(MPI_Reduce(&T_Count_loc, &T_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  check_error(MPI_Reduce(&G_Count_loc, &G_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  if (rank==0) {
    ofstream os("output.txt");

    if(os.fail()) {
        std::cout << "Unable to open output file.  Exiting " << std::endl;
    }

    os << "A " << A_Count << std::endl;
    os << "C " << C_Count << std::endl;
    os << "T " << T_Count << std::endl;
    os << "G " << G_Count << std::endl;

    os.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}