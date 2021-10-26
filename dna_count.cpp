#include <mpi.h>
#include <iostream>

using namespace std;

void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

#define MAX_BUF 1000

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
  int numCharsToSend = 100;
  if(rank == 0) {
      for(int i = 0; i < numCharsToSend; ++i) {
        if(i % 4 == 0) {
          n[i] = 'A';
        } else if (i % 4 == 1) {
          n[i] = 'C';
        } else if (i % 4 == 2) {
          n[i] = 'T';
        } else {
          n[i] = 'G';
        }
      }
  }
  // TODO: Scatter chunks of the string instead of an int
  check_error(MPI_Scatter(n, numCharsToSend, MPI_CHAR, recv_buf, numCharsToSend, MPI_CHAR, 0, MPI_COMM_WORLD));

    std::cout << "Thread rank: " << rank << " with recv buffer of " << recv_buf[0] << std::endl;

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
        std::cerr << "Unknown character detected.  Skipping" << std::endl;
    }
  }

  delete(recv_buf);



  check_error(MPI_Reduce(&A_Count_loc, &A_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  check_error(MPI_Reduce(&C_Count_loc, &C_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  check_error(MPI_Reduce(&T_Count_loc, &T_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  check_error(MPI_Reduce(&G_Count_loc, &G_Count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  if (rank==0) {
    // TODO: Check!
    std::cout << "Counts: " << std::endl;
    std::cout << "A_Count: " << A_Count << std::endl;
    std::cout << "C_Count: " << C_Count << std::endl;
    std::cout << "T_Count: " << T_Count << std::endl;
    std::cout << "G_Count: " << G_Count << std::endl;
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}