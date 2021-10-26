#include <mpi.h>
#include <iostream>

using namespace std;

void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

int main (int argc, char *argv[]) {
  int rank;
  int p;

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  // example code
  int n[5];
  int recv_buf[1];
  if(rank == 0) {
      n[0] = 1;
      n[1] = 2;
      n[2] = 3;
      n[3] = 4;
      n[4] = 5;
  }
  // TODO: Scatter chunks of the string instead of an int
  check_error(MPI_Scatter(n, 1, MPI_INT, recv_buf, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::cout << "Thread rank: " << rank << " with recv buffer of " << recv_buf[0] << std::endl;

  // TODO: Count number of As, Cs, Ts, and Gs and store them separately

  // TODO: Reduce counts of As, Cs, Ts, and Gs into process of rank 0, independently as 4 separate reduce calls
  int sum = 0;
  check_error(MPI_Reduce(recv_buf, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  if (rank==0) {
    if (sum != (1+2+3+4+5)) { cerr << "error!  Sum is \n" << sum << std::endl; exit(1); }
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}