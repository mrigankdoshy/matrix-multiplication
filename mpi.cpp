#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <algorithm>
#include "matrix.h" 	       

using namespace std;

void FillMatricesRandomly(Matrix<double> &A, Matrix<double> &B ); 
void PrintMatrices( Matrix<double> &A, Matrix<double> &B, Matrix<double> &C );

#define ROW_START_TAG 0    //tag for communicating the start row of the workload for a slave
#define ROW_END_TAG 1      //tag for communicating the end row of the workload for a slave
#define A_ROWS_TAG 2       //tag for communicating the address of the data to be worked on to slave
#define C_ROWS_TAG 3       //tag for communicating the address of the calculated data to master
#define LOCAL_TIME_TAG 4   //tag for communicating the address of the local matrix calculation time to master

// Instantiate global variables used in the parallelization
int rank;                  // mpi: process id number
int nProcesses;            // mpi: number of total processess 
MPI_Status status;         // mpi: store status of a MPI_Recv
MPI_Request request;       // mpi: capture request of a MPI_Isend
int rowStart, rowEnd;      // which rows of A that are calculated by the slave process
int granularity; 	   // granularity of parallelization (# of rows per processor) 

double start_time, end_time;
double localTimeSaver;

int randomHigh = 100;    
int randomLow = 0;

int main(int argc, char *argv[]) {
  if ( argv[1]== NULL ) { 
   cout << "ERROR: The program must be executed in the following way  \n\n  \t \"mpirun -n NumberOfProcesses mpi N \"  \n\n where N is an integer. \n \n " << endl;
   return 1;
  }

  int N = atoi(argv[1]);

  int numberOfRowsA = N;  
  int numberOfColsA = N;  
  int numberOfRowsB = N;  
  int numberOfColsB = N;  
  //Declare matrices: Matix class is a 2D vetor
  Matrix <double> A = Matrix <double>(numberOfRowsA, numberOfColsA); 
  Matrix <double> B = Matrix <double>(numberOfRowsB, numberOfColsB); 
  Matrix <double> C = Matrix <double>(numberOfRowsA, numberOfColsB); 

  MPI_Init(&argc, &argv);		                  /* initialize MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);	      /* store the rank */
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses); /* store the number of processes */

  if (rank == 0) {  /* Master initializes work*/
    cout << "Starting an MPI parallel matrix multiplication. \n  " << endl;
    cout << "The matrices are: " << N << "x" << N << endl;    
    FillMatricesRandomly( A, B); 	/* Excluded from timing!!!  */    

    /* Begin Timing: used for total multiplication time: communication + calculations  */    
    start_time = MPI_Wtime(); 	
    for (int i = 1; i < nProcesses; i++) { /* for each slave */
      // calculate granularity (-1 comes from excluding the master process)      
      granularity = (numberOfRowsA / (nProcesses - 1)); 
      rowStart = (i - 1) * granularity;

      if (((i + 1) == nProcesses) && ((numberOfRowsA % (nProcesses - 1)) != 0)) { //if rows of [A] cannot be equally divided among slaves
        rowEnd = numberOfRowsA; //last slave gets all the remaining rows
      } else {
        rowEnd = rowStart + granularity; //rows of [A] are equally divisable among slaves
      }

      //send the low bound, without blocking, to the intended slave 
      MPI_Isend(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &request);
      //next send the upper bound without blocking, to the intended slave
      MPI_Isend(&rowEnd, 1, MPI_INT, i ,ROW_START_TAG, MPI_COMM_WORLD, &request);
      //finally send the allocated row granularity of [A] without blocking, to the intended slave
      MPI_Isend(&A(rowStart,0), (rowEnd - rowStart) * numberOfColsA, MPI_DOUBLE, i, A_ROWS_TAG, MPI_COMM_WORLD, &request);  
    }
  }
  //broadcast B (MPI_Bcast: Broadcasts a message from the process with rank "root" to all other processes of the communicator)
  MPI_Bcast(&B(0,0), numberOfRowsB*numberOfColsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank > 0) {   /* work done by slaves (not rank = 0)*/
    //receive low bound from the master
    MPI_Recv(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &status);
    //next receive upper bound from the master
    MPI_Recv(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &status);
    //finally receive row granularity of [A] to be processed from the master
    MPI_Recv(&A(rowStart,0), (rowEnd - rowStart) * numberOfColsA, MPI_DOUBLE, 0, A_ROWS_TAG, MPI_COMM_WORLD, &status);

    // start time for local time: the amount of time to do matrix calculation for this process
    localTimeSaver = MPI_Wtime(); 
    
    for (int i = rowStart; i < rowEnd; i++) {//the given set of rows of A (parallelized loop)
      for (int j = 0; j < B.cols(); j++) {//iterate through columns of [B]
        for (int k = 0; k < B.rows(); k++) {//iterate through rows of [B]
	         C(i,j) += (A(i,k) * B(k,j));
        }
     }
   }
    // calculate local time: the amount of time to do matrix calculation for this process
    localTimeSaver = MPI_Wtime() - localTimeSaver;
    
    //send back the low bound first without blocking, to the master
    MPI_Isend(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &request);
    //send the upper bound next without blocking, to the master
    MPI_Isend(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &request);
    //finally send the processed granularity of data without blocking, to the master
    MPI_Isend(&C(rowStart,0), (rowEnd - rowStart) * numberOfColsB, MPI_DOUBLE, 0, C_ROWS_TAG, MPI_COMM_WORLD, &request);
    //send back the local calculation time without blocking, to the master
    MPI_Isend(&localTimeSaver, 1, MPI_DOUBLE, 0, LOCAL_TIME_TAG, MPI_COMM_WORLD, &request);
    

  }

  if (rank == 0) { /* master gathers processed work*/
    for (int i = 1; i < nProcesses; i++) {// untill all slaves have handed back the processed data
      //receive low bound from a slave
      MPI_Recv(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &status);
      //receive upper bound from a slave
      MPI_Recv(&rowEnd, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &status);
      // //receive processed data from a slave
      MPI_Recv(&C(rowStart,0), (rowEnd - rowStart) * numberOfColsB, MPI_DOUBLE, i, C_ROWS_TAG, MPI_COMM_WORLD, &status);     
     
    }
    end_time = MPI_Wtime(); //end time of the total matrix matrix multiplication
    double totalMultiplicationTime = end_time - start_time;

    // find the longest local calulation (which we take as the total amount of calculation time, the rest comming from communication. 
    std::vector<double> LocalMultiplicationTimes = std::vector<double>(nProcesses);

    for (int i = 1; i < nProcesses; i++) {
      MPI_Recv(&LocalMultiplicationTimes[i], 1, MPI_DOUBLE, i, LOCAL_TIME_TAG, MPI_COMM_WORLD, &status); 
    }    
    double maxLocalMultiplicationTime = *std::max_element(LocalMultiplicationTimes.begin(), LocalMultiplicationTimes.end());

    // print out the results
    cout <<"Total multiplication time =  " << totalMultiplicationTime <<"\n"<< endl;       
    cout <<"Longest multiplication time =  " << maxLocalMultiplicationTime  <<"\n"<< endl;
    cout <<"Approximate communication time =  " << totalMultiplicationTime -maxLocalMultiplicationTime  <<"\n\n"<< endl;

    PrintMatrices(A, B, C);
    
  }
  MPI_Finalize(); //finalize MPI operations
  return 0;
}

void FillMatricesRandomly(Matrix <double> &A, Matrix <double> &B){
/* initialize the random number generator with the current time */  
  srand(time(NULL));		
  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < A.cols(); j++) {
      A(i,j) = rand() % (randomHigh - randomLow) + randomLow;      
    }
  }
  for (int i = 0; i < B.rows(); i++) {
    for (int j = 0; j < B.cols(); j++) {
      B(i,j) = rand() % (randomHigh - randomLow) + randomLow;      
    }
  }
}

void PrintMatrices(Matrix <double> &A, Matrix <double> &B, Matrix <double> &C){
  cout << "\n\nMatrix A" << endl;
  for (int i = 0; i < A.rows(); i++) {
    cout << endl << endl;
    for (int j = 0; j < A.cols(); j++)
      cout << A(i,j) << " ";
  }
  
  cout << "\n\n\n\nMatrix B" << endl;  
  
  for (int i = 0; i < B.rows(); i++) {
    cout << "\n" << endl;
    for (int j = 0; j < B.cols(); j++)
      cout << B(i,j) << " ";
  }
  
  cout << "\n\n\n\nMultiplied Matrix C" << endl;  
  
  for (int i = 0; i < C.rows(); i++) {
    cout << "\n" << endl;  
    for (int j = 0; j < C.cols(); j++)
      cout << C(i,j) << " ";
  }
  
  cout << endl << endl << endl;    
}   