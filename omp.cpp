#include <cstdlib>
#include <time.h>
#include <iostream>
#include <omp.h>
#include "matrix.h"

using namespace std;

void FillMatricesRandomly(Matrix <double> &A, Matrix <double> &B );
void PrintMatrices( Matrix <double> &A, Matrix <double> &B, Matrix <double> &C );

int randomHigh = 100;
int randomLow = 0;

int main(int argc, char *argv[]) {
  cout << "Starting an OpenMP parallel matrix multiplication. \n  " << endl;  

  if ( argv[1]== NULL || argv[2] == NULL)
  {
    cout << "ERROR: The program must be executed in the following way  \n\n  \t \"./omp NumberOfThreads N \"  \n\n where NuberOfThreads and N are integers. \n \n " << endl;
    return 1;
  }
  
  int numThreads = atoi(argv[1]); 
  cout << "The number of OpenMP threads: " << numThreads << endl;
  omp_set_dynamic(0);		// do not allow the number of threads to be set internally
  omp_set_num_threads(numThreads);  //set the number of threads 

  int N = atoi(argv[2]);
  cout << "The matrices are: " << N << "x" << N << endl;

  int numberOfRowsA = N;  
  int numberOfColsA = N;  
  int numberOfRowsB = N;  
  int numberOfColsB = N;  

  Matrix <double> A = Matrix <double>(numberOfRowsA, numberOfColsA); 
  Matrix <double> B = Matrix <double>(numberOfRowsB, numberOfColsB); 
  Matrix <double> C = Matrix <double>(numberOfRowsA, numberOfColsB); 

  FillMatricesRandomly(A, B);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

 
  double sum = 0;
  double val = 0;
  
#pragma omp parallel for private(val) reduction(+:sum)
  for (int i = 0; i < A.rows(); i++) {
    val = omp_get_wtime();
    for (int j = 0; j < B.cols(); j++) {
      for (int k = 0; k < B.rows(); k++) {
        C(i,j) += (A(i,k) * B(k,j));
      }
    }
    sum +=  omp_get_wtime() - val;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double totalMatrixCalculationTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
  
  cout << "Total multplication time = " << totalMatrixCalculationTime << endl;
  cout << "average multplication time = " << sum/numThreads << endl;
  cout << "Approximate Communication time = " << totalMatrixCalculationTime - sum / numThreads << endl;  

  PrintMatrices(A, B, C);

  return 0;
}

void FillMatricesRandomly(Matrix <double> &A, Matrix <double> &B){
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