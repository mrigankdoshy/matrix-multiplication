#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include "matrix.h"

using namespace std;

void FillMatricesRandomly(Matrix <double> &A, Matrix <double> &B );
void PrintMatrices( Matrix <double> &A, Matrix <double> &B, Matrix <double> &C );

int randomHigh = 100; 		
int randomLow = 0;

int main(int argc, char *argv[]) {
  cout << "Starting a serial matrix multiplication. \n  " << endl;  

  if ( argv[1]== NULL )
  { 
    cout << "ERROR: The program must be executed in the following way  \n\n  \t \"./a N \"  \n\n where N is an integer. \n \n " << endl;
    return 1;
  }

  int N = atoi(argv[1]);
  cout << "The matrices are: " << N <<"x"<< N << endl;

  int numberOfRowsA = N;  
  int numberOfColsA = N;  
  int numberOfRowsB = N;  
  int numberOfColsB = N;  


  Matrix <double> A = Matrix <double> (numberOfRowsA, numberOfColsA); 
  Matrix <double> B = Matrix <double> (numberOfRowsB, numberOfColsB); 
  Matrix <double> C = Matrix <double> (numberOfRowsA, numberOfColsB); 

  FillMatricesRandomly(A, B);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  
  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < B.cols(); j++) {
      for (int k = 0; k < B.rows(); k++) {
	       C(i,j) += (A(i,k) * B(k,j));
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double matrixCalculationTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

  cout << "\nTotal multplication time = " << matrixCalculationTime << endl;
  
  PrintMatrices(A, B, C); 

  return 0;
}

void FillMatricesRandomly(Matrix <double> &A, Matrix <double> &B) {  
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