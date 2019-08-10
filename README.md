# Matrix Multiplication

The multiplication of two matrices is to be implemented as:
1. A sequential program
2. An OpenMP shared memory program
3. A Message Passing Program using the MPI Standard

This repository will serve as a comparison of serial, OpenMP parallel and MPI parallel code that accomplishes the same
task: matrix multiplication. Along with comparing the total matrix multiplication times of the codes, we will look at the
ratio of time spent calculating the multiplication to the time the parallel tool spends communicating data.
