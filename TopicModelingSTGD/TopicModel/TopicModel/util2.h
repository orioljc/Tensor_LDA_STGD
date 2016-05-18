//
//  util2.h
//
/*******************************************************
* Copyright (C) 2016 {Oriol Julia Carrillo} <{email}>
*
* All rights reserved.
*******************************************************/
#include "stdafx.h"
int * read_vector_ints(const char *, int );
double compute_loss(MatrixXd, MatrixXd);
double compute_loss_point(int datapoint, MatrixXd phi, MatrixXd emp_T3_whitened_x);
MatrixXd compute_empirical_T3(SparseMatrix<double>, VectorXd);
MatrixXd create_tensor_phi( VectorXd eigval, MatrixXd phi, int size);
VectorXd compute_M1_topic( SparseMatrix<double> data );
SparseMatrix<double> compute_M2_topic( SparseMatrix<double> data );
MatrixXd compute_w( SparseMatrix<double> M2, int KHID );
MatrixXd compute_derivatives(MatrixXd, MatrixXd);
void tensor_STGD( MatrixXd tensor, SparseMatrix<double> evec, VectorXd evec_mean, VectorXd &lambda, MatrixXd & evec_new, VectorXd Lengths, int t);
MatrixXd compute_mean_term_tensor_x(VectorXd whitenedMean);
MatrixXd compute_empirical_derivatives(MatrixXd tensor_x, MatrixXd evec, int datapoint);