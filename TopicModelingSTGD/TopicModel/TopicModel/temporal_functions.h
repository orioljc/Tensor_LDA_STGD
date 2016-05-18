//
//  time_functions.h
//  CommunityDetection
/*******************************************************
* Copyright (C) 2016 {Oriol Julia Carrillo} <{email}>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#include "stdafx.h"
int * read_vector_ints(const char *, int );
void tensorDecom_alpha0_online( MatrixXd, SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new, int t);
MatrixXd Diff_Loss(MatrixXd emp_T3_whitened, SparseMatrix<double> D_a_mat, VectorXd D_a_mu, VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, MatrixXd phi, double learningrate_phi, int datapoint);
VectorXd The_second_term(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, VectorXd phi);
double compute_loss(MatrixXd, MatrixXd);
double compute_loss_point(int datapoint, MatrixXd phi, MatrixXd emp_T3_whitened_x);
MatrixXd compute_empirical_T3(SparseMatrix<double>, VectorXd);
MatrixXd compute_empirical_T3_point(SparseMatrix<double> whitened_data, VectorXd y_mean, int datapoint);
MatrixXd create_tensor_phi( VectorXd eigval, MatrixXd phi, int size);
VectorXd compute_M1_topic( SparseMatrix<double> data );
SparseMatrix<double> compute_M2_topic( SparseMatrix<double> data );
MatrixXd compute_w( SparseMatrix<double> M2, int KHID );
MatrixXd calculate_derivatives_oriol( MatrixXd phi, MatrixXd empirical_T3_x );
MatrixXd compute_derivatives(MatrixXd, MatrixXd);
void tensor_STGD( MatrixXd tensor, SparseMatrix<double> evec, VectorXd evec_mean, VectorXd &lambda, MatrixXd & evec_new, VectorXd Lengths, int t);
MatrixXd compute_mean_term_tensor_x(VectorXd whitenedMean);
MatrixXd compute_empirical_derivatives(MatrixXd tensor_x, MatrixXd evec, int datapoint);