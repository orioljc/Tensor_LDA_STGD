//============================================================================
// Name        : TopicModel.cpp
// Type        : main
// Created by  : Oriol Julià Carrillo, 2016
// Description : Learn LDA using tensor methods with Stochastic Gradient Descent
//============================================================================

#include "stdafx.h"
#include "temporal.h"
#define _CRT_SECURE_NO_WARNINGS
using namespace Eigen;
using namespace std;
clock_t TIME_start, TIME_end;
int *NX;
int STGD_batch_size;
//int *NX_test;
const char *NX_file;
const char *NX_test_file;
int NT;
int NA;
int KHID;
double alpha0;
int DATATYPE;
int print = false;

int option = 2;
// option = 1 --> tensor decomposition using ALS
// option = 2 --> tensor decomposition using Stochastic Gradient Descent

int main(int argc, const char * argv[])
{
	printf("\n\n");
	int t;
	//Files containing the number of documents for each time
	//NX_file = "";
	//NX_test_file = "";

	//REMOVE TEST
	NX_file = argv[1];
	//printf("%s\n", NX_file);
	//NX_test_file = argv[2];

	// Number of different times
	//NT = furong_atoi(argv[2]);

	NA = furong_atoi(argv[3]);
	NT = furong_atoi(argv[2]);
	KHID = furong_atoi(argv[4]);
	alpha0 = furong_atof(argv[5]);
	DATATYPE = furong_atoi(argv[6]);


	//===============================================================================================================================================================
	// User Manual:
	// (1) Data specs
	// NX is the training sample size
	// NX_test is the test sample size
	// NA is the vocabulary size
	// KHID is the number of topics you want to learn
	// alpha0 is the mixing parameter, usually set to < 1
	// DATATYPE denotes the index convention.
	// -> DATATYPE == 1 assumes MATLAB index which starts from 1,DATATYPE ==0 assumes C++ index which starts from 0 .
	// e.g.  10000 100 500 3 0.01 1
	const char* FILE_GA = argv[7];
	const char* FILE_GA_test = argv[8];

	// (2) Input files
	// $(SolutionDir)\datasets\$(CorpusName)\samples_train.txt
	// $(SolutionDir)\datasets\$(CorpusName)\samples_test.txt
	// e.g. $(SolutionDir)datasets\synthetic\samples_train.txt $(SolutionDir)datasets\synthetic\samples_test.txt
	const char* FILE_alpha_WRITE = argv[9];
	const char* FILE_beta_WRITE = argv[10];
	const char* FILE_hi_WRITE = argv[11];
	// (3) Output files
	// FILE_alpha_WRITE denotes the filename for estimated topic marginal distribution
	// FILE_beta_WRITE denotes the filename for estimated topic-word probability matrix
	// FILE_hi_WRITE denote the estimation of topics per document for the test data.
	// The format is:
	// $(SolutionDir)\datasets\$(CorpusName)\result\alpha.txt
	// $(SolutionDir)\datasets\$(CorpusName)\result\beta.txt
	// $(SolutionDir)\datasets\$(CorpusName)\result\hi.txt
	// e.g. $(SolutionDir)datasets\synthetic\result\alpha.txt $(SolutionDir)datasets\synthetic\result\beta.txt $(SolutionDir)datasets\synthetic\result\hi.txt

	// batch size fot Stochastic Gradient Descent
	STGD_batch_size = furong_atoi(argv[12]);

	//==============================================================================================================================================================
	TIME_start = clock();
	//Read NX file (contains the number of documents at each time)
	NX = read_vector_ints(NX_file,NT);

	//NX_test = read_vector_ints(char *NX_test_file, int NX_test);
	//Initialize temporal array of matrices
	std::vector<SparseMatrix<double>> Gx_a_time;

	//Read bag of words data for each time
	char FILE_GA_temp[200];
	for(t=0; t<NT; t++){
		sprintf(FILE_GA_temp, "%s_%d_%s.txt", FILE_GA, t+1, "train");
		SparseMatrix<double> aux(NX[t], NA);
		Gx_a_time.push_back(aux);	Gx_a_time[t].resize(NX[t], NA);
		Gx_a_time[t].makeCompressed();
		Gx_a_time[t] = read_G_sparse((char *)FILE_GA_temp, "Word Counts Training Data", NX[t], NA);
	}

	TIME_end = clock();
	double time_readfile = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time reading matrices before preproc = %5.10e (Seconds)\n", time_readfile);


	cout << "(1) Whitening--------------------------" << endl;
	TIME_start = clock();

	//Initialize memory for temporal arrays
	std::vector<SparseMatrix<double>> W_time, Whitened_data, Uw_time, diag_Lw_sqrt_inv_s_time;
	std::vector<VectorXd> Lengths_time, Whitened_meanData;

	std::vector<SparseMatrix<double>> M2_time;

	for(t=0; t<NT; t++){

		SparseMatrix<double> temp1(NA, KHID); temp1.resize(NA, KHID); temp1.makeCompressed();
		W_time.push_back(temp1);

		SparseMatrix<double> temp2(NA, KHID);  temp2.resize(NA, KHID); temp2.makeCompressed();
		Uw_time.push_back(temp2);
		SparseMatrix<double> temp3(KHID, KHID); temp3.resize(NA, KHID); temp3.makeCompressed();
		diag_Lw_sqrt_inv_s_time.push_back(temp3);
		VectorXd temp4(NX[t]);
		Lengths_time.push_back(temp4);
		VectorXd mu_a(NA);
		second_whiten_topic(Gx_a_time[t], W_time[t], mu_a, Uw_time[t], diag_Lw_sqrt_inv_s_time[t], Lengths_time[t]);
		//mu_a is a vector of size NA (vocab size) st mu_a(i) = sum_k(proportion of times word(i) appears on each document(k) ) / number of documents
		//Lengths_time contains number of times each word is found in the corpus??

		// whitened datapoints
		// Whitened_data contains the whitened datapoints as its columns
		Whitened_data.push_back( W_time[t].transpose() * Gx_a_time[t].transpose() );	
		Whitened_meanData.push_back( W_time[t].transpose() * mu_a );
	}


	TIME_end = clock();
	double time_whitening = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.10e (Seconds)\n", time_whitening);


	cout << "(1.5) Matricization---------------------" << endl;
	
	//Initialize temporal array of matrices
	std::vector<MatrixXd> T_time, emp_T3_whitened;

	for( t=0; t<NT; t++){
		MatrixXd aux = MatrixXd::Zero(KHID,KHID*KHID);
		T_time.push_back(aux);
		Compute_M3_topic((MatrixXd)Whitened_data[t], Whitened_meanData[t], Lengths_time[t], T_time[t], t);
		//emp_T3_whitened.push_back( compute_empirical_T3(Whitened_data[t], Whitened_meanData[t]) );
	}


	cout << "(2) Tensor decomposition----------------" << endl;
	TIME_start = clock();
	
	std::vector<VectorXd> lambda_time, current_lambda_time;
	std::vector<MatrixXd> phi_new_time, current_phi_time;

	for( t=0; t<NT; t++){
		//The brackets limit the scope of the variable temp
		{	VectorXd temp(KHID);	lambda_time.push_back( temp );	}
		{	MatrixXd temp(KHID, KHID);	phi_new_time.push_back( temp );	}
		{	VectorXd temp(KHID);	current_lambda_time.push_back( temp );	}
		{	MatrixXd temp(KHID, KHID);	current_phi_time.push_back( temp );	}

	    double err_min = 1000;double current_err=1000;
	    int restart_num = 0;
	    int whichRun = 0;

	    if ( option == 1) {
		    printf ("Running ALS: ");
		    while(restart_num<3){
		    	printf (" %d .. ", restart_num);
		        //cout << "Running ALS " << restart_num << endl;
		        current_err = tensorDecom_batchALS(T_time[t],current_lambda_time[t],current_phi_time[t]);
		        if(current_err <err_min){
		        	//replace current eigenvalue and eigenvectors with this run
		            //cout << "replace current eigenvalue and eigenvectors with this run"<< endl;
		            whichRun = restart_num;
		            lambda_time[t] = current_lambda_time[t];
		            phi_new_time[t] = current_phi_time[t];
		            err_min = current_err;
		        }
		        restart_num +=1;
		    }
		    printf("\n");
		    cout << "time " << t << ", FINAL ERROR (" << whichRun << "-th run)" <<" : " << err_min << endl;
	    }else if ( option == 2 ) {
		    printf ("Running Gradient Descent:\n");
			TIME_start = clock();
			tensor_STGD( T_time[t], Whitened_data[t], Whitened_meanData[t], lambda_time[t], phi_new_time[t], Lengths_time[t], t);

			TIME_end = clock();
			double time_stpm = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
			printf("time taken by whitening = %5.25e (Seconds)\n", time_stpm);
	    }else{
	    	printf("Wrong decomposition option\n");
	    }

	    //Print tensors
    	//cout << "T_time[t]:\n" << T_time[0] << endl;
    	//cout << "emp_T3_whitened[t]:\n" << emp_T3_whitened[0] << endl;
    	//cout << "(T_time[t]-emp_T3_whitened[t]).squaredNorm()" << (T_time[t]-emp_T3_whitened[t]).squaredNorm() << endl;
    	MatrixXd tensor = create_tensor_phi( lambda_time[t], phi_new_time[t], KHID );
		//cout << "Reconstructed tensor:\n" << tensor << endl;
	}

	TIME_end = clock();
	double time_stpm = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by ALS = %5.10e (Seconds)\n", time_stpm);	


	cout << "(3) Unwhitening-----------" << endl;
	TIME_start = clock();
	
	std::vector<VectorXd> alpha;
	std::vector<MatrixXd> beta;
	for( t=0; t<NT; t++){
		VectorXd temp1(KHID);
		alpha.push_back( temp1 );
		MatrixXd temp2(NA, KHID);
		beta.push_back( temp2 );

		VectorXd aux1 = VectorXd::Random(KHID);
		MatrixXd aux2 = MatrixXd::Random(KHID, KHID);
		SparseMatrix<double> aux3 = MatrixXd::Random(diag_Lw_sqrt_inv_s_time[t].rows(), diag_Lw_sqrt_inv_s_time[t].cols()).sparseView();
		SparseMatrix<double> aux4 = MatrixXd::Random(Uw_time[t].rows(), Uw_time[t].cols()).sparseView();
		Unwhitening(lambda_time[t], phi_new_time[t], Uw_time[t], diag_Lw_sqrt_inv_s_time[t], alpha[t], beta[t]);
	}

	TIME_end = clock();
	double time_post = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for post processing = %5.10e (Seconds)\n", time_post);


	cout << "(4) Writing results----------" << endl;
	for( t=0; t<NT; t++){
		char FILE_alpha_WRITE_temp[200];
		char FILE_beta_WRITE_temp[200];
		sprintf(FILE_alpha_WRITE_temp, "%s_%d.txt", FILE_alpha_WRITE, t+1);
		sprintf(FILE_beta_WRITE_temp, "%s_%d.txt", FILE_beta_WRITE, t+1);
		write_alpha(FILE_alpha_WRITE_temp, alpha[t]);
		write_beta(FILE_beta_WRITE_temp, beta[t]);	
	}

	/*
	// decode
	cout << "(5) Decoding-----------" << endl;
	TIME_start = clock();

	SparseMatrix<double> Gx_a_test(NX_test, NA); Gx_a_test.resize(NX_test, NA);
	Gx_a_test.makeCompressed();
	Gx_a_test = read_G_sparse((char *)FILE_GA_test, "Word Counts Test Data", NX_test, NA);
	double nx_test = (double)Gx_a_test.rows();
	double na = (double)Gx_a_test.cols();
	VectorXd OnesPseudoA = VectorXd::Ones(na);
	VectorXd OnesPseudoX = VectorXd::Ones(nx_test);
	VectorXd lengths_test = Gx_a_test * OnesPseudoA;
	lengths_test = lengths_test.cwiseMax(3.0*OnesPseudoX);
	int inference = decode(alpha, beta, lengths_test, Gx_a_test, (char*)FILE_hi_WRITE);
	TIME_end = clock();
	double time_decode = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for decoding = %5.10e (Seconds)\n", time_decode);
	*/

	cout << "(6) Program over------------" << endl;
	//printf("\ntime taken for execution of the whole program = %5.10e (Seconds)\n", time_whitening + time_stpm + time_post);
	return 0;
}
