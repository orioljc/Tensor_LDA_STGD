#include "stdafx.h"
#include <stdint.h>
extern double alpha0;
extern int KHID;
extern int *NX;
extern int STGD_batch_size;
extern bool print;
using namespace Eigen;
using namespace std;

static int
accumulate_matricization(MatrixXd &Ta, VectorXd Wc, unsigned int length, double a0, VectorXd m1, unsigned int kstart, unsigned int kend, unsigned int k)
{
	if (length > 2)
	{

		double scale2fac = a0 * (a0 + 1.0)
			/ (2.0 * length * (length - 1));

		double scale3fac = (a0 + 1.0) * (a0 + 2.0)
			/ (2.0 * length * (length - 1) * (length - 2) );

		for (unsigned int i = kstart; i <= kend; ++i)
		{
			for (unsigned int j = 0; j < k; ++j)
			{
				for (unsigned int l = 0; l < k; ++l)
				{
					/* Wc ( Wc \odot Wc )^\top */
					/*topic shift scale3fac first term*/

					Ta(i - kstart,k*j + l) += scale3fac * Wc[i] * Wc[l] * Wc[j];

					/*dirichelet second order term (new term!!!)*/
					Ta(i - kstart,k*j + l) -= scale2fac * Wc[i] * m1[l] * Wc[j];
					Ta(i - kstart,k*j + l) -= scale2fac * Wc[i] * Wc[l] * m1[j];
					Ta(i - kstart,k*j + l) -= scale2fac * m1[i] * Wc[l] * Wc[j];
				}

				/* - \sum_{i=1}^d \sum_{j=1}^d Wc_i Wc_j e_i (e_i \odot e_j)^\top
				- \sum_{i=1}^d \sum_{j=1}^d Wc_i Wc_j e_i (e_j \odot e_i)^\top
				- \sum_{i=1}^d \sum_{j=1}^d Wc_i Wc_j e_i (e_j \odot e_j)^\top
				*/
				/*tpic shift scale3fac 2nd, 3rd, 4th term*/
				Ta(i - kstart,k*i + j) -= scale3fac * Wc[i] * Wc[j];
				Ta(i - kstart,k*j + i) -= scale3fac * Wc[i] * Wc[j];
				Ta(i - kstart,k*j + j) -= scale3fac * Wc[i] * Wc[j];

				/* - \sum_{j=1}^d Wc_j e_j (e_j \odot m1)^\top
				- \sum_{j=1}^d Wc_j e_j (m1 \odot e_j)^\top
				- \sum_{j=1}^d Wc_j m1 (e_j \odot e_j)^\top */
				/*this is problematic*/
				Ta(i - kstart,k*i + j) += scale2fac * Wc[i] * m1[j];
				Ta(i - kstart,k*j + i) += scale2fac * Wc[i] * m1[j];
				Ta(i - kstart,k*j + j) += scale2fac * m1[i] * Wc[j];
			}

			/* + 2 \sum_{i=1}^d Wc_i e_i (e_i \odot e_i)^\top */
			/*topic shift scale3fac fifth term */
			Ta(i - kstart,k*i + i) += 2.0 * scale3fac * Wc[i];
		}
	}

	return (length > 2);
}

int * read_vector_ints(const char *file_name, int NT){
	FILE *my_file;
	int *x;
	int i;
	int prova;
	
	my_file = fopen(file_name, "r");
	x = (int*)calloc(NT, sizeof(int));
	if( x == NULL){
		printf("error: could not allocate memory in function 'read_vector_ints'\n");
		exit(1);
	}

	//Read number of times
	for(i=0; i<NT; i++){
		fscanf(my_file, "%d", &x[i]);
	}
	fclose(my_file);

	return x;
}



double compute_loss(MatrixXd emp_T3_whitened, MatrixXd phi){
	double loss = 0.;
	for(int n = 0; n<KHID; n++){
		MatrixXd temp = MatrixXd::Zero(KHID, KHID);
		for (int i = 0; i < KHID; i++){
			//temp = ( phi.col(i) * phi.col(i).transpose() );
			temp += phi(n,i) * ( phi.col(i) * phi.col(i).transpose() );
		}
		//Calculate squared Frobenius Norm
		//temp /= KHID;
		loss += THETA * temp.squaredNorm();
		temp -= emp_T3_whitened.block(0,n*KHID, KHID, KHID);
		//Calculate squared Frobenius Norm
		loss += temp.squaredNorm();	
	}
	return loss;
}

double compute_loss_point(int datapoint, MatrixXd phi, MatrixXd emp_T3_whitened_x){
	double loss = 0.;
	for(int n = 0; n<KHID; n++){
		MatrixXd temp = MatrixXd::Zero(KHID, KHID);
		for (int i = 0; i < KHID; i++){
			temp += phi(n,i) * ( phi.col(i) * phi.col(i).transpose() );
		}
		//Calculate squared Frobenius Norm
		loss += (1.+THETA) * temp.squaredNorm();

		//Calculate and add Second term
		temp = temp.array() * emp_T3_whitened_x.block(0,n*KHID, KHID, KHID).array();
		loss -= 2.*temp.sum();	
	}
	return loss;
}


MatrixXd compute_empirical_T3(SparseMatrix<double> whitened_data, VectorXd y_mean){
	int nx = (int)whitened_data.cols();
	//printf("nx = %d\n", nx);
	double shift12 = (alpha0 + 1.0)*(alpha0 + 2.0) / (2.0*(double)nx);
	double shift01 = -alpha0 *(alpha0 + 1.0) / (2.0*(double)nx);
	double shift00 = alpha0*alpha0;

	MatrixXd emp_T3_whitened = MatrixXd::Zero(KHID, KHID * KHID);
	for(int n=0; n<nx; n++){
		VectorXd y = whitened_data.col(n);
		
		MatrixXd temp0, temp1, temp2, temp3, temp4;
		temp0.noalias() = shift12 * (y * y.transpose());
		temp1.noalias() = shift01 * (y * y.transpose());
		temp2.noalias() = shift01 * (y * y_mean.transpose());
		temp3.noalias() = shift01 * (y_mean * y.transpose());
		temp4.noalias() = shift00 * (y_mean * y_mean.transpose());
		for(int i=0; i < KHID; i++){
			emp_T3_whitened.block(0,i*KHID, KHID, KHID).noalias() += temp0*y(i);
			emp_T3_whitened.block(0,i*KHID, KHID, KHID).noalias() += temp1*y_mean(i);
			emp_T3_whitened.block(0,i*KHID, KHID, KHID).noalias() += temp2*y(i);
			emp_T3_whitened.block(0,i*KHID, KHID, KHID).noalias() += temp3*y(i);
			emp_T3_whitened.block(0,i*KHID, KHID, KHID).noalias() += temp4*y_mean(i);			
		}
	}
	return emp_T3_whitened;
}


MatrixXd create_tensor_phi( VectorXd eigval, MatrixXd phi, int size){
	MatrixXd tensor = MatrixXd::Zero(KHID, KHID*KHID);
	for(int n=0; n<size; n++){
		for(int i=0; i < size; i++){
			tensor.block(0,i*size, size, size).noalias() += eigval(n) * phi(i,n) * ( phi.col(n) * phi.col(n).transpose() );
		}
	}
	return tensor;
}



///////////////// BEGIN Compute M1, M2 and W not efficiently /////////////////

VectorXd compute_M1_topic( SparseMatrix<double> data ){
	cout << "Computing M1 topic..." << endl;
	int n = data.rows();
	VectorXd M1 = VectorXd::Ones(n).transpose() * data;

	return M1/n;
}

SparseMatrix<double> compute_M2_topic( SparseMatrix<double> data ){
	cout << "Computing M2 topic..." << endl;
	int n = data.rows();
	int na = data.cols();
	SparseMatrix<double> M2(na,na);
	for (int t = 0; t < n; ++t){
		cout << t << endl;

		VectorXd c = data.row(t);
		SparseVector<double> c_sp = data.row(t);
		MatrixXd temp = c.asDiagonal();
		M2 += (c_sp*c_sp.transpose() - temp.sparseView());
	}
	M2 *= (alpha0+1.)/n;

	SparseVector<double> M1 = compute_M1_topic(data).sparseView();

	cout << M2.cols() << endl;
	cout << M2.rows() << endl;
	cout << M1.size() << endl;
	SparseMatrix<double> aux = alpha0 * ( M1 * M1.transpose() );
	cout << aux.cols() << endl;
	cout << aux.rows() << endl;
	M2 = M2 - aux;
	return M2;
}

MatrixXd compute_w( SparseMatrix<double> M2, int KHID ){
	cout << "Computing the whitening matrix..." << endl;
	int m = M2.rows();
	cout << "Calculating SVD..." << endl;
	JacobiSVD<MatrixXd> svd( M2, ComputeThinU );
	cout << "SVD successfully calculated" << endl;
	VectorXd E = svd.singularValues();
	E = E.head(KHID);
	E = E.array().sqrt().inverse().matrix();
	MatrixXd E_diag = E.asDiagonal();
	MatrixXd U = svd.matrixU();
	U = U.block(0,0,m,KHID);
	cout << U.rows() << endl;
	cout << U.cols() << endl;
	return U*E_diag;
}

///////////////// END Compute M1, M2 and W not efficiently /////////////////



void tensor_STGD( MatrixXd tensor, SparseMatrix<double> whitened_data, VectorXd whitened_mean, VectorXd &lambda, MatrixXd & evec_new, VectorXd Lengths, int t) {
	double error, loss;
	MatrixXd A_random(MatrixXd::Random(KHID, KHID));
	MatrixXd evec_old;

	A_random.setRandom();
	HouseholderQR<MatrixXd> qr(A_random);
	evec_new = normc(qr.householderQ());

	VectorXd eigval = VectorXd::Ones(KHID);

	lambda = VectorXd::Zero(KHID);
	A_random.resize(0, 0);
	long iteration = 1;

	int iter_aux, aux2, aux3;
	double learningrate_evec = 1.;
	printf("Introduce a number n such that the learning rate will be 10^{-n}.\nIntroduce the number of iterationf for this learning rate.\n");
	scanf("%d", &aux2);
	for( int i=0; i<aux2; i++)
		learningrate_evec /= 10.;
	scanf("%d", &iter_aux);
	// MatrixXd temp = create_tensor_phi( eigval, phi_aux, KHID);

	MatrixXd mean_term;
	mean_term = compute_mean_term_tensor_x(whitened_mean);
	while (true) {
		int iii = iteration % NX[t];

		/////////////// Compute tensor_x
		unsigned int k = KHID, kstart = 0, kend = k-1;	assert(kend < k);
		unsigned int krange = 1 + (kend - kstart);
		MatrixXd deriv = MatrixXd::Zero(KHID, KHID);
		MatrixXd emp_deriv = MatrixXd::Zero(KHID, KHID);
		//MatrixXd tensor_test = MatrixXd::Zero(KHID, KHID*KHID);

		// Generate permutation
		std::vector<unsigned int> permutation(NX[t]);
		std::iota(permutation.begin(), permutation.end(), 0);
		std::random_shuffle(permutation.begin(), permutation.end());

  		//int batch_size = NX[t];
  		int batch_size = STGD_batch_size;
  		// double time_AM = 0., time_deriv = 0., time_emp_deriv = 0.;
		for (int examples = 0; examples < batch_size; ++examples){
			clock_t time0, time1, time2, time3;
			VectorXd currentDataPoint = whitened_data.col(permutation[examples]);
			MatrixXd tensor_x = mean_term;
			// time0 = clock();
			accumulate_matricization(tensor_x, currentDataPoint, (unsigned int)Lengths(permutation[examples]), alpha0, whitened_mean, kstart, kend, k);
			// time1 = clock();
			deriv += compute_derivatives(tensor_x, evec_new);
			// time2 = clock();
			//emp_deriv += compute_empirical_derivatives(tensor_x, evec_new, examples);	
			// time3 = clock();
			
			// time_AM += double(time1 - time0) / CLOCKS_PER_SEC;
			// time_deriv += double(time2 - time1) / CLOCKS_PER_SEC;
			// time_emp_deriv += double(time3 - time2) / CLOCKS_PER_SEC;
			
			//cout << "deriv:\n" << deriv << endl;
			//cout << "emp_deriv:\n" << emp_deriv << endl;
			//tensor_test += tensor_x;
		}

		// printf("time_AM = %5.10e (Seconds)\n", time_AM);	
		// printf("time_deriv = %5.10e (Seconds)\n", time_deriv);	
		// printf("time_emp_deriv = %5.10e (Seconds)\n", time_emp_deriv);	

		deriv /= batch_size;
		//emp_deriv /= batch_size;
		//tensor_test /= batch_size;
		if( print == true ){
			// cout << "tensor:\n" << tensor << endl;
			// cout << "tensor_test:\n" << tensor_test << endl;
			//printf("(tensor - tensor_test).norm(): %lf\n", (tensor - tensor_test).norm());
			cout << "deriv:\n" << deriv << endl;	
			//cout << "emp_deriv:\n" << emp_deriv << endl;
		}
		evec_old = evec_new;
		evec_new = evec_new - learningrate_evec * deriv;

		if( print == true ){
			cout << "evec_new: " << evec_new << endl;
		}

		///////////////////////////////////////////////
		if (iteration >= 0 /*MINITER*/){
			error = (normc(evec_new) - normc(evec_old)).norm();
			loss = compute_loss( tensor, evec_new );
			printf("loss: %6.5e \t error: %6.5e \t iteration: %ld\n", loss, error, iteration);
			if ( error < TOLERANCE || iteration > MAXITER) break;
		}

		// //Pause program until a key is pressed
		// std::cout << "Press any key to continue\n";
		// std::cin.ignore();
		if( iteration >= iter_aux ){
			if( print == true ){
				//cout << "evec_new:\n" << evec_new << endl;
				MatrixXd temp = create_tensor_phi( eigval, evec_new, KHID);
				cout << "reconstructed tensor (all):\n" << temp << endl;
				cout << "tensor:\n" << tensor << endl;
			}
			int aux1;
			int aux2;
			printf("Introduce a number n such that the learning rate will be 10^{-n}.\nIntroduce the number of iterationf for this learning rate.\n");
			scanf("%d", &aux2);
			learningrate_evec = 1.;
			for( int i=0; i<aux2; i++){
				learningrate_evec /= 10.;
			}
			scanf("%d", &aux3);
			iter_aux += aux3;
			double learningrate_evec = min(1e-9, 1.0 / sqrt((double)iteration));
		}

		iteration++;
	}

	lambda = (((evec_new.array().pow(2)).colwise().sum()).pow(3.0 / 2.0)).transpose();
	evec_new = normc(evec_new);
}

MatrixXd compute_derivatives(MatrixXd tensor_x, MatrixXd evec){
	MatrixXd mat;
	MatrixXd deriv = MatrixXd::Zero(KHID, KHID);
	
	// compute term1
	mat = evec.transpose()*evec;
	mat = mat.cwiseProduct(mat);
	deriv = evec * mat;

	// compute term2
	MatrixXd term2 = MatrixXd::Zero(KHID, KHID);
	for(int i=0; i<KHID; i++){
		MatrixXd mat2 = tensor_x.block(0, i*KHID, KHID, KHID);
		mat2 = mat2.transpose() * evec;
		MatrixXd mat3 = mat2.cwiseProduct(evec);
		term2.row(i) = mat3.colwise().sum();
	}
	deriv = (1. + THETA) * deriv - term2;

	return 6.*deriv;
}

MatrixXd compute_empirical_derivatives(MatrixXd tensor, MatrixXd evec, int datapoint){
	MatrixXd deriv = MatrixXd::Zero(KHID, KHID);
	MatrixXd evec2 = evec;
	double h = 1e-3;
	for(int i=0; i<KHID; i++){
		for(int j=0; j<KHID; j++){
			evec2 = evec;
			evec2(i,j) += h;
			deriv(i,j) = (compute_loss_point( datapoint, evec2, tensor ) - compute_loss_point( datapoint, evec, tensor )) / h;
		}
	}
	return deriv;
}

MatrixXd compute_mean_term_tensor_x(VectorXd whitenedMean){
	unsigned int k = KHID;
	unsigned int kstart = 0;	unsigned int kend = k-1;	assert(kend < k);
	unsigned int krange = 1 + (kend - kstart);
	double alpha0sq = alpha0 * alpha0;
	MatrixXd mean_term = MatrixXd::Zero(KHID, KHID*KHID);
	for (unsigned int i = kstart; i <= kend; ++i){
		for (unsigned int j = 0; j < k; ++j){
			for (unsigned int l = 0; l < k; ++l){
				mean_term(i - kstart, k*j + l) += alpha0sq * whitenedMean[i] * whitenedMean[l] * whitenedMean[j];
			}
		}		
	}
	return mean_term;
}