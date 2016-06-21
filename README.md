#Topic Modeling via Method of Moments

This code performs learning of topic models via method of moments using tensor decomposition on a single machine. Specifically, the code first performs a pre-processing step which implements a whitening transformation for matrix/tensor orthogonalisation and dimensionality reduction and then finds a decomposition of the tensor using Stochastic Gradient Descent. This code derives from the project in https://github.com/FurongHuang/TensorDecomposition4TopicModeling.git, for which a full description can be found [here](http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/TopicModeling.html).

This project has also been adapted to be able to learn multiple LDA models of many datasets on the same run. The code for inference has been left commented as it still needs to be adapted to work with multiple LDA datasets, although it works well for one single dataset.

#Basic usage
To start using this code, go to a terminal and do: 

1) cd TopicModelingSTGD/TopicModel/TopicModel/
2) make exe-topicmodel 
3) make runtopic

To change the way the tensor is learnt (using either Alternated Least Squares (ALS) or Stochastic Gradient Descent (STGD)), the variable "option" can be changed in TopicModelingSTGD/TopicModel/TopicModel/TopicModel.cpp:
- option = 1 --> use ALS
- option = 2 --> use STGD
#Results
The results are printed in TopicModelingSTGD/TopicModel/datasets/(your_dataset)/result
This will create three output files called "corpus_topic_weights_(number_of_dataset).txt", "topic_word_matrix_(number_of_dataset).txt" and can be easily adapted to produce also a file "inferred_topic_weights_per_document_(number_of_dataset).txt". These files are the estimated parameters of the model.

#Arguments
	1) Directory of a .txt file containing the number of documents on each of the N_times LDA datasets to learn (one dataset/number on each line)
	2) N_times: number of LDA models to learn
	3) Voca_size: number of words in the xulary
	4) Hidden_size: number of topics to learn
	5) ALPHA0: alpha_0 parameter of the LDA model
	6) IndexStart: 1
	7) First part of the directory of the files containing the training dataset, in the format "directory/name_of_file" (see format of input files)
	8) First part of the directory of the files containing the testing dataset, in the format "directory/name_of_file" (see format of input files)
	9) Result_topic_eigenvalue: directory for the result file "corpus_topic_weights.txt"
	10) Result_topic_eigenvector: directory for the result file "topic_word_matrix.txt"
	11) Result_topic_inferred_membership: directory for the result file "inferred_topic_weights_per_document.txt"
	12) STGD_batch_size: Batch size for Stochastic Gradient Descent (number of points to use on each iteration)
	13) Directory of a .txt file containing the number of documents on each of the N_times LDA datasets to test (one dataset/number on each line)



#See format of input files
For each LDA model j in {1, ..., N_times}, the files will be in the same folder and named in the following way:
	- directory/name_of_file_1_train.txt
	- directory/name_of_file_1_test.txt
	- ...
	- ...
	- directory/name_of_file_(N_times)_train.txt
	- directory/name_of_file_(N_times)_test.txt

The format of each bag-of-words file is:

docID wordID count 

docID wordID count 

docID wordID count 

docID wordID count 

... 

docID wordID count 

docID wordID count 

docID wordID count 
