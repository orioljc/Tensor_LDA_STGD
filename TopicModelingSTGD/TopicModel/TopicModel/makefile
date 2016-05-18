SRC=TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp
CFLAGS=-lsvd -lm
COPTZ=-O3 -g  -I.

SUPP_WARN=-Wno-write-strings
NX_test_file = 
Hidden_size=3
N_times = 2
ALPHA0=1
IndexStart=1

Voca_size=100
NX_file = number_of_documents_train.txt
Ntest_file = number_of_documents_test.txt
DataFolder =../datasets/synthetic
Sample_train= splitted_data/samples
Sample_test = splitted_data/samples
STGD_batch_size = 10

ResultFolder= $(DataFolder)/result
Result_topic_eigenvalue = corpus_topic_weights
Result_topic_eigenvector = topic_word_matrix
Result_topic_inferred_membership = inferred_topic_weights_per_document


exe-topicmodel: TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp util2.cpp
	g++ -o $@ $^ $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

runtopic:
	./exe-topicmodel $(DataFolder)/$(NX_file) $(N_times) $(Voca_size)  $(Hidden_size)  $(ALPHA0) $(IndexStart) $(DataFolder)/$(Sample_train)  $(DataFolder)/$(Sample_test)  $(ResultFolder)/$(Result_topic_eigenvalue)  $(ResultFolder)/$(Result_topic_eigenvector) $(ResultFolder)/$(Result_topic_inferred_membership) $(STGD_batch_size) $(DataFolder)/$(Ntest_file)

cleantopic:
	rm exe-topicmodel
