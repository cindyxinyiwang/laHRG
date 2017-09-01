import re
import sys
import copy
sys.path.insert(0, 'script/')

from random import random
from bisect import bisect

import new_em
import numpy as np
import networkx as nx
from collections import deque
import os, errno

def ensure_dir(dirname):
	"""
	Ensure that a named directory exists; if it does not, attempt to create it.
	"""
	try:
		os.makedirs(dirname)
	except OSError, e:
		if e.errno != errno.EEXIST:
			raise

def train_test(train_file, test_file_base, result_dir, test_nums, split, train_sample_size, 
	subgraph_size, smooth=True, use_converge=True, converge=1):
	# grammar output : results/graph_name/split_n/grammar.txt
	result_file = open("%sresult_split_%d.txt" % (result_dir, split), 'w')
	grammar_file = open("%ssplit_%d.gram" % (result_dir, split), 'w')
	print "subgraph size: %d, train sample size: %d" % (subgraph_size, train_sample_size)
	print "split: ", split
	result_file.write("subgraph size:%d, train sample size:%d\n" % (subgraph_size, train_sample_size))
	result_file.write("split: %d\n"%split)

	cv_train = new_em.ConvertRule(train_file, tree_count=train_sample_size)
	max_likelihood = float("-inf")
	gram = new_em.Grammar(cv_train.rule_dict, split)
	for i in xrange(5):
		em = new_em.EM(gram, cv_train.Tree)
		em.iterations(use_converge, converge=converge)
		train_loglikelihood = em.loglikelihood
		result_file.write("train loglikelihood:"+str(train_loglikelihood)+'\n') 
		if train_loglikelihood > max_likelihood:
			max_likelihood = train_loglikelihood
	# output grammar
	rules = em.gram.get_valid_rules(cv_train)
	rules = sorted(rules, key=lambda x: x[2], reverse=True)	# sort by probability	
	for (id, hrg, prob) in rules:
		lhs, rhs = hrg
		grammar_file.write(lhs + " " + str(rhs) + " " + str(prob) + '\n')

	# get test likelihoods
	original_train_gram_ruledict = copy.deepcopy(em.gram.rule_dict)
	original_train_gram_alphabet = copy.deepcopy(em.gram.alphabet)
	for test_num in test_nums:
		# reset training grammar
		em.gram.alphabet = copy.deepcopy(original_train_gram_alphabet)
		train_gram_rules = copy.deepcopy(original_train_gram_ruledict)
		test_file = test_file_base + str(test_num) + ".txt"
		cv_test = new_em.ConvertRule(test_file, tree_count=4)
		test_gram_rules = new_em.Grammar(cv_test.rule_dict, split).rule_dict
		added_nonterms = set()
		added_rules = set()
		epsilon = float("1e-323")
		if smooth:
			added_gram_count = 0
			for lhs in test_gram_rules:
				if lhs not in train_gram_rules:
					train_gram_rules[lhs] = {}
					added_nonterms.add(lhs)
				for rhs, prob in test_gram_rules[lhs].items():
					if rhs not in train_gram_rules[lhs]:
						train_gram_rules[lhs][rhs] = epsilon
						added_gram_count += 1
						added_nonterms.union(rhs.split())
						added_rules.add(lhs + "->" + rhs)
			em.gram.rule_dict = train_gram_rules
			em.gram.alphabet = em.gram.alphabet.union(added_nonterms)
		# get test likelihood
		em_test = new_em.EM(em.gram, cv_test.Tree)
		use_added_rules = em_test.get_loglikelihood(added_rules, result_file)
		if use_added_rules == 0:
			print "zero smooth test num: ", test_num
		test_loglikelihood = em_test.loglikelihood 
		
		result_file.write("test number:%d\n" % test_num)
		result_file.write("smooth count:%s\n" % str(use_added_rules))
		result_file.write("test loglikelihood:%s\n" % str(test_loglikelihood))
		

if __name__ == "__main__":
	subgraph_size = int(sys.argv[1])
	train_sample_size = int(sys.argv[2])
	split = int(sys.argv[3])
	train_graph_name = 'ba-train'
	test_graph_name = 'ba-test'
	result_dir = "results_new/ba/"
	ensure_dir(result_dir)
	num_test_sample = 100
	test_nums = [i for i in xrange(100)]
	train_file = "prepare_tree_rules/%s/%d_sub/nonpartition/%d_sample/%s_train.txt" % (train_graph_name, subgraph_size, 500, train_graph_name)
	test_file_base = "prepare_tree_rules/%s/%d_sub/nonpartition/%d_sample/%s_" % (test_graph_name, subgraph_size, 4, test_graph_name)
	test_loglikelihood = train_test(train_file, test_file_base, result_dir, test_nums, split, train_sample_size, subgraph_size,
			smooth=True, use_converge=True, converge=1)
