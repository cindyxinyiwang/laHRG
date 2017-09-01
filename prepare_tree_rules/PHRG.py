__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import os
import pprint as pp
import re

import networkx as nx

import david as pcfg
import graph_sampler as gs
import tree_decomposition as td

# prod_rules = {}
DEBUG =  False

left_deriv_file_name = "karate_left_deriv_rules.txt"

def graph_checks (G):
  ## Target number of nodes
  global num_nodes
  num_nodes = G.number_of_nodes()

  if not nx.is_connected(G):
	if DEBUG: print "Graph must be connected";
	os._exit(1)

  if G.number_of_selfloops() > 0:
	if DEBUG: print "Graph must be not contain self-loops";
	os._exit(1)


def matcher (lhs, N):
  if lhs == "S":
	return [("S", "S")]
  m = []
  for x in N:
	if len(x) == lhs.count(",") + 1:
	  i = 0
	  for y in lhs.split(","):
		m.append((x[i], y))
		i += 1
	  return m


def binarize (tree):
  (node, children) = tree
  children = [binarize(child) for child in children]
  if len(children) <= 2:
	return (node, children)
  else:
	# Just make copies of node.
	# This is the simplest way to do it, but it might be better to trim unnecessary members from each copy.
	# The order that the children is visited is arbitrary.
	binarized = (node, children[:2])
	for child in children[2:]:
	  binarized = (node, [binarized, child])
	return binarized


def grow (rule_list, grammar, diam=0):
  D = list()
  newD = diam
  H = list()
  N = list()
  N.append(["S"])  # starting node
  ttt = 0
  # pick non terminal
  num = 0
  for r in rule_list:
	rule = grammar.by_id[r][0]
	lhs_match = matcher(rule.lhs, N)
	e = []  # edge list
	match = []
	for tup in lhs_match:
	  match.append(tup[0])
	  e.append(tup[1])
	lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"

	new_idx = {}
	n_rhs = rule.rhs
	if 1: print lhs_str, "->", n_rhs
	for x in n_rhs:
	  new_he = []
	  he = x.split(":")[0]
	  term_symb = x.split(":")[1]
	  for y in he.split(","):
		if y.isdigit():  # y is internal node
		  if y not in new_idx:
			new_idx[y] = num
			num += 1
			if diam > 0 and num >= newD and len(H) > 0:
			  newD = newD + diam
			  newG = nx.Graph()
			  for e in H:
				if (len(e) == 1):
				  newG.add_node(e[0])
				else:
				  newG.add_edge(e[0], e[1])
				# D.append(metrics.bfs_eff_diam(newG, 20, 0.9))
		  new_he.append(new_idx[y])
		else:  # y is external node
		  for tup in lhs_match:  # which external node?
			if tup[1] == y:
			  new_he.append(tup[0])
			  break
	  # prod = "(" + ",".join(str(x) for x in new_he) + ")"
	  if term_symb == "N":
		N.append(sorted(new_he))
	  elif term_symb == "T":
		H.append(new_he)  # new (h)yper(e)dge
	  # print n_rhs, new_he
	match = sorted(match)
	N.remove(match)

  newG = nx.Graph()
  for e in H:
	if (len(e) == 1):
	  newG.add_node(e[0])
	else:
	  newG.add_edge(e[0], e[1])

  return newG, D


def probabilistic_hrg (G, num_samples=1, n=None):
  '''
  Args:
  ------------
	G: input graph (nx obj)
	num_samples:   (int) in the 'grow' process, this is number of
				   synthetic graphs to generate
	n: (int) num_nodes; number of nodes in the resulting graphs
	Returns: List of synthetic graphs (H^stars)
  '''
  graphletG = []

  if DEBUG: print G.number_of_nodes()
  if DEBUG: print G.number_of_edges()

  G.remove_edges_from(G.selfloop_edges())
  giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
  G = nx.subgraph(G, giant_nodes)

  if n is None:
	num_nodes = G.number_of_nodes()
  else:
	num_nodes = n

  if DEBUG: print G.number_of_nodes()
  if DEBUG: print G.number_of_edges()

  graph_checks(G)

  if DEBUG: print
  if DEBUG: print "--------------------"
  if DEBUG: print "-Tree Decomposition-"
  if DEBUG: print "--------------------"

  prod_rules = {}
  if num_nodes >= 500:
	for Gprime in gs.rwr_sample(G, 2, 300):
	  T = td.quickbb(Gprime)
	  root = list(T)[0]
	  T = td.make_rooted(T, root)
	  T = binarize(T)
	  root = list(T)[0]
	  root, children = T
	  td.new_visit(T, G, prod_rules, TD)
  else:
	T = td.quickbb(G)
	root = list(T)[0]
	T = td.make_rooted(T, root)
	T = binarize(T)
	root = list(T)[0]
	root, children = T
	td.new_visit(T, G, prod_rules, TD)

  if DEBUG: print
  if DEBUG: print "--------------------"
  if DEBUG: print "- Production Rules -"
  if DEBUG: print "--------------------"

  for k in prod_rules.iterkeys():
	#if DEBUG: print k
	print k
	s = 0
	for d in prod_rules[k]:
	  s += prod_rules[k][d]
	for d in prod_rules[k]:
	  prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
	  if DEBUG: print '\t -> ', d, prod_rules[k][d]

  rules = []
  id = 0
  for k, v in prod_rules.iteritems():
	sid = 0
	for x in prod_rules[k]:
	  rhs = re.findall("[^()]+", x)
	  rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
	  print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
	  sid += 1
	id += 1
  # print rules
  exit()

  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in rules:
	# print type(id), type(lhs), type(rhs), type(prob)
	g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  if DEBUG: print "Starting max size"
  num_nodes = num_nodes
  num_samples = num_samples

  g.set_max_size(num_nodes)

  if DEBUG: print "Done with max size"

  Hstars = []

  for i in range(0, num_samples):
	rule_list = g.sample(num_nodes)
	# print rule_list
	hstar = grow(rule_list, g)[0]
	# print "H* nodes: " + str(hstar.number_of_nodes())
	# print "H* edges: " + str(hstar.number_of_edges())
	Hstars.append(hstar)

  return Hstars


def phrg_derive_prod_rules_partition(G, file_name_list, sample_size_list, subgraph_size):
  G.remove_edges_from(G.selfloop_edges()) 
  giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
  G = nx.subgraph(G, giant_nodes)

  num_nodes = G.number_of_nodes()
  graph_checks(G)

  total_sample = sum(sample_size_list)
  file_list = []
  for name in file_name_list:
    file_list.append(open(name, "a+"))
  cur_sample_count = 0
  f_idx = 0	# index of current file to write
  for Gprime in gs.disjoint_sample(G, sample_size_list, subgraph_size):
    prod_rules = {}
    left_deriv_prod_rules = []
    T = td.quickbb(Gprime)
    root = list(T)[0]
    T = td.make_rooted(T, root)
    T = binarize(T)
    root = list(T)[0]
    root, children = T
    td.new_visit(T, G, prod_rules, left_deriv_prod_rules)
    cur_sample_count += 1
    
    if sum(sample_size_list[:f_idx+1]) < cur_sample_count:
    	f_idx += 1
    print cur_sample_count, file_name_list[f_idx]
    for r in left_deriv_prod_rules:
      file_list[f_idx].write(r)
      file_list[f_idx].write('\n')
  for file in file_list:
  	file.close()

# def probabilistic_hrg_deriving_prod_rules(G, num_samples=1, n=None):
# hrg_baseline.py
def probabilistic_hrg_deriving_prod_rules (G, left_deriv_file_name, num_samples=4, subgraph_size=500, n=None):
  '''
	Rule extraction procedure

		'''
  if G is None: return

  G.remove_edges_from(G.selfloop_edges())
  giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
  G = nx.subgraph(G, giant_nodes)

  if n is None:
	num_nodes = G.number_of_nodes()
  else:
	num_nodes = n

  graph_checks(G)

  #k = num_nodes / 300
  #print "number of samples:", k * 20
  if DEBUG: print
  if DEBUG: print "--------------------"
  if DEBUG: print "-Tree Decomposition-"
  if DEBUG: print "--------------------"
  prod_rules = {}
  left_deriv_prod_rules = []
  if num_nodes >= 500:
	for Gprime in gs.rwr_sample(G, num_samples, subgraph_size):
	  T = td.quickbb(Gprime)
	  root = list(T)[0]
	  T = td.make_rooted(T, root)
	  T = binarize(T)
	  root = list(T)[0]
	  root, children = T
	  td.new_visit(T, G, prod_rules, left_deriv_prod_rules)
  else:
	T = td.quickbb(G)
	root = list(T)[0]
	T = td.make_rooted(T, root)
	T = binarize(T)
	root = list(T)[0]
	root, children = T
	td.iter_dfs_visit(T, G, prod_rules, left_deriv_prod_rules)

  left_derive_file = open(left_deriv_file_name, 'w')
  for r in left_deriv_prod_rules:
  	left_derive_file.write(r)
  	left_derive_file.write('\n')
  left_derive_file.close()
	#exit()
  
  if DEBUG: print
  if DEBUG: print "--------------------"
  if DEBUG: print "- Production Rules -"
  if DEBUG: print "--------------------"

  for k in prod_rules.iterkeys():
	#print k
	s = 0
	for d in prod_rules[k]:
	  s += prod_rules[k][d]
	for d in prod_rules[k]:
	  prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
	  #print '\t -> ', d, prod_rules[k][d]

  # pp.pprint(prod_rules)

  rules = []
  id = 0
  for k, v in prod_rules.iteritems():
	sid = 0
	for x in prod_rules[k]:
	  rhs = re.findall("[^()]+", x)
	  rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
	  #print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
	  sid += 1
	id += 1

  return rules
