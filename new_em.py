import re
import sys
sys.path.insert(0, 'script/')

from random import random
from bisect import bisect
from collections import deque, OrderedDict
import numpy as np
import networkx as nx

class Grammar(object):
	def __init__(self, rule_dict, k, no_random_init=False):
		self.alphabet = set(["t"])
		self.start = "S"
		self.rule_dict = rule_dict
		self.k = k

		if not no_random_init:
			self.getAllRules()
		else:
			new_rule_dict = {}
			for lhs, dic in self.rule_dict.items():
				rules = self.rule_dict[lhs]
				total = sum(rules.values())

				if lhs != 'S':
					lhs = lhs + '_1'
				self.alphabet.add(lhs)
				new_rule_dict[lhs] = {}
				for rhs in dic:
					r_tok = rhs.split()
					rhs_list = []
					for r in r_tok:
						if r != 't':
							self.alphabet.add(r + "_1")
							rhs_list.append(r + "_1")
						else:
							rhs_list.append(r)
					new_rule_dict[lhs][" ".join(rhs_list)] = rules[rhs] / float(total)
			self.rule_dict = new_rule_dict

	def getAllRules(self):
		"""
		convert original rule_dict into rule_dict with k copies of each nonterm
		"""
		# get a combination of all rules
		def get_combs(nonterms, terms, rules_comb, idx, cur_rule):
			if idx == len(nonterms):
				r = " ".join(cur_rule) + " " + " ".join(terms)
				rules_comb.append(r.strip())
				return
			for i in xrange(1, self.k+1):
				cur_nonterm = nonterms[idx] + "_" + str(i)
				self.alphabet.add(cur_nonterm)
				cur_rule.append(cur_nonterm)
				get_combs(nonterms, terms, rules_comb, idx + 1, cur_rule)
				cur_rule.pop()

		new_rule_dict = {}
		for lhs in self.rule_dict:
			rules = self.rule_dict[lhs]
			# get all possible combs of rules
			rules_comb = []
			for r in rules:
				r_tok = r.split()
				nonterms = [t for t in r_tok if t != "t"]
				terms = [t for t in r_tok if t == "t"]
				# find all combs
				get_combs(nonterms, terms, rules_comb, 0, [])
			new_lhs = []
			if lhs == "S":
				new_lhs = ["S"]
				self.alphabet.add("S")
			else:
				for i in xrange(1, self.k+1):
					new_lhs.append(lhs + "_" + str(i))
					self.alphabet.add(lhs + "_" + str(i))
			for l in new_lhs:
				new_rule_dict[l] = self._get_rules(rules_comb)
		self.rule_dict = new_rule_dict

	def printAllRules(self):
		for s in self.rule_dict:
			for r in self.rule_dict[s]:
				print s + "->" + r + " " + str(self.rule_dict[s][r])

	def _get_rules(self, rules):
		"""
		input: list of all rules
		output: dictionary of rules and random probs
		"""
		n = len(rules)
		a = np.random.random(n)
		a /= a.sum()
		d = {}
		count = 0
		for i in rules:
			d[i] = a[count]
			count += 1
		return d

	def sample(self):
		tree = TreeNode(self.start)
		rules = []
		cur_nonterm = deque([self.start])
		cur_tree = deque([tree])
		term_count = 0
		while cur_nonterm:
			lhs = cur_nonterm.popleft()
			root_node = cur_tree.popleft()
			rhs_dic = self.rule_dict[lhs]
			rule = np.random.choice(rhs_dic.keys(), p = rhs_dic.values())
			rules.append((lhs, rule, rhs_dic[rule]))
			rule = rule.split()
			for r in rule:
				child_node = TreeNode(r)
				root_node.children.append(child_node)
				if r != "t":
					cur_nonterm.append(r)
					cur_tree.append(child_node)
				else:
					term_count += 1
		return rules, term_count, tree

	def get_valid_rules(self, cv):
		# return all rules with positive probability
		count = 0
		total = 0
		rules = []
		for lhs in self.rule_dict:
			for rhs, prob in self.rule_dict[lhs].items():
				total += 1
				hrg = cv.get_orig_cfg(lhs, rhs)
				
				if prob > 0:
					#rules.append(("r" + str(count), lhs, rhs.split(), prob))
					rules.append(("r" + str(count), hrg, prob))
					count += 1
		print "number of original rules", str(total)
		print "number of reduced rules", str(count)
		return rules

	def get_valid_rule_count(self):
		# return number of rules with positive probability
		count = 0
		for lhs in self.rule_dict:
			for rhs, prob in self.rule_dict[lhs].items():
				if prob > 0:
					count += 1
		return count

class TreeNode(object):
	def __init__(self, val, children=None):
		self.val = val
		self.children = []
		self.term_count = 0
		self.counted = False
		self.num_nodes = -1

	def get_num_nodes(self):
		"""
		Get total number of nodes of subtree rooted at current node
		"""
		if self.num_nodes >= 0:
			return self.num_nodes
		if not self.children:
			self.num_nodes = 1
			return 1
		count = 1
		for c in self.children:
			count += c.get_num_nodes()
		self.num_nodes = count
		return count


	def print_tree(self):
		cur_level = deque([self])
		next_level = deque()
		while cur_level:
			cur = cur_level.popleft()
			sys.stdout.write(cur.val + " ")
			for child in cur.children:
				next_level.append(child)
			if not cur_level:
				sys.stdout.write("\n")
				cur_level = next_level
				next_level = deque()

	def get_term_count(self, term_count_dic, nonterm_count_dict, nonterm_size_dic):
		if self.val == 't':
			#count_dic[self.val] = count_dic.get(self.val, 0) + 1
			return 1
		nonterm_count_dict[self.val] = nonterm_count_dict.get(self.val, 0) + 1
		if self.counted:
			return self.term_count
		count = 0
		for c in self.children:
			count += c.get_term_count(term_count_dic, nonterm_count_dict, nonterm_size_dic)

		self.term_count = count
		self.counted = True

		term_count_dic[self.val] = term_count_dic.get(self.val, 0) + count

		if self.val not in nonterm_size_dic:
			nonterm_size_dic[self.val] = []
		nonterm_size_dic[self.val].append(count)

		return count

class ConvertRule(object):
	def __init__(self, tree_file, tree_count=-1):
		self.treefile = tree_file
		self.cfg_to_hrg_map = {}
		self.Tree = []

		self.nonterm = set(["S"])
		self.rule_dict = {}

		self.build_rules(tree_count)
		#self.Tree.print_tree()
		#print self.nonterm
		#print self.level_tree_nodes

	def hrg_to_cfg(self, lhs, rhs):
		if lhs != 'S':
			size = lhs.count(",") + 1
			lhs = 'N' + str(size)
			self.nonterm.add(lhs)

		t_symb = set()
		n_symb = []
		for r in rhs:
			if r.endswith(":N"):
				size = r.count(",") + 1
				n_symb.append('N' + str(size))
				self.nonterm.add('N' + str(size))
			for x in r.split(":")[0].split(","):
				if x.isdigit(): 
					t_symb.add(x)
		rhs = " ".join(n_symb) + " " + " ".join(["t" for i in xrange(len(t_symb))])
		#rhs = n_symb + ["t" for i in xrange(len(t_symb))]
		#print lhs, rhs
		return lhs, rhs.strip()

	def build_rules(self, tree_count):
		tree = TreeNode("S")
		stack = [tree]

		with open(self.treefile) as myfile:
			for line in myfile:
				rules = line.split()
				lhs, rhs = rules[0], rules[2]
				lhs = re.findall(r'\((.*?)\)', lhs)
				rhs = re.findall(r'\((.*?)\)', rhs)
				hrg = (lhs[0], rhs)
				#print lhs, rhs
				lhs, rhs = self.hrg_to_cfg(lhs[0], rhs)

				if not stack:
					self.Tree.append(tree)
					tree_count -= 1
					if tree_count == 0:
						break
					tree = TreeNode("S")
					stack = [tree]

				parent = stack.pop()
				
				add_stack = []
				rhs_list = []
				for r in rhs.split():
					n = TreeNode(r)
					if r != "t":
						add_stack.append(n)
					parent.children.append(n)
					rhs_list.append(n)

				add_stack.reverse()
				stack.extend(add_stack)

				# add to grammar
				if lhs not in self.rule_dict:
					self.rule_dict[lhs] = {}
				self.rule_dict[lhs][rhs] = self.rule_dict[lhs].get(rhs, 0) + 1

				self.cfg_to_hrg_map[lhs + "->" + rhs] = hrg
				#self.cfg_to_hrg_map[] = line
		if tree_count == 0:
			return
		self.Tree.append(tree)

	def get_orig_cfg(self, lhs, rhs):
		if lhs == "S":
			base_lhs, c_lhs = "S", -1
		else:
			base_lhs, c_lhs = lhs.split("_")[0], lhs.split("_")[1]
		base_rhs, c_rhs = [], []
		rhs = rhs.split()
		for t in rhs:
			if t != "t":
				t = t.split("_")
				base_rhs.append(t[0])
				c_rhs.append(t[1])
			else:
				base_rhs.append("t")
		orig_hrg = self.cfg_to_hrg_map[base_lhs + "->" + " ".join(base_rhs)]
		orig_hrg_lhs, orig_hrg_rhs = orig_hrg[0], orig_hrg[1]
		if c_lhs >= 0:
			hrg_lhs = orig_hrg_lhs + "_" + c_lhs
		else:
			hrg_lhs =  orig_hrg_lhs 
		hrg_rhs = []
		i = 0
		for t in orig_hrg_rhs:
			if t.endswith("N"):
				t = t + "_" + c_rhs[i]
				i += 1
			hrg_rhs.append(t)
		return (hrg_lhs, hrg_rhs)

def get_level_nodes(tree):
	i = 0
	level_tree_nodes = {}
	level_tree_nodes[i] = [tree]
	cur_level = deque([tree])
	next_level = deque()
	while cur_level:
		cur = cur_level.popleft()
		for child in cur.children:
			if child.val != "t":
				next_level.append(child)
		if not cur_level:
			i += 1
			level_tree_nodes[i] = [n for n in next_level]
			cur_level = next_level
			next_level = deque()
	return level_tree_nodes

class EM(object):
	def __init__(self, gram, tree, cur_str_result=None):
		self.gram = gram
		self.tree = tree
		self.loglikelihood = 0
		self.cur_str_result = cur_str_result	# record the print strings in a list
		for x in self.gram.rule_dict:
			r_rules = self.gram.rule_dict[x]
			for r in r_rules:
				self.gram.rule_dict[x][r] = np.log(self.gram.rule_dict[x][r])

	def get_nonterm_num(self, string):
		# get how many numbers the nonterminal contains
		# N4_1 returns 4
		toks = string.split("_")
		
		return int(toks[0][1])

	def get_loglikelihood(self, added_rules, result_file=None):
		# set added_rule probability to extremely small
		smooth_prob = -10000

		for r in added_rules:
			parts = r.split("->")
			lhs, rhs = parts[0], parts[1]
			self.gram.rule_dict[lhs][rhs] = smooth_prob
		self.loglikelihood = 0
		
		self.f_rules = {}
		for s in self.gram.rule_dict:
			self.f_rules[s] = {}
			for r in self.gram.rule_dict[s]:
				self.f_rules[s][r] = -float("inf")
		#use_added_rules = 0
		use_smooth_count = -float("inf")
		for tree in self.tree:
			self.inside = {}
			self.outside = {}

			level_tree_nodes = get_level_nodes(tree)
			level_tree_nodes = OrderedDict(sorted(level_tree_nodes.items(), reverse=True))
			for level in level_tree_nodes:
				for n in level_tree_nodes[level]:
					if n not in self.inside:
						self.inside[n] = {}
						#print n.val
					node_nonterm = set()
					if n.val == "S":
						node_nonterm.add("S")
					else:
						for i in xrange(1, self.gram.k+1):
							node_nonterm.add(n.val + "_" + str(i))
					for x in self.gram.alphabet:
						self.inside[n][x] = -float("inf")
						if x not in node_nonterm:
							continue
						for rhs, prob in self.gram.rule_dict[x].items():
							tmp_contribs = []
							rule = x + "->" + rhs
							rhs = rhs.split()
							l_children = len(n.children)
							if len(rhs) == l_children:
								tmp = prob
								tmp_contribs.append(prob)
								for ni, xi in zip(n.children, rhs):
									if ni.val == "t":
										if xi == "t":
											continue
										else:
											tmp = -float("inf")
											break
									tmp += self.inside[ni][xi]
									tmp_contribs.append(self.inside[ni][xi])
								self.inside[n][x] = np.logaddexp(tmp, self.inside[n][x])
								#print self.inside[n][x], tmp, np.logaddexp(tmp, self.inside[n][x]), tmp_contribs
			level_tree_nodes = OrderedDict(sorted(level_tree_nodes.items()))
			self.outside[level_tree_nodes[0][0]] = {}
	
			for x in self.gram.alphabet:
				self.outside[level_tree_nodes[0][0]][x] = -float("inf")
			self.outside[level_tree_nodes[0][0]][self.gram.start] = 0.0
	
			root_inside_likelihood = self.inside[level_tree_nodes[0][0]][self.gram.start]
			self.loglikelihood += root_inside_likelihood

			for level in level_tree_nodes:
				for n in level_tree_nodes[level]:
					node_nonterm = set()
					if n.val == "S":
						node_nonterm.add("S")
					else:
						for i in xrange(1, self.gram.k+1):
							node_nonterm.add(n.val + "_" + str(i))
					for x in self.gram.alphabet:
						if x not in node_nonterm:
							continue
						for rhs, prob in self.gram.rule_dict[x].items():
							rhs = rhs.split()
							#print product
							l_children = len(n.children)
							if len(rhs) == l_children:
								for ni, xi in zip(n.children, rhs):
									if ni not in self.outside:
										self.outside[ni] = {}
										for x_p in self.gram.alphabet:
											self.outside[ni][x_p] = - float("inf")
									#print self.inside[ni][xi], product, prob
									product = self.outside[n][x] + prob
									#print [c.val for c in n.children]
									for ni_p, xi_p in zip(n.children, rhs):
										if ni_p.val == "t":
											if xi_p == "t":
												continue
											else:
												product = - float("inf")
												break
										if ni != ni_p:
											product += self.inside[ni_p][xi_p]
									#self.outside[ni][xi] += self.outside[n][x] * prob * product / self.inside[ni][xi]
									#t = self.outside[n][x] + prob + product
									self.outside[ni][xi] = np.logaddexp(product, self.outside[ni][xi])
	
			# get expected counts
			for level in level_tree_nodes:
				for n in level_tree_nodes[level]:
					node_nonterm = set()
					if n.val == "S":
						node_nonterm.add("S")
					else:
						for i in xrange(1, self.gram.k+1):
							node_nonterm.add(n.val + "_" + str(i))
					for x in self.gram.alphabet:
						if x not in node_nonterm:
							continue
						for rhs, prob in self.gram.rule_dict[x].items():
							rhs = rhs.split()
	
							l_children = len(n.children)
							if len(rhs) == l_children:
								tmp = prob + self.outside[n][x]
								for ni, xi in zip(n.children, rhs):
									if ni.val == "t":
										if xi == "t":
											continue
										else:
											tmp = -float("inf")
											break
									tmp += self.inside[ni][xi]
								tmp -= root_inside_likelihood

								self.f_rules[x][" ".join(rhs)] = np.logaddexp(tmp, self.f_rules[x][" ".join(rhs)])
		use_total_count = -float("inf")
		for lhs in self.f_rules:
			for rhs in self.f_rules[lhs]:
				use_total_count = np.logaddexp(use_total_count, self.f_rules[lhs][rhs])
				if lhs + "->" + rhs in added_rules:
					use_smooth_count = np.logaddexp(use_smooth_count, self.f_rules[lhs][rhs])
		use_smooth_count = np.exp(use_smooth_count)
		use_total_count = np.exp(use_total_count)
		print self.loglikelihood, self.loglikelihood-use_smooth_count * smooth_prob, use_smooth_count, use_total_count, use_smooth_count / use_total_count
		if result_file:
			result_file.write("%s %s %s %s %s\n" % (str(self.loglikelihood), str(self.loglikelihood-use_smooth_count * smooth_prob), str(use_smooth_count), str(use_total_count), str(use_smooth_count / use_total_count)))
		self.loglikelihood -= use_smooth_count * smooth_prob
		return use_smooth_count

	def expect(self, tree):
		self.inside = {}
		self.outside = {}

		level_tree_nodes = get_level_nodes(tree)
		level_tree_nodes = OrderedDict(sorted(level_tree_nodes.items(), reverse=True))
		for level in level_tree_nodes:
			for n in level_tree_nodes[level]:
				if n not in self.inside:
					self.inside[n] = {}
					#print n.val
				node_nonterm = set()
				if n.val == "S":
					node_nonterm.add("S")
				else:
					for i in xrange(1, self.gram.k+1):
						node_nonterm.add(n.val + "_" + str(i))
				for x in self.gram.alphabet:
					self.inside[n][x] = -float("inf")
					if x not in node_nonterm:
						continue
					
					for rhs, prob in self.gram.rule_dict[x].items():
						rhs = rhs.split()
						l_children = len(n.children)
						if len(rhs) == l_children:
							tmp = prob
							for ni, xi in zip(n.children, rhs):
								if ni.val == "t":
									if xi == "t":
										continue
									else:
										tmp = -float("inf")
										break
								tmp += self.inside[ni][xi]
							self.inside[n][x] = np.logaddexp(tmp, self.inside[n][x])

		level_tree_nodes = OrderedDict(sorted(level_tree_nodes.items()))
		self.outside[level_tree_nodes[0][0]] = {}

		for x in self.gram.alphabet:
			self.outside[level_tree_nodes[0][0]][x] = -float("inf")
		self.outside[level_tree_nodes[0][0]][self.gram.start] = 0.0

		root_inside_likelihood = self.inside[level_tree_nodes[0][0]][self.gram.start]
		self.loglikelihood += root_inside_likelihood

		for level in level_tree_nodes:
			for n in level_tree_nodes[level]:
				
				node_nonterm = set()
				if n.val == "S":
					node_nonterm.add("S")
				else:
					for i in xrange(1, self.gram.k+1):
						node_nonterm.add(n.val + "_" + str(i))
				for x in self.gram.alphabet:
					if x not in node_nonterm:
						continue
					for rhs, prob in self.gram.rule_dict[x].items():
						rhs = rhs.split()
						#print product
						l_children = len(n.children)
						if len(rhs) == l_children:
							for ni, xi in zip(n.children, rhs):
								if ni not in self.outside:
									self.outside[ni] = {}
									for x_p in self.gram.alphabet:
										self.outside[ni][x_p] = - float("inf")
								#print self.inside[ni][xi], product, prob
								product = self.outside[n][x] + prob
								#print [c.val for c in n.children]
								for ni_p, xi_p in zip(n.children, rhs):
									if ni_p.val == "t":
										if xi_p == "t":
											continue
										else:
											product = - float("inf")
											break
									if ni != ni_p:
										product += self.inside[ni_p][xi_p]
								#self.outside[ni][xi] += self.outside[n][x] * prob * product / self.inside[ni][xi]
								#t = self.outside[n][x] + prob + product
								self.outside[ni][xi] = np.logaddexp(product, self.outside[ni][xi])

		# get expected counts
		for level in level_tree_nodes:
			for n in level_tree_nodes[level]:
				node_nonterm = set()
				if n.val == "S":
					node_nonterm.add("S")
				else:
					for i in xrange(1, self.gram.k+1):
						node_nonterm.add(n.val + "_" + str(i))

				
				for x in self.gram.alphabet:
					if x not in node_nonterm:
						continue
					for rhs, prob in self.gram.rule_dict[x].items():
						rhs = rhs.split()

						l_children = len(n.children)
						if len(rhs) == l_children:
							tmp = prob + self.outside[n][x]
							for ni, xi in zip(n.children, rhs):
								if ni.val == "t":
									if xi == "t":
										continue
									else:
										tmp = -float("inf")
										break
								tmp += self.inside[ni][xi]
							tmp -= root_inside_likelihood
							self.f_rules[x][" ".join(rhs)] = np.logaddexp(tmp, self.f_rules[x][" ".join(rhs)])

	def maximize(self):
		for x in self.gram.rule_dict:
			r_rules = self.gram.rule_dict[x]
			r_expects = self.f_rules[x]
			sum_x = None
			for s in r_expects:
				if sum_x is None:
					sum_x = r_expects[s]
				else:
					sum_x = np.logaddexp(sum_x, r_expects[s])

			for r in r_rules:
				self.gram.rule_dict[x][r] = r_expects[r] - sum_x

	def iterations(self, use_converge, iteration=10,  converge=1):
		if use_converge:
			prev_loglikelihood = float("-inf")
			while True:
				self.f_rules = {}
				for s in self.gram.rule_dict:
					self.f_rules[s] = {}
					for r in self.gram.rule_dict[s]:
						self.f_rules[s][r] = -float("inf")
				self.loglikelihood = 0
	
				for tree in self.tree:
					self.expect(tree)
				self.maximize()
				if not self.cur_str_result is None:
					self.cur_str_result.append( "log likelihood: " + str(self.loglikelihood / len(self.tree)))
				if self.loglikelihood / len(self.tree) - prev_loglikelihood < converge:
					self.loglikelihood = self.loglikelihood / len(self.tree)
					break
				prev_loglikelihood = self.loglikelihood / len(self.tree)
		else:
			for i in xrange(iteration):
				self.f_rules = {}
				for s in self.gram.rule_dict:
					self.f_rules[s] = {}
					for r in self.gram.rule_dict[s]:
						self.f_rules[s][r] = -float("inf")
				self.loglikelihood = 0
	
				for tree in self.tree:
					self.expect(tree)
				self.maximize()
				if not self.cur_str_result is None:
					self.cur_str_result.append( "log likelihood: "+ str(self.loglikelihood / len(self.tree)))

		for x in self.gram.rule_dict:
			r_rules = self.gram.rule_dict[x]
			delete_rules = set()
			for r in r_rules:
				self.gram.rule_dict[x][r] = np.exp(self.gram.rule_dict[x][r])
				# remove useless rules
				if self.gram.rule_dict[x][r] == 0:
					delete_rules.add(r)
			for r in delete_rules:
				del self.gram.rule_dict[x][r]
		#print "tree nodes: ", self.tree[0].get_num_nodes()
		#print "grammar size: ", self.gram.get_valid_rule_count()
		# get BIC = -2 * L + k * ln(N)
		#bic = -2 * self.loglikelihood + self.gram.get_valid_rule_count() * np.log(self.tree[0].get_num_nodes())
		#print "bic: ", bic
		#aic = -2 * self.loglikelihood + 2 * self.gram.get_valid_rule_count()
		#print "aic: ", aic

def plot_nonterm_stats(nonterm_size_dic):
	nonterm_groups = {}
	for n in nonterm_ave_size_dic.keys():
		base = n.split("_")[0]
		if not base in nonterm_groups:
			nonterm_groups[base] = []
		nonterm_groups[base].append(n)

	for base in nonterm_groups:
		nonterms = nonterm_groups[base]
		for n in nonterms:
			print (n, "mean: ", np.mean(nonterm_size_dic[n]), "std: ", np.std(nonterm_size_dic[n]))
			print nonterm_size_dic[n]

def choice(values, p):
	total = 0
	cum_weights = []
	for w in p:
		total += w
		cum_weights.append(total)
	x = random() * total
	i = bisect(cum_weights, x)
	return values[i]

def grow_graph_with_root(grammar_dict, root_symbol, out_file):
	graph_rules = []
	graph_file = open(out_file, "w")

	queue = deque([root_symbol])
	while queue:
		lhs = queue.popleft()
		rhs = choice(grammar_dict[lhs].keys(), grammar_dict[lhs].values())
		#print (rhs)
		for t in rhs[:-1]:
			if not t.endswith("T"):
				toks = t.split("_")
				queue.append("N" + str(len(toks[0].split(","))) + "_" + toks[1])
		graph_rules.append((lhs, rhs, grammar_dict[lhs][rhs]))

	k = 0
	for lhs, rhs, prob in graph_rules:
		if k == 0:
			new_lhs = lhs
			graph_file.write("(%s) -> " % (new_lhs))
			k += 1
		else:
			node_count = int(lhs.split("_")[0][1:])
			nonterm_idx = lhs.split("_")[1]
			new_lhs = []
			for i in range(node_count):
				new_lhs.append(chr(ord('a') + i))
			new_lhs = ",".join(new_lhs)
			graph_file.write("(%s)_%s -> " % (new_lhs, nonterm_idx))
		for t in rhs[:-1]:
			graph_file.write("(%s)" % t)
		graph_file.write(" " + rhs[-1])
		graph_file.write("\n")

	graph_file.close()

def grow_nonterminal_graphs(grammar, out_dir):
	# convert grammar format
	grammar_dict = {}
	nonterms = set()
	versions = set()
	for id, rules, prob in grammar:
		lhs, rhs = rules[0], rules[1]
		rhs.append(id)
		if lhs == '(S)':
			new_lhs = 'S'
		else:
			toks = lhs.split("_")
			n_type, v_type = str(len(toks[0].split(","))), toks[1]
			new_lhs = "N" + n_type + "_" + v_type
			nonterms.add(n_type)
			versions.add(v_type)
		if new_lhs not in grammar_dict:
			grammar_dict[new_lhs] = {}
		grammar_dict[new_lhs][tuple(rhs)] = prob
	for i in range(1, len(versions) + 1):
		for j in range(1, len(nonterms) + 1):
			grow_graph_with_root(grammar_dict, 'N%d_%d' % (j, i), '%s/N%d_%d.txt' % (out_dir, j, i))

def grow_graph(grammar):
	grammar_dict = {}
	graph_rules = []
	graph_file = open("graph.txt", "w")
	for id, rules, prob in grammar:
		lhs, rhs = rules[0], rules[1]
		rhs.append(id)
		if lhs == '(S)':
			new_lhs = 'S'
		else:
			toks = lhs.split("_")
			new_lhs = "N" + str(len(toks[0].split(","))) + "_" + toks[1]
		if new_lhs not in grammar_dict:
			grammar_dict[new_lhs] = {}
		grammar_dict[new_lhs][tuple(rhs)] = prob

	queue = deque("S")
	while queue:
		lhs = queue.popleft()
		rhs = choice(grammar_dict[lhs].keys(), grammar_dict[lhs].values())
		for t in rhs[:-1]:
			if not t.endswith("T"):
				toks = t.split("_")
				queue.append("N" + str(len(toks[0].split(","))) + "_" + toks[1])
		graph_rules.append((lhs, rhs, grammar_dict[lhs][rhs]))
	#print graph_rules
	for lhs, rhs, prob in graph_rules:
		if lhs == 'S':
			new_lhs = lhs
			graph_file.write("(%s) -> " % (new_lhs))
		else:
			node_count = int(lhs.split("_")[0][1])
			nonterm_idx = lhs.split("_")[1]
			new_lhs = []
			for i in range(node_count):
				new_lhs.append(chr(ord('a') + i))
			new_lhs = ",".join(new_lhs)
			graph_file.write("(%s)_%s -> " % (new_lhs, nonterm_idx))
		for t in rhs[:-1]:
			graph_file.write("(%s)" % t)
			#if t.endswith("T"):
			#	graph_file.write("(%s)" % t)
			#else:
			#	t = t.split(":")
			#	graph_file.write("(%s:%s)" % (t[0], "N"))
		graph_file.write(" " + rhs[-1])
		graph_file.write("\n")

	graph_file.close()


if __name__ == "__main__":
	cv = ConvertRule("data/enron_left_derive.txt")
	#for tree in cv.tree_list:
	#	tree.print_tree()
	gram = Grammar(cv.rule_dict, 2)
	#gram.printAllRules()
	
	#cv.Tree.print_tree()
	em = EM(gram, cv.Tree)
	em.iterations(100)

	# sample graph size
	graph_size_counts = []
	rules = []
	
	nonterm_ave_size_dic = {}
	nonterm_size_dic = {}
	for i in xrange(100):
		term_count_dict = {}
		nonterm_count_dict = {}
		rules, term_count, tree = em.gram.sample()
		tree.get_term_count(term_count_dict, nonterm_count_dict, nonterm_size_dic)
		for t in term_count_dict:
			if t not in nonterm_ave_size_dic:
				nonterm_ave_size_dic[t] = []
			nonterm_ave_size_dic[t].append(term_count_dict[t] / float(nonterm_count_dict[t]))
		graph_size_counts.append(term_count)

	#plot_nonterm_stats(nonterm_ave_size_dic)
	#plot_nonterm_stats(nonterm_size_dic)

	graph_size_counts.sort()
	print "graph size mean: ", np.mean(graph_size_counts), "graph size std: ", np.std(graph_size_counts)

  	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('size of tree')
	ax.set_ylabel('count')
	ax.hist(graph_size_counts, bins=[i for i in xrange(100)])

	grammar = em.gram.get_valid_rules(cv)

	#for r in grammar:
	#	print r 

	grow_graph(grammar)


	grammar = em.gram.get_valid_rules(cv)
	grow_nonterminal_graphs(grammar, "out_graphs")
	#visualize.dir_node_count("out_graphs")
	#plt.show()

	
