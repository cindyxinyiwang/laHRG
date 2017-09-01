import os

subgraph_size = [25]
train_samples = [500]
#graph_name_size = {"ba": 25 , "twws": 25, "cit-HepTh": 25, "as-train-topo": 25, "amazon": 25, "wiki": 25}
graph_name_size = {"as": 25}

split_list = [2]

str = "#! /bin/bash\n source ../../HRG/hrgenv/bin/activate\n python generate_graphs.py %s %d %d"

for graph_name, graph_size in graph_name_size.iteritems():
	for split in split_list:
		script_name = "results_final/generate_%s_split%d.sh" % (graph_name, split)
		script_file = open(script_name, 'w')
		script_file.write(str % (graph_name, graph_size, split))
		script_file.close()
		job_name = "%s-generate-%d" % (graph_name, split)
		os.system('qsub -o results_final/%s.out -N %s %s' % (job_name, job_name, script_name))
