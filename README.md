SRW
===

Implementation of Supervised Random Walks (SRW) for link prediction originally proposed by Backstrom and Leskovec (see doc/linkpred-wsdm11.pdf).

The main idea of (SRW) is to guide the personalized PageRank in the direction of nodes that are more likely to be linked to in the future.
Whereas plain PageRank picks the outgoing edge to be followed from a uniform distribution, the SRW method learns a vector of parameters for a function that assigns each edge a particular weight.

As in the original paper, L-BFGS was used for optimization. L-BFGS is implemented in Alglib (http://www.alglib.net) for C++.


Building
===
This is not a ready to build project. Additional code for graph representation must be added in order to build and run it. Also the features extraction code can be modified to select the features you need.

First compile the files in alglib directory into object (.o) files. These compiles files will be linked later to the main program.

You must provide a custom implementation of a Graph class. 
It must support the following:
iterate_outgoing_edges() -- returns a Graph::iterator object.
Graph::iterator has the following fields:
v2 -- neighbouring node
data -- timestamp when the connection was created
Graph is loaded with load_graph_file() function that must also be implemented.

The implementation relies on OpenMP for parallelization of some code parts.

