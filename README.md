# Uni Mannheim - Business Informatics Masters Thesis - Xuehui Jiang
This is a public reposity for my master thesis "Enhancing RDF2vec with Spreading Activation for Improved Knowledge Graph Embedding" for University of Mannheim Master in Business Informatics, Chair of Data and Web Science.

1. The main Constrained Spreading Activation algorithm with BFS or DFS exist in python script called `main.py`.
2. The python script called `sa_helper.py` contain output function, input function and activation function along with the calculate_excl function and calculate_pop function, which assist the input function.
3. `Grid Search for Parameter Turing.ipynb` exhibites the evaluation metrics of the original AIFB dataset and baseline, as well as illustrates the process of utilizing the Grid Search method for parameter tuning.
4. `Experiments_on_Parameters_BFS.ipynb` and `Experiments_on_Parameters_DFS.ipynb` conduct the parameter research and parameter optimation experiment when respectively using BFS and DFS in Spreading Activation approach. Their results are downloaded as .csv files and stored in file `parameter_experiment_results_BFS` and `parameter_experiment_results_DFS`, respectively.
5. In `Experimental_Evaluation` file, there are pyrhon notebooks on each datasets for classification and regression experiments, in addition to pyrhon notebooks for entity relatedness nad document similarity. The datasets we used in the experiments and the obtained results are also ivolved.

Please note that the complete English vesion of 2016-10 DBpedia dataset is too large and exceeds the compution capability of our equipment. In order to obtain the DBpedia knowledge graph related to the entities in the dataset, we tried two methods. One is to use the SPARQL endpoint provided by DBpedia to execute the query, retrieving direct types from the ontology, as well as outgoing and ingoing edges (Please refer to python notebook `Experiment_Cities.ipynb`). But the The quality of the queried knowlege graph is not good enough for our experiments. The second method, which is the method we finally adopted to obtain the knowledge graph, is to download important files directly. Given that in our evaluation, we only consider object properties, and ignore datatype properties and literals, we read each file and retain only those triples that are neither datatype properties nor literals.

For files that describe core relationships and entity types, we choose to filter directly for triples related to entities in the Cities Dataset. For files that may provide more context and indirect relationships, we use a recursive approach to get entities and relationships that are directly or indirectly related to the target entity.


