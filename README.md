# Torch RGCN 

This is a PyTorch implementation of the Relational Graph Convolutional Network (RGCN), a graph embedding network 
proposed by by Schlichtkrull **et al.** 
[Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103).
 
In our recent workshop paper, we reproduce the link prediction and node classification experiments from the original 
paper and furthermore, present four RGCN extensions that address the shortcomings of typical GCNs.
 
Our workshop paper: 
[Reproducing the Relational Graph Convolutional Network: Lessons We Learned!](https://www.overleaf.com/read/hnvnwhhhtxrt) 

[#TODO: Upload paper to Arxiv]

## Getting started

Requirements: 
* Conda >= 4.8
* Python >= 3.7

To install the necessary packages and download all datasets use:

`bash setup_dependencies.sh`

Now, you may install the RGCN module with: 

`pip install -e .`

## Usage

### Configuration files

The hyper-parameters for the different experiments can be found in the configurations files under 
[configs](configs). The naming convention of the files is as follows: 

`{MODEL}-{EXPERIMENT}-{DATASET}.yaml`

### Models
* `standard`
* `compression`
* `embedding`
* `global`
* `overparam`

### Experiments
* `lp` (link prediction)
* `nc` (node classification) 

[ #TODO: Come up with better names for these models! ]

[#TODO: Update configuration files after hyper-parameter tuning]

### Datasets

#### Link Prediction

 * `AIFB` from 
 **Stephan Bloehdorn and York Sure. 
 [Kernel methods for mining instance data in ontologies.](https://link.springer.com/content/pdf/10.1007%2F978-3-540-76298-0_5.pdf) 
 In The Semantic Web, 6th International Semantic Web Conference,  2007.** 
 * `MUTAG` from 
 **A. K. Debnath, R. L. Lopez de Compadre, G. Debnath, A. J.Shusterman, and C. Hansch. 
 [Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro-compounds correlation 
 with molecular orbital energies and hydrophobicity.](https://pubs.acs.org/doi/pdf/10.1021/jm00106a046?casa_token=ECo0FUp3gNoAAAAA:6Xgkt3vGuQeVFnGwlPlyDWm-fIflRmsRe7s5X_SH143O4-wVz5eIMHj_cmDvBWCVon6LLvVt0nTgy-4) 
 J Med Chem,34:786–797, 1991.**
 * `BGS` from 
 **de Vries, G.K.D.
 [A fast approximation of the Weisfeiler-Lehman graph kernel for RDF data.](https://link.springer.com/content/pdf/10.1007%2F978-3-642-40988-2_39.pdf) 
 In European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. 2013.**
 * `AM` from 
**de Boer, V., Wielemaker, J., van Gent, J., Hildebrand, M., Isaac, A., van Ossen-bruggen, J., Schreiber, G.
[Supporting linked data production for cultural heritageinstitutes: The amsterdam museum case study.](https://link.springer.com/content/pdf/10.1007%2F978-3-642-30284-8_56.pdf) 
In The Semantic Web: Research and Applications. 2012.**

#### Node Classification
 
 * `WN18` from 
 **Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran , Jason Weston, and Oksana Yakhnenko. 
 [Translating embeddings for modeling multi-relational data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela).
 In Advances in Neural Information Processing Systems, 2013.** 
 * `WN18RR` from 
 **Tim  Dettmers,  Pasquale  Minervini,  Pontus  Stenetorp,  and  Sebastian  Riedel. 
 [Convolutional  2D knowledge graph embeddings](https://arxiv.org/abs/1707.01476).  
 In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI), 2018.** 
 * `FB15K` from 
 **Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran , Jason Weston, and Oksana Yakhnenko. 
 [Translating embeddings for modeling multi-relational data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela).
 In Advances in Neural Information Processing Systems, 2013.** 
 * `FB15K-237` from 
 **Kristina Toutanova and Danqi Chen.  
 [Observed versus latent features for knowledge base and text inference](https://www.aclweb.org/anthology/W15-4007.pdf).
 In Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality (CVSC@ACL), 2015.**

[#TODO: Double check if these citations are correct. Check formatting.]

## Part 1: Experiment Reproduction  

### Link Prediction

Original Link Prediction Implementation: https://github.com/MichSchli/RelationPrediction 

Run link prediction using the standard RGCN model using:

`python experiments/predict_links.py with configs/standard-lp-{DATASET}.yaml`

Make sure to replace `{DATASET}` with one of the following dataset names: `FB15k`, `FB15k-237`, `WN18` or `WN18RR`.

### Node Classification

Original Node Classification Implementation: https://github.com/tkipf/relational-gcn

Run node classification using the standard RGCN model using:

`python experiments/classify_nodes.py with configs/standard-nc-{DATASET}.yaml`

Make sure to replace `{DATASET}` with one of the following dataset names: `AIFB`, `MUTAG`, `BGS` or `AM`.

## Part 2: New Configurations 

### RGCN with Node Embeddings 

Typically, the RGCN takes in a feature matrix consisting of one-hot encoded vectors which indicate the presence or 
absence of a particular node feature. Replacing the one-hot encoded vectors with rich node embeddings would (in theory) 
help to produce better latent graph representations. 

To run the experiment use: 

`python experiments/predict_links.py with configs/embedding-lp-{DATASET}.yaml`

Make sure to replace `{DATASET}` with one of the following dataset names: `FB15k`, `FB15k-237`, `WN18` or `WN18RR`.

### RGCN with Compressed inputs 

It is common knowledge that GNNs do not scale well to large graphs because they require large memory allocation. This 
problem is more significant for RGCNs due to the high-dimensional tensors required to represent the diverse types of 
edges and nodes. In this model, we compress the input graph prior to feeding it into an RGCN and expand the resulting 
embedding back to a high-dimensional vectors. This model would be similar a bottleneck architecture.  

[ #TODO: For this experiment, we can also make use of Wikidata-5M dataset]

To run the experiment use: 

`python experiments/predict_links.py with configs/compression-lp-{DATASET}.yaml`

Make sure to replace `{DATASET}` with one of the following dataset names: `FB15k`, `FB15k-237`, `WN18` or `WN18RR`.

### RGCN with Global Readouts

A common drawback of GCNs, including RGCNs, is that they only consider local structure during message propagation. 
With global feature computation, such as [global attribute](https://openreview.net/pdf?id=r1lZ7AEKvB#cite.all2018graphs) 
and [global readout](https://openreview.net/pdf?id=r1lZ7AEKvB), the RGCN should be able to incorporate global 
information about the graph into ever layer of the GNN.

To run the experiment use: 

`python experiments/predict_links.py with configs/global-lp-{DATASET}.yaml`

Make sure to replace `{DATASET}` with one of the following dataset names: `FB15k`, `FB15k-237`, `WN18` or `WN18RR`.

### Overparameterising the RGCN

This experiment will test the [deep double descent phenomena](https://arxiv.org/abs/1912.02292) that is typically 
observed in large CNNs. This can be simply done using the stanard model by increasing the number of blocks (in block diagonal
decomposition) or number of bases (in basis decomposition) to a value greater than the number of relations.

To run the experiment use: 

`python experiments/predict_links.py with configs/overparam-lp-{DATASET}.yaml`

Make sure to replace `{DATASET}` with one of the following dataset names: `FB15k`, `FB15k-237`, `WN18` or `WN18RR`.

## Model Performance

### Part 1: Reproduction Experiments

#### Link Prediction using standard RGCN

| Dataset                       | Mean Reciprocal Rank (filtered)  | Hits@1 (filtered)      | Hits@3 (filtered)       | Hits@10 (filtered)      |
| ----------------------------- |:--------------------------------:|:----------------------:|:-----------------------:|:-----------------------:|
| FB15k                         | -                                | -                      | -                       | -                       |
| FB15k-237                     | -                                | -                      | -                       | -                       |
| WN18                          | -                                | -                      | -                       | -                       |
| WN18RR                        | -                                | -                      | -                       | -                       |

#### Node Classification using standard RGCN

| Dataset                       | Accuracy                         |
| ----------------------------- |:--------------------------------:|
| AIFB                          | -                                |
| AM                            | -                                |
| BGS                           | -                                |
| MUTAG                         | -                                |

### Part 2: New Configurations

Here are the link prediction results for the different models.

#### Embedding 

| Dataset                       | Mean Reciprocal Rank (filtered)  | Hits@1 (filtered)      | Hits@3 (filtered)       | Hits@10 (filtered)      |
| ----------------------------- |:--------------------------------:|:----------------------:|:-----------------------:|:-----------------------:|
| FB15k                         | -                                | -                      | -                       | -                       |
| FB15k-237                     | -                                | -                      | -                       | -                       |
| WN18                          | -                                | -                      | -                       | -                       |
| WN18RR                        | -                                | -                      | -                       | -                       |

#### Compression (Bottleneck Architecture) 

| Dataset                       | Mean Reciprocal Rank (filtered)  | Hits@1 (filtered)      | Hits@3 (filtered)       | Hits@10 (filtered)      |
| ----------------------------- |:--------------------------------:|:----------------------:|:-----------------------:|:-----------------------:|
| FB15k                         | -                                | -                      | -                       | -                       |
| FB15k-237                     | -                                | -                      | -                       | -                       |
| WN18                          | -                                | -                      | -                       | -                       |
| WN18RR                        | -                                | -                      | -                       | -                       |
| Wikidata 5M                   | -                                | -                      | -                       | -                       |

#### Global Readout 

| Dataset                       | Mean Reciprocal Rank (filtered)  | Hits@1 (filtered)      | Hits@3 (filtered)       | Hits@10 (filtered)      |
| ----------------------------- |:--------------------------------:|:----------------------:|:-----------------------:|:-----------------------:|
| FB15k                         | -                                | -                      | -                       | -                       |
| FB15k-237                     | -                                | -                      | -                       | -                       |
| WN18                          | -                                | -                      | -                       | -                       |
| WN18RR                        | -                                | -                      | -                       | -                       |

## Unit Tests 

To run all unit tests use:
 
`pytest tests/test_nn.py`

To run unit tests for neural network util function use:

`pytest tests/test_nn.py`

To run unit tests for utils functions use:

`pytest tests/test_utils.py`

To run unit tests for misc functions use:

`pytest tests/test_misc.py`

## Citing Our Code 

If you use our implementation in your own work, you may cite our paper as

```
[ #TODO: INSERT PAPER BIBTEX CITATION ]
```