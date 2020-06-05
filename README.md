# Torch RGCN: Reproducing the Relational Graph Convolutional Network 

A PyTorch implementation of the Relational Graph Convolutional Network (R-GCN). [#TODO: Expand on this.]

Original Paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) 

### Datasets

#### Link Prediction:

Original Link Prediction Implementation: https://github.com/MichSchli/RelationPrediction 

 * AIFB **(cite original source)**
 * MUTAG **(cite original source)**
 * BGS **(cite original source)**
 * AM **(cite original source)** 

#### Node Classification:

Original Node Classification Implementation: https://github.com/tkipf/relational-gcn
 
 * WN18 **(cite original source)**
 * WN18RR **(cite original source)**
 * FB15K **(cite original source)**
 * FB15K-237 **(cite original source)**

### Getting started

[ #TODO: WRITE SOME SETUP INSTRUCTIONS ]
 
[ #TODO: TEST SETUP INSTRUCTIONS ] 

Requirements: 
* Conda >= 4.8
* Python >= 3.7

To create a virtual environment and install the necessary packages use:

`bash setup.sh`

Now, you may install the RGCN module with: 

`pip install -e .`

### Usage

#### Configuration files

The hyper-parameters for the different experiments can be found in the configurations files under 
[configurations](configurations).


[#TODO: Update configuration files after hyper-parameter tuning]

#### Link Prediction

Run link prediction experiments using:

`python experiments/predict_links.py with configurations/lp-FB15k.yaml`

`python experiments/predict_links.py with configurations/lp-FB15k-237.yaml`

`python experiments/predict_links.py with configurations/lp-WN18.yaml`

`python experiments/predict_links.py with configurations/lp-WN18RR.yaml`

#### Node Classification

Run node classification experiments using:

`python experiments/classify_nodes.py with configurations/nc-AIFB.yaml`

`python experiments/classify_nodes.py with configurations/nc-AM.yaml`

`python experiments/classify_nodes.py with configurations/nc-BGS.yaml`

`python experiments/classify_nodes.py with configurations/nc-MUTAG.yaml`

### Unit test 

[#TODO: Decide whether to keep this section or not]

To run unit tests for neural network util function use:

`pytest tests/test_nn.py`

To run unit tests to test utils function use:

`pytest tests/test_utils.py`

### Model Performance

#### Link Prediction

| Dataset                       | Mean Reciprocal Rank (filtered)  | Hits@1 (filtered)      | Hits@3 (filtered)       | Hits@10 (filtered)      |
| ----------------------------- |:--------------------------------:|:----------------------:|:-----------------------:|:-----------------------:|
| FB15k                         | -                                | -                      | -                       | -                       |
| FB15k-237                     | -                                | -                      | -                       | -                       |
| WN18                          | -                                | -                      | -                       | -                       |
| WN18RR                        | -                                | -                      | -                       | -                       |

#### Node Classification

| Dataset                       | Accuracy                         |
| ----------------------------- |:--------------------------------:|
| AIFB                          | -                                |
| AM                            | -                                |
| BGS                           | -                                |
| MUTAG                         | -                                |

### Cite as 

[ #TODO: INSERT PAPER CITATION ]

```
[ BIBTEX HERE ]
```