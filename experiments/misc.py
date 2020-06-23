from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
import random
import torch
import os


def create_experiment(name='exp', database=None):
    """ Create Scared experiment object for experiment logging """
    ex = Experiment(name)

    atlas_user = os.environ.get('MONGO_DB_USER')
    atlas_password = os.environ.get('MONGO_DB_PASS')
    atlas_host = os.environ.get('MONGO_DB_HOST')

    # Add remote MongoDB observer, only if environment variables are set
    if atlas_user and atlas_password and atlas_host:
        ex.observers.append(MongoObserver(
            url=f"mongodb+srv://{atlas_user}:{atlas_password}@{atlas_host}",
            db_name=database))
    return ex

#######################################################################################################################
# Link Prediction Utils
#######################################################################################################################

def negative_sampling(positive_triples, entity_dictionary, neg_sample_rate):
    """ Generates a set of negative samples by corrupting triples """

    all_triples = np.array(positive_triples)
    s = np.resize(all_triples[:, 0], (len(positive_triples)*neg_sample_rate,))
    p = np.resize(all_triples[:, 1], (len(positive_triples)*neg_sample_rate,))
    o = np.random.randint(low=0, high=len(entity_dictionary), size=(len(positive_triples)*neg_sample_rate,))
    negative_triples = np.stack([s, p, o], axis=1)

    return negative_triples.tolist()


def perturb(test_triple, entity_dictionary, device='cpu'):
    """
    Generate candidates by replacing the tail with every entity for each test triplet
    """
    entites = list(entity_dictionary.keys())
    num_entities = len(entites)
    s, p, _ = test_triple

    s = torch.tensor([s], dtype=torch.long, device=device).repeat(num_entities, 1)
    p = torch.tensor([p], dtype=torch.long, device=device).repeat(num_entities, 1)
    o = torch.tensor(entites).view(num_entities, -1)

    candidates = torch.cat([s, p, o], dim=1)
    return candidates

def rank(results):
    """ Ranks the results according to the scores according to the probabilities"""
    sorted_results = results

    # Sort list according in descending order of probabilities

    return sorted_results

def filter_triples(candidate_triples, all_triples):
    """ Filter triples that are present in the train, validation and test split """
    return [ triple for triple in candidate_triples if not triple in all_triples ]

def compute_mrr(triples, correct_answer):
    """ Compute Mean Reciprocal Rank value for a particular ranked list """
    mrr = 0
    return mrr

def compute_hits(triples, correct_answer, k):
    """ Compute Hits@k value for a particular ranked list """
    hits = 0
    return hits

def compute_scores(scored_candidates, all_triples, correct_answer, filter=True, k=None):
    """ Returns MRR and Hits@k (k=1,3,10) values for a given triple prediction """
    if k is None:
        k = [1, 3, 10]

    ranked_triples = rank(scored_candidates)

    if filter:
        ranked_triples = filter_triples(ranked_triples, all_triples)

    mrr = compute_mrr(ranked_triples, correct_answer)
    hits_at_k = dict()
    for i in k:
        hits_at_k[k] = compute_hits(ranked_triples, correct_answer, i)

    return mrr, hits_at_k

#######################################################################################################################
# Node classification Utils
#######################################################################################################################

