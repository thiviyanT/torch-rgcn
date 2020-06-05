from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
import random
import torch
import os


def create_experiment(name='exp', database=None):
    """ Create Scared experiment object for logging"""
    ex = Experiment(name)

    # Add a remote MongoDB observer
    if database is not None:
        atlas_user = os.environ.get('ATLAS_USER')
        atlas_password = os.environ.get('ATLAS_PASSWORD')
        atlas_host = os.environ.get('ATLAS_HOST')

        assert (atlas_user and atlas_password and atlas_host), \
            "Cannot add sacred observer! The following environment variables must be set: " \
            "'ATLAS_USER', 'ATLAS_PASSWORD' and 'ATLAS_HOST'"

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


def perturb(test_triples, entity_dictionary):
    """
    Generate candidates for link prediction by replacing the tail with every entity in the dataset for each triplet
    in the test dataset
    """
    entites_list = list(entity_dictionary.keys())

    candidates = list()
    for test_triple in test_triples:
        s, p, o = test_triple

        # TODO Write this as a list comprehension for performance benefits - process negatives first and then add true triple
        for entity in entites_list:
            # Label the correct answer with 1 if the triple holds true and 0 if otherwise
            if o == entity:
                ground_truth = 1
            else:
                ground_truth = 0

            candidates.append([s, p, entity, ground_truth])

    return candidates

def rank(results):
    """ Ranks the results according to the scores according to the probabilities"""
    sorted_results = 0

    # Sort list according in descending order of probabilities

    return sorted_results

def filter_triples(candidate_triples, all_triples):
    """ Filter triples that are present in the train, validation and test split """
    filtered_triples = list()

    # TODO: Write this for loop as a list comprehension
    for triple in candidate_triples:
        if not triple in all_triples:
            filtered_triples.append(triple)

    return filtered_triples

def compute_mrr(triples, correct_answer):
    """ Compute Mean Reciprocal Rank value for a particular ranked list """
    mrr = 0
    return mrr

def compute_hits(triples, correct_answer, k):
    """ Compute Hits@k value for a particular ranked list """
    hits = 0
    return hits

def compute_scores(triples, train, val, test, correct_answer, filter_gt=True, k=None):
    """ Computes MRR and Hits@k (k=1,3,10) values for a given triple prediction """
    if k is None:
        k = [1, 3, 10]

    ranked_triples = rank(triples)

    # Combine train + val + test by concatenating vertically
    ground_truth = train + val +  test
    if filter_gt:
        ranked_triples = filter_triples(ranked_triples, ground_truth)

    mrr = compute_mrr(ranked_triples, correct_answer)
    hits = dict()
    for i in k:
        hits[k] = compute_hits(ranked_triples, correct_answer, i)

    return mrr, hits

#######################################################################################################################
# Node classification Utils
#######################################################################################################################

