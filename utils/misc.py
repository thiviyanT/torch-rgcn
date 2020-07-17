from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from random import sample
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
# Relation Prediction Utils
#######################################################################################################################

def uniform_sampling(graph, sample_size=None):
    """ Randomly samples triples from graph """
    assert sample_size is not None and type(sample_size) is int, "Specify graph batch size!"
    return [i for i in sample(graph, sample_size)]

def negative_sampling(positive_triples, entity_dictionary, neg_sample_rate):
    """ Generates a set of negative samples by corrupting triples """

    all_triples = np.array(positive_triples)
    s = np.resize(all_triples[:, 0], (len(positive_triples)*neg_sample_rate,))
    p = np.resize(all_triples[:, 1], (len(positive_triples)*neg_sample_rate,))
    o = np.random.randint(low=0, high=len(entity_dictionary), size=(len(positive_triples)*neg_sample_rate,))
    negative_triples = np.stack([s, p, o], axis=1)

    return negative_triples.tolist()


def generate_candidates(s, p, entity_dictionary):
    """ Generate a list of candidate triples by replacing the tail with every entity for each test triplet """
    return [(s, p, o) for o in range(len(entity_dictionary))]

def filter_triples(candidate_triples, all_triples, correct_triple):
    """ Filter out candidate_triples that are present in all_triples, but keep correct_triple """
    return [triple for triple in set(candidate_triples) if not triple in all_triples or triple == correct_triple]

def compute_mrr(rank):
    """ Compute Mean Reciprocal Rank for a given list of ranked triples """
    return 1.0/rank

def compute_hits(rank, k):
    """ Compute Precision at K for a given list of ranked triples """
    if k == 1:
        return 1 if rank == k else 0
    else:
        return 1 if rank <= k else 0

def rank_triple(scores, candidate_triples, correct_triple):
    """ Finds rank of the correct triple after sorting candidates by their scores """
    sorted_candidates = [tuple(i[0]) for i in sorted(zip(candidate_triples.tolist(), scores.tolist()), key=lambda i: -i[1])]
    rank = sorted_candidates.index(correct_triple) + 1
    return rank

def compute_metrics(scores, candidates, correct_triple, k=None):
    """ Returns MRR and Hits@k (k=1,3,10) values for a given triple prediction """
    if k is None:
        k = [1, 3, 10]

    rank = rank_triple(scores, candidates, correct_triple)
    mrr = compute_mrr(rank)
    hits_at_k = { i:compute_hits(rank, i) for i in k }

    return mrr, hits_at_k
