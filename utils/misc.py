from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from random import sample
import torch
import tqdm
import os


def create_experiment(name='exp', database=None):
    """ Create Scared experiment object for tracking experiments """
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

def generate_true_dict(all_triples):
    """ Generates a pair of dictionaries containing all true tail and head completions """
    heads, tails = {(p, o) : [] for _, p, o in all_triples}, {(s, p) : [] for s, p, _ in all_triples}

    for s, p, o in all_triples:
        heads[p, o].append(s)
        tails[s, p].append(o)

    return heads, tails

def filter_scores(scores, batch, true_triples, head=True):
    """ Filters a score matrix by setting the scores of known non-target true triples to -infinity """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    indices = [] # indices of triples whose scores should be set to -infinity

    heads, tails = true_triples

    for i, (s, p, o) in enumerate(batch):
        s, p, o = (s.item(), p.item(), o.item())
        if head:
            indices.extend([(i, si) for si in heads[p, o] if si != s])
        else:
            indices.extend([(i, oi) for oi in tails[s, p] if oi != o])
        #-- We add the indices of all know triples except the one corresponding to the target triples.

    indices = torch.tensor(indices, device=device)

    scores[indices[:, 0], indices[:, 1]] = float('-inf')

def evaluate(model, graph, test_set, true_triples, num_nodes, batch_size=16, hits_at_k=[1, 3, 10], filter_candidates=True, verbose=True):
    """ Evaluates a triple scoring model. Does the sorting in a single, GPU-accelerated operation. """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rng = tqdm.trange if verbose else range

    ranks = []
    for head in [True, False]:  # head or tail prediction

        for fr in rng(0, len(test_set), batch_size):
            to = min(fr + batch_size, len(test_set))

            batch = test_set[fr:to, :].to(device=device)
            bn, _ = batch.size()

            # compute the full score matrix (filter later)
            bases   = batch[:, 1:] if head else batch[:, :2]
            targets = batch[:, 0]  if head else batch[:, 2]

            # collect the triples for which to compute scores
            bexp = bases.view(bn, 1, 2).expand(bn, num_nodes, 2)
            ar   = torch.arange(num_nodes, device=device).view(1, num_nodes, 1).expand(bn, num_nodes, 1)
            toscore = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)
            assert toscore.size() == (bn, num_nodes, 3)

            scores, _ = model(graph, toscore)
            assert scores.size() == (bn, num_nodes)

            # filter out the true triples that aren't the target
            if filter_candidates:
                filter_scores(scores, batch, true_triples, head=head)

            # Select the true scores, and count the number of values larger than than
            true_scores = scores[torch.arange(bn, device=device), targets]
            raw_ranks = torch.sum(scores > true_scores.view(bn, 1), dim=1, dtype=torch.long)
            # -- This is the "optimistic" rank (assuming it's sorted to the front of the ties)
            num_ties = torch.sum(scores == true_scores.view(bn, 1), dim=1, dtype=torch.long)

            # Account for ties (put the true example halfway down the ties)
            branks = raw_ranks + (num_ties - 1) // 2

            ranks.extend((branks + 1).tolist())

    mrr = sum([1.0/rank for rank in ranks])/len(ranks)

    hits = []
    for k in hits_at_k:
        hits.append(sum([1.0 if rank <= k else 0.0 for rank in ranks]) / len(ranks))

    return mrr, tuple(hits), ranks

def select_sampling(method):
    method = method.lower()
    if method == 'uniform':
        return uniform_sampling
    elif method == 'edge-neighborhood':
        return edge_neighborhood
    else:
        raise NotImplementedError(f'{method} sampling method has not been implemented!')

def uniform_sampling(graph, sample_size=30000, entities=None, train_triplets=None):
    """Random uniform sampling"""
    return sample(graph, sample_size)

def edge_neighborhood(train_triples, sample_size=30000, entities=None):
    """ Edge neighborhood sampling """

    entities = {v: k for k, v in entities.items()}
    adj_list = [[] for _ in entities]
    for i, triplet in enumerate(train_triples):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]

    edges = np.zeros((sample_size), dtype=np.int32)

    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in train_triples])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    edges = [train_triples[e] for e in edges]
    return edges

def negative_sampling(batch, num_nodes, head_corrupt_prob, device='cpu'):
    """ Samples negative examples in a batch of triples. Randomly corrupts either heads or tails."""
    bs, ns, _ = batch.size()

    # new entities to insert
    corruptions = torch.randint(size=(bs * ns,),low=0, high=num_nodes, dtype=torch.long, device=device)

    # boolean mask for entries to corrupt
    mask = torch.bernoulli(torch.empty(
        size=(bs, ns, 1), dtype=torch.float, device=device).fill_(head_corrupt_prob)).to(torch.bool)
    zeros = torch.zeros(size=(bs, ns, 1), dtype=torch.bool, device=device)
    mask = torch.cat([mask, zeros, ~mask], dim=2)

    batch[mask] = corruptions

    return batch.view(bs * ns, -1)
