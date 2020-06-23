from misc import create_experiment, negative_sampling, perturb, rank, filter_triples, compute_scores
from rgcn.models import LinkPredictor, CompressionLinkPredictor, EmbeddingLinkPredictor, GlobalLinkPredictor
from data import load_link_prediction_data
import torch.nn.functional as F
from sacred import Experiment
import numpy as np
import torch
import time

""" 
Relational Graph Convolution Network for link prediction. 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 4).
"""


# Create sacred experiment for experiment tracking
ex = create_experiment(name='R-GCN Link Prediction', database='link_pred')


@ex.config
def exp_config():
    """ Declare default configs for link prediction experiment """
    model = 'standard-lp'  # Default model to be used
    dataset = 'FB15k-237'  # Dataset name
    epochs = 50  # Number of training epochs
    learn_rate = 0.001  # Learning rate for optimiser
    hid_units = 16  # Size of the hidden layer
    embedding_size = 128  # Dimension of the embedding
    neg_sample_rate = 1  # Number of negative samples generated for every positive sample
    optimiser = 'adam'  # Type of learning optimiser
    weight_decay = 0.0  # Weight decay
    dropout = 0.6  # Dropout rate for RGCN
    decomposition = None  # Weight decomposition (type of decomposition, number of basis/blocks)
    edge_dropout = None  # Edge dropout rates (general, self-loop)
    eval_every = 10  # Evaluate every n epochs
    use_cuda = True  # If true, model training is performed on GPU if they are available
    test_run = False  # If true, test set is used for evaluation. If False, validation set is used for evaluation.


@ex.automain
def train(model,
          dataset,
          hid_units,
          embedding_size,
          optimiser,
          learn_rate,
          weight_decay,
          decomposition,
          neg_sample_rate,
          dropout,
          edge_dropout,
          epochs,
          eval_every,
          use_cuda,
          test_run,
          _run):

    # Note: Validation dataset will be used as test if this is not a test run
    (n2i, i2n), (r2i, i2r), train, test = load_link_prediction_data(dataset, use_test_set=test_run)

    # Check for available GPUs
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    num_nodes = len(n2i)
    num_relations = len(r2i)

    all_triples = train + test  # Required for filtering out triples

    if model == 'standard-lp':
        model = LinkPredictor
    elif model == 'embedding-lp':
        model = EmbeddingLinkPredictor
    elif model == 'compression-lp':
        model = CompressionLinkPredictor
    elif model == 'global-lp':
        model = GlobalLinkPredictor
    else:
        raise NotImplementedError(f'\'{model}\' model has not been implemented!')

    model = model(
        triples=train,
        nnodes=num_nodes,
        nrel=num_relations,
        nhid=hid_units,
        out=embedding_size,
        decomposition=decomposition,
        dropout=dropout,
        edge_dropout=edge_dropout,
        device=device)

    if use_cuda:
        model.cuda()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam
    elif optimiser == 'adamw':
        optimiser = torch.optim.AdamW
    else:
        raise NotImplementedError(f'\'{optimiser}\' optimiser has not been implemented!')

    optimiser = optimiser(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    # NOTE: Evaluation is done on the CPU due to insufficient GPU memory
    test = torch.tensor(test, dtype=torch.long, device='cpu')

    print("Start training...")
    for epoch in range(epochs):

        t1 = time.time()
        optimiser.zero_grad()
        model.train()

        with torch.no_grad():
            # Generate negative samples triples for training
            pos_train = train
            neg_train = negative_sampling(pos_train, n2i, neg_sample_rate)
            train_idx = pos_train + neg_train
            train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)

            # Label training data (0 for positive class and 1 for negative class)
            pos_labels = torch.ones(len(pos_train), 1, dtype=torch.long, device=device)
            neg_labels = torch.zeros(len(neg_train), 1, dtype=torch.long, device=device)
            train_lbl = torch.cat([pos_labels, neg_labels], dim=0)

        # Train model on training data
        z = model(train_idx)
        loss = F.binary_cross_entropy_with_logits(z, train_lbl)
        _run.log_scalar("training.loss", loss.item(), step=epoch)

        loss.backward()
        optimiser.step()
        t2 = time.time()
        print(f'Epoch {epoch}: Loss: {loss} Time: {(t2 - t1):.3f}')

        # During training, evaluate link prediction model every n epochs
        if epoch % eval_every == 0:
            mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()
            for triple in test:
                candidates = perturb(triple, n2i)
                scored_candidates = model.score(candidates)
                mrr, hits_at_k = compute_scores(scored_candidates, all_triples, triple, filter=True, k=[1, 3, 10])
                mrr_scores.append(mrr)
                hits_at_1.append(hits_at_k[1])
                hits_at_3.append(hits_at_k[3])
                hits_at_10.append(hits_at_k[10])

            mrr = np.mean(np.array(mrr_scores), axis=1)
            hits_at_1 = np.mean(np.array(hits_at_1), axis=1)
            hits_at_3 = np.mean(np.array(hits_at_3), axis=1)
            hits_at_10 = np.mean(np.array(hits_at_10), axis=1)

            _run.log_scalar("test.mrr", mrr, step=epoch)
            _run.log_scalar("test.hits_at_1", hits_at_1, step=epoch)
            _run.log_scalar("test.hits_at_3", hits_at_3, step=epoch)
            _run.log_scalar("test.hits_at_10", hits_at_10, step=epoch)

            print(f'Epoch {epoch}: Loss: {loss} Time: {(t2 - t1):.3f} '
                  f'MRR: {mrr:.3f} Hits@1: {hits_at_1:.3f} MRR: {hits_at_3:.3f} MRR: {hits_at_10:.3f}')

    print('Training is complete!')

    print("Starting final evaluation on test data...")
    mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()
    for triple in test:
        candidates = perturb(triple, n2i)
        scored_candidates = model.score(candidates)
        mrr, hits_at_k = compute_scores(scored_candidates, all_triples, triple, filter=True, k=[1, 3, 10])
        mrr_scores.append(mrr)
        hits_at_1.append(hits_at_k[1])
        hits_at_3.append(hits_at_k[3])
        hits_at_10.append(hits_at_k[10])

    mrr = np.mean(np.array(mrr_scores), axis=1)
    hits_at_1 = np.mean(np.array(hits_at_1), axis=1)
    hits_at_3 = np.mean(np.array(hits_at_3), axis=1)
    hits_at_10 = np.mean(np.array(hits_at_10), axis=1)

    _run.log_scalar("test.mrr", mrr, step=epochs)
    _run.log_scalar("test.hits_at_1", hits_at_1, step=epochs)
    _run.log_scalar("test.hits_at_3", hits_at_3, step=epochs)
    _run.log_scalar("test.hits_at_10", hits_at_10, step=epochs)