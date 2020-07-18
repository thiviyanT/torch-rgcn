from utils.misc import *
from torch_rgcn.models import RelationPredictor, CompressionRelationPredictor
from utils.data import load_link_prediction_data
import torch.nn.functional as F
import numpy as np
import random
import torch
import time
import tqdm

""" 
Relational Graph Convolution Network for relation prediction . 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 4).
"""


# Create sacred object for experiment tracking
ex = create_experiment(name='R-GCN Relation Prediction ', database='link_pred')


@ex.automain
def train(dataset,
          training,
          encoder,
          decoder,
          evaluation,
          _run):

    # Set default values
    max_epochs = training["epochs"] if "epochs" in training else 5000
    use_cuda = training["use_cuda"] if "use_cuda" in training else False
    graph_batch_size = training["graph_batch_size"] if "graph_batch_size" in training else None
    neg_sample_rate = training["negative_sampling"]["sampling_rate"] if "negative_sampling" in training else None
    head_corrupt_prob = training["negative_sampling"]["head_prob"] if "negative_sampling" in training else None
    edge_dropout = encoder["edge_dropout"]["general"] if "edge_dropout" in encoder else 0.0
    decoder_l2_penalty = decoder["l2_penalty"] if "l2_penalty" in decoder else 0.0
    final_run = evaluation["final_run"] if "final_run" in evaluation else False
    filtered = evaluation["filtered"] if "filtered" in evaluation else False
    eval_size = evaluation["early_stopping"]["eval_size"] if "eval_size" in evaluation["early_stopping"] else 100
    eval_every = evaluation["early_stopping"]["check_every"] if "check_every" in evaluation["early_stopping"] else 100
    early_stop_metric = evaluation["early_stopping"]["metric"] if "metric" in evaluation["early_stopping"] else 'mrr'
    num_stops = evaluation["early_stopping"]["num_stops"] if "num_stops" in evaluation["early_stopping"] else 2
    min_epochs = evaluation["early_stopping"]["min_epochs"] if "min_epochs" in evaluation["early_stopping"] else 1000
    final_eval_size = evaluation["final_eval_size"] if "final_eval_size" in evaluation else None

    # Note: Validation dataset will be used as test if this is not a test run
    (n2i, i2n), (r2i, i2r), train, test, all_triples = load_link_prediction_data(dataset["name"], use_test_set=final_run)

    # Pad the node list to make it divisible by the number of blocks
    if "decomposition" in encoder and encoder["decomposition"]["type"] == 'block':
        added = 0
        while len(i2n) % encoder["decomposition"]["num_blocks"] != 0:
            label = 'null' + str(added)
            i2n.append(label)
            n2i[label] = len(i2n) - 1
            added += 1
        print(f'nodes padded to {len(i2n)} to make it divisible by {encoder["decomposition"]["num_blocks"]} (added {added} null nodes).')

    # Check for available GPUs
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    num_nodes = len(n2i)
    num_relations = len(r2i)
    test = torch.tensor(test, dtype=torch.long, device=device)

    # Split off a portion of training data for early stopping
    random.shuffle(train)
    early_stop_sample = train[:eval_size]
    train = train[eval_size:]

    if encoder["model"] == 'rgcn':
        model = RelationPredictor
    else:
        raise NotImplementedError(f'\'{encoder["model"]}\' encoder has not been implemented!')

    model = model(
        nnodes=num_nodes,
        nrel=num_relations,
        encoder_config=encoder,
        decoder_config=decoder
    )

    if use_cuda:
        model.cuda()

    if training["optimiser"]["algorithm"] == 'adam':
        optimiser = torch.optim.Adam
    elif training["optimiser"]["algorithm"] == 'adamw':
        optimiser = torch.optim.AdamW
    elif training["optimiser"]["algorithm"] == 'adagrad':
        optimiser = torch.optim.Adagrad
    else:
        raise NotImplementedError(f'\'{training["optimiser"]["algorithm"]}\' optimiser has not been implemented!')

    optimiser = optimiser(
        model.parameters(),
        lr=training["optimiser"]["learn_rate"],
        weight_decay=training["optimiser"]["weight_decay"]
    )

    best_mrr = 0
    num_no_improvements = 0
    epoch_counter = 0

    print("Start training...")
    for epoch in range(1, max_epochs+1):
        epoch_counter += 1
        t1 = time.time()
        optimiser.zero_grad()
        model.train()

        with torch.no_grad():
            # Generate negative samples triples for training
            positives = uniform_sampling(train, sample_size=graph_batch_size)
            positives = torch.tensor(positives, dtype=torch.long, device=device)
            negatives = positives.clone()[:, None, :].expand(graph_batch_size, neg_sample_rate, 3).contiguous()
            negatives = corrupt(negatives, num_nodes, head_corrupt_prob)
            batch_idx = torch.cat([positives, negatives], dim=0)

            # Label training data (0 for positive class and 1 for negative class)
            pos_labels = torch.ones(graph_batch_size, 1, dtype=torch.float, device=device)
            neg_labels = torch.zeros(graph_batch_size*neg_sample_rate, 1, dtype=torch.float, device=device)
            train_lbl = torch.cat([pos_labels, neg_labels], dim=0).view(-1)

        graph = positives
        # Apply edge dropout on all edges
        if model.training and edge_dropout > 0.0:
            keep_prob = 1 - edge_dropout
            mask = torch.bernoulli(torch.empty(size=(graph_batch_size,), dtype=torch.float, device=device).fill_(
                keep_prob)).to(torch.bool)
            graph = graph[mask, :]

        # Train model on training data
        predictions = model(graph, batch_idx)
        loss = F.binary_cross_entropy_with_logits(predictions, train_lbl)

        # Apply l2 penalty on decoder (i.e. relations parameter)
        if decoder_l2_penalty > 0.0:
            decoder_l2 = model.relations.pow(2).sum()
            loss = loss + decoder_l2_penalty * decoder_l2

        t2 = time.time()
        loss.backward()
        optimiser.step()
        t3 = time.time()

        # Evaluate
        if epoch % eval_every == 0:
            print("Starting evaluation...")
            with torch.no_grad():

                if use_cuda:
                    model.cpu()

                model.eval()
                mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()

                graph = torch.tensor(train, dtype=torch.long, device=device)

                for s, p, o in tqdm.tqdm(early_stop_sample):
                    s, p, o = s, p, o
                    correct_triple = (s, p, o)
                    c_heads = corrupt_heads(n2i, p, o)
                    c_tails = corrupt_tails(s, p, n2i)
                    batch = c_heads + c_tails
                    if filtered:
                        batch = filter_triples(batch, all_triples, correct_triple)
                    batch = torch.tensor(batch, dtype=torch.long, device=device)
                    scores = model(graph, batch)
                    mrr, hits_at_k = compute_metrics(scores, batch, correct_triple, k=[1, 3, 10])
                    mrr_scores.append(mrr)
                    hits_at_1.append(hits_at_k[1])
                    hits_at_3.append(hits_at_k[3])
                    hits_at_10.append(hits_at_k[10])

                mrr = np.mean(np.array(mrr_scores))
                hits_at_1 = np.mean(np.array(hits_at_1))
                hits_at_3 = np.mean(np.array(hits_at_3))
                hits_at_10 = np.mean(np.array(hits_at_10))

            if use_cuda:
                model.cuda()

            _run.log_scalar("training.loss", loss.item(), step=epoch)
            _run.log_scalar("test.mrr", mrr, step=epoch)
            _run.log_scalar("test.hits_at_1", hits_at_1, step=epoch)
            _run.log_scalar("test.hits_at_3", hits_at_3, step=epoch)
            _run.log_scalar("test.hits_at_10", hits_at_10, step=epoch)

            filtered = 'filtered' if filtered else 'raw'
            print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
                  f'MRR({filtered}): {mrr:.3f} \t'
                  f'Hits@1({filtered}): {hits_at_1:.3f} \t'
                  f'Hits@3({filtered}): {hits_at_3:.3f} \t'
                  f'Hits@10({filtered}): {hits_at_10:.3f}')

            # Early stopping
            if mrr > best_mrr:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': loss
                }, f'./{encoder["model"]}_{dataset["name"]}.model')
                num_no_improvements = 0
                best_mrr = mrr
            else:
                num_no_improvements += 1
            if epoch > min_epochs and num_stops == num_no_improvements or epoch > max_epochs:
                print('Early Stopping!')
                break
            else:
                continue
        else:
            _run.log_scalar("training.loss", loss.item(), step=epoch)
            print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s ')

    print('Training is complete!')

    print("Starting final evaluation...")
    mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()

    with torch.no_grad():
        model.eval()

        if final_eval_size is None or final_run:
            test_sample = test
        else:
            num_test_triples = test.shape[0]
            test_sample = test[random.sample(range(num_test_triples), k=eval_size)]

        graph = torch.tensor(train, dtype=torch.long, device=device)

        # Final evaluation is carried out on the entire dataset
        for s, p, o in tqdm.tqdm(test_sample):
            s, p, o = s.item(), p.item(), o.item()
            correct_triple = (s, p, o)
            c_heads = corrupt_heads(n2i, p, o)
            c_tails = corrupt_tails(s, p, n2i)
            batch = c_heads + c_tails
            if filtered:
                batch = filter_triples(batch, all_triples, correct_triple)
            batch = torch.tensor(batch, dtype=torch.long, device=device)
            scores = model(graph, batch)
            mrr, hits_at_k = compute_metrics(scores, batch, correct_triple, k=[1, 3, 10])
            mrr_scores.append(mrr)
            hits_at_1.append(hits_at_k[1])
            hits_at_3.append(hits_at_k[3])
            hits_at_10.append(hits_at_k[10])

    mrr = np.mean(np.array(mrr_scores))
    hits_at_1 = np.mean(np.array(hits_at_1))
    hits_at_3 = np.mean(np.array(hits_at_3))
    hits_at_10 = np.mean(np.array(hits_at_10))

    _run.log_scalar("test.mrr", mrr)
    _run.log_scalar("test.hits_at_1", hits_at_1)
    _run.log_scalar("test.hits_at_3", hits_at_3)
    _run.log_scalar("test.hits_at_10", hits_at_10)

    print(f'[Final Evaluation] '
          f'Total Epoch {epoch_counter} \t'
          f'MRR({filtered}): {mrr:.3f} \t'
          f'Hits@1({filtered}): {hits_at_1:.3f} \t'
          f'Hits@3({filtered}): {hits_at_3:.3f} \t'
          f'Hits@10({filtered}): {hits_at_10:.3f}')