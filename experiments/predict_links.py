from utils.misc import create_experiment, negative_sampling, corrupt_tails, filter_triples, compute_metrics
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
    max_epochs = training["epochs"] if "epochs" in training else 500
    use_cuda = training["use_cuda"] if "use_cuda" in training else False
    node_embedding_l2_penalty = encoder["node_embedding_l2_penalty"] if "node_embedding_l2_penalty" in encoder else 0.0
    decoder_l2_penalty = decoder["l2_penalty"] if "l2_penalty" in decoder else 0.0
    final_run = evaluation["final_run"] if "final_run" in evaluation else False
    filtered = evaluation["filtered"] if "filtered" in evaluation else False
    eval_size = evaluation["early_stopping"]["eval_size"] if "eval_size" in evaluation["early_stopping"] else None
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

    if encoder["model"] == 'c-rgcn':
        model = CompressionRelationPredictor
    else:
        raise NotImplementedError(f'\'{encoder["model"]}\' encoder has not been implemented!')

    model = model(
        triples=train,
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

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print('Total number of parameters:', pytorch_total_params)

    print("Start training...")
    for epoch in range(1, max_epochs+1):
        epoch_counter += 1
        t1 = time.time()
        optimiser.zero_grad()
        model.train()

        with torch.no_grad():
            # Generate negative samples triples for training
            pos_train = train
            neg_train = negative_sampling(pos_train, n2i, training["neg_sample_rate"])
            train_idx = pos_train + neg_train
            train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)

            # Label training data (0 for positive class and 1 for negative class)
            pos_labels = torch.ones(len(pos_train), 1, dtype=torch.float, device=device)
            neg_labels = torch.zeros(len(neg_train), 1, dtype=torch.float, device=device)
            train_lbl = torch.cat([pos_labels, neg_labels], dim=0).view(-1)

        # Train model on training data
        predictions = model(train_idx)
        loss = F.binary_cross_entropy_with_logits(predictions, train_lbl)

        # Apply l2 penalty on decoder (i.e. relations parameter)
        if decoder_l2_penalty > 0.0:
            decoder_l2 = model.relations.pow(2).sum()
            loss = loss + decoder_l2_penalty * decoder_l2

        # Apply l2 penalty on node embeddings
        if node_embedding_l2_penalty > 0.0:
            if encoder["model"] == 'c-rgcn':
                node_embedding_l2 = model.node_embeddings.pow(2).sum()
                loss = loss + node_embedding_l2_penalty * node_embedding_l2
            else:
                raise ValueError(f"Cannot apply L2-regularisation on node embeddings for {encoder['model']} model")

        t2 = time.time()
        loss.backward()
        optimiser.step()
        t3 = time.time()

        # Evaluate on validation set
        if epoch % eval_every == 0:
            print("Starting evaluation...")
            with torch.no_grad():
                model.eval()
                mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()

                if eval_size is None:
                    test_sample = test
                else:
                    num_test_triples = test.shape[0]
                    test_sample = test[random.sample(range(num_test_triples), k=eval_size)]

                for s, p, o in tqdm.tqdm(test_sample):
                    s, p, o = s, p, o
                    correct_triple = (s, p, o)
                    candidates = corrupt_tails(s, p, n2i)
                    if filtered:
                        candidates = filter_triples(candidates, all_triples, correct_triple)
                    candidates = torch.tensor(candidates, dtype=torch.long, device=device)
                    scores = model(candidates)
                    mrr, hits_at_k = compute_metrics(scores, candidates, correct_triple, k=[1, 3, 10])
                    mrr_scores.append(mrr)
                    hits_at_1.append(hits_at_k[1])
                    hits_at_3.append(hits_at_k[3])
                    hits_at_10.append(hits_at_k[10])

                mrr = np.mean(np.array(mrr_scores))
                hits_at_1 = np.mean(np.array(hits_at_1))
                hits_at_3 = np.mean(np.array(hits_at_3))
                hits_at_10 = np.mean(np.array(hits_at_10))

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
                }, f'./{encoder["model"]}.model')
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
            test_sample = test[random.sample(range(num_test_triples), k=final_eval_size)]

        # Final evaluation is carried out on the entire dataset
        for s, p, o in tqdm.tqdm(test_sample):
            s, p, o = s.item(), p.item(), o.item()
            correct_triple = (s, p, o)
            candidates = corrupt_tails(s, p, n2i)
            if filtered:
                candidates = filter_triples(candidates, all_triples, correct_triple)
            candidates = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model(candidates)
            mrr, hits_at_k = compute_metrics(scores, candidates, correct_triple, k=[1, 3, 10])
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
