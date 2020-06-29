from misc import create_experiment, negative_sampling, generate_candidates, filter_triples, compute_metrics
from rgcn.models import RelationPredictor, CompressionRelationPredictor
from data import load_link_prediction_data
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
    epochs = training["epochs"] if "epochs" in training else 50
    use_cuda = training["use_cuda"] if "use_cuda" in training else False
    decoder_l2_penalty = decoder["l2_penalty"] if "l2_penalty" in decoder else 0.0
    test_run = evaluation["test_run"] if "test_run" in evaluation else False
    filtered = evaluation["filtered"] if "filtered" in evaluation else True
    eval_size =  evaluation["eval_size"] if "eval_size" in evaluation else None
    eval_on_cuda = evaluation["use_cuda"] if "use_cuda" in evaluation else False
    eval_every = evaluation["early_stopping"]["check_every"] if "check_every" in evaluation["early_stopping"] else 100
    early_stop_metric = evaluation["early_stopping"]["metric"] if "metric" in evaluation["early_stopping"] else 'mrr'
    num_stops = evaluation["early_stopping"]["num_stops"] if "num_stops" in evaluation["early_stopping"] else 2
    min_epochs = evaluation["early_stopping"]["min_epochs"] if "min_epochs" in evaluation["early_stopping"] else 1000

    # Note: Validation dataset will be used as test if this is not a test run
    (n2i, i2n), (r2i, i2r), train, test = load_link_prediction_data(dataset, use_test_set=test_run)

    # Check for available GPUs
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    eval_on_cuda = eval_on_cuda and torch.cuda.is_available()
    eval_device = torch.device('cuda' if eval_on_cuda else 'cpu')

    num_nodes = len(n2i)
    num_relations = len(r2i)
    all_triples = train + test  # Required for filtering out triples
    test = torch.tensor(test, dtype=torch.long, device=eval_device)

    if encoder["model"] == 'rgcn':
        model = RelationPredictor
    elif encoder["model"] == 'c-rgcn':
        model = CompressionRelationPredictor
    else:
        raise NotImplementedError(f'\'{encoder["model"]}\' encoder has not been implemented!')

    model = model(
        triples=train,
        nnodes=num_nodes,
        nrel=num_relations,
        encoder_config=encoder,
        decoder_config=decoder,
        device=device)

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

    print("Start training...")
    for epoch in range(1, epochs+1):
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

        if decoder_l2_penalty > 0.0:
            # Apply l2 penalty on decoder (i.e. relations parameter)
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
                if use_cuda and not eval_on_cuda:
                    print('Evaluating on CPU')
                    print('hit')
                    model.cpu()

                model.eval()
                mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()

                if eval_size is None:
                    test_sample = test
                else:
                    num_test_triples = test.shape[0]
                    test_sample = test[random.sample(range(num_test_triples), k=eval_size)]

                for s, p, o in tqdm.tqdm(test_sample):
                    s, p, o = s.item(), p.item(), o.item()
                    correct_triple = [s, p, o]
                    candidates = generate_candidates(s, p, n2i)
                    if filtered:
                        candidates = filter_triples(candidates, all_triples, correct_triple)
                    candidates = torch.tensor(candidates, dtype=torch.long, device=eval_device)
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

            if use_cuda:
                model.cuda()

            print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
                  f'MRR: {mrr:.3f} Hits@1: {hits_at_1:.3f} MRR: {hits_at_3:.3f} MRR: {hits_at_10:.3f}')

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
            if epoch > min_epochs and num_stops == num_no_improvements:
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

        # Final evaluation is carried out on the entire dataset

        for s, p, o in tqdm.tqdm(test_sample):
            s, p, o = s.item(), p.item(), o.item()
            correct_triple = [s, p, o]
            candidates = generate_candidates(s, p, n2i)
            if filtered:
                candidates = filter_triples(candidates, all_triples, correct_triple)
            candidates = torch.tensor(candidates, dtype=torch.long, device=eval_device)
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

    print(f'[Final Evaluation] MRR: {mrr:.3f} Hits@1: {hits_at_1:.3f} MRR: {hits_at_3:.3f} MRR: {hits_at_10:.3f}')
