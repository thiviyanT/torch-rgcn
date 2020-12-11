from utils.misc import *
from torch_rgcn.models import RelationPredictor
from utils.data import load_link_prediction_data
import torch.nn.functional as F
import numpy as np
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

    # DEBUG
    _run.add_artifact('./experiments/predict_links_batched.py')
    _run.add_artifact('./torch_rgcn/layers.py')
    _run.add_artifact('./torch_rgcn/models.py')
    _run.add_artifact('./torch_rgcn/utils.py')
    _run.add_artifact('./utils/data.py')
    _run.add_artifact('./utils/misc.py')

    # Set default values
    max_epochs = training["epochs"] if "epochs" in training else 5000
    use_cuda = training["use_cuda"] if "use_cuda" in training else False
    graph_batch_size = training["graph_batch_size"] if "graph_batch_size" in training else None
    sampling_method = training["sampling_method"] if "sampling_method" in training else 'uniform'
    neg_sample_rate = training["negative_sampling"]["sampling_rate"] if "negative_sampling" in training else None
    head_corrupt_prob = training["negative_sampling"]["head_prob"] if "negative_sampling" in training else None
    edge_dropout = encoder["edge_dropout"]["general"] if "edge_dropout" in encoder else 0.0
    decoder_l2_penalty = decoder["l2_penalty"] if "l2_penalty" in decoder else 0.0
    final_run = evaluation["final_run"] if "final_run" in evaluation else False
    filtered = evaluation["filtered"] if "filtered" in evaluation else False
    eval_every = evaluation["check_every"] if "check_every" in evaluation else 2000

    # Note: Validation dataset will be used as test if this is not a test run
    (n2i, nodes), (r2i, relations), train, test, all_triples = load_link_prediction_data(dataset["name"], use_test_set=final_run)

    # Pad the node list to make it divisible by the number of blocks
    if "decomposition" in encoder and encoder["decomposition"]["type"] == 'block':
        added = 0

        if "node_embedding" in encoder:
            block_size = encoder["node_embedding"] / encoder["decomposition"]["num_blocks"]
        else:
            # TODO
            raise NotImplementedError()

        while len(nodes) % block_size != 0:
            label = 'null' + str(added)
            nodes.append(label)
            n2i[label] = len(nodes) - 1
            added += 1
        print(f'nodes padded to {len(nodes)} to make it divisible by {block_size} (added {added} null nodes).')

    # Check for available GPUs
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    num_nodes = len(n2i)
    num_relations = len(r2i)
    test = torch.tensor(test, dtype=torch.long, device=torch.device('cpu'))  # Note: Evaluation is performed on the CPU

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
    elif training["optimiser"]["algorithm"] == 'sgd':
        optimiser = torch.optim.SGD
    else:
        raise NotImplementedError(f'\'{training["optimiser"]["algorithm"]}\' optimiser has not been implemented!')

    optimiser = optimiser(
        model.parameters(),
        lr=training["optimiser"]["learn_rate"],
        weight_decay=training["optimiser"]["weight_decay"]
    )
    # def prod(x):
    #     m = 1
    #     for a in x:
    #         m *= a
    #     return m
    #
    # print(sum([prod(parm.size()) for parm in model.parameters()]))
    #
    # for name, param in model.named_parameters():
    #     print(name, param.size(), param.min().item(), param.max().item(), param.mean().item(), param.std().item())
    # exit()

    sampling_function = select_sampling(sampling_method)

    epoch_counter = 0


    print("Start training...")
    for epoch in range(1, max_epochs+1):
        epoch_counter += 1
        t1 = time.time()
        optimiser.zero_grad()
        model.train()

        with torch.no_grad():
            # Randomly sample positive triples
            positives = sampling_function(train, sample_size=graph_batch_size, entities=n2i)
            positives = torch.tensor(positives, dtype=torch.long, device=device)
            # Generate negative samples triples for training
            negatives = positives.clone()[:, None, :].expand(graph_batch_size, neg_sample_rate, 3).contiguous()
            negatives = corrupt(negatives, num_nodes, head_corrupt_prob, device=device)
            batch_idx = torch.cat([positives, negatives], dim=0)

            # Label training data (0 for positive class and 1 for negative class)
            pos_labels = torch.ones(graph_batch_size, 1, dtype=torch.float, device=device)
            neg_labels = torch.zeros(graph_batch_size*neg_sample_rate, 1, dtype=torch.float, device=device)
            train_lbl = torch.cat([pos_labels, neg_labels], dim=0).view(-1)

            graph = positives
            # Apply edge dropout
            if model.training and edge_dropout > 0.0:
                keep_prob = 1 - edge_dropout
                graph = graph[torch.randperm(graph.size(0))]
                sample_size = round(keep_prob * graph.size(0))
                graph = graph[sample_size:, :]

        # Train model on training data
        predictions, penalty = model(graph, batch_idx)
        loss = F.binary_cross_entropy_with_logits(predictions, train_lbl)
        loss  = loss + (decoder_l2_penalty * penalty)

        t2 = time.time()
        loss.backward()
        optimiser.step()
        t3 = time.time()

        # Evaluate on validation set
        if epoch % eval_every == 0:
            with torch.no_grad():
                print("Starting evaluation...")

                # Note: Evaluation is performed on the CPU due to memory requirements
                # if use_cuda:
                #     model.cpu()

                model.eval()
                mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()

                graph = torch.tensor(train, dtype=torch.long)

                for s, p, o in tqdm.tqdm(test):
                    s, p, o = s.item(), p.item(), o.item()
                    mrr, hits_at_k = 0, dict({1:0.0, 3:0.0, 10:0.0})
                    for corrupt_head in [True, False]:
                        correct_triple = (s, p, o)
                        if corrupt_head:
                            batch = corrupt_heads(n2i, p, o)
                        else:
                            batch = corrupt_tails(s, p, n2i)
                        if filtered:
                            batch = filter_triples(batch, all_triples, correct_triple)
                        batch = torch.tensor(batch, dtype=torch.long)
                        scores, _ = model(graph, batch)
                        mrr_, hits_at_k_ = compute_metrics(scores, batch, correct_triple, k=[1, 3, 10])
                        mrr += mrr_ * 0.5
                        hits_at_k[1] += hits_at_k_[1] * 0.5
                        hits_at_k[3] += hits_at_k_[3] * 0.5
                        hits_at_k[10] += hits_at_k_[10] * 0.5

                    mrr_scores.append(mrr)
                    hits_at_1.append(hits_at_k[1])
                    hits_at_3.append(hits_at_k[3])
                    hits_at_10.append(hits_at_k[10])

                mrr = round(np.mean(np.array(mrr_scores)), 3)
                hits_at_1 = round(np.mean(np.array(hits_at_1)), 3)
                hits_at_3 = round(np.mean(np.array(hits_at_3)), 3)
                hits_at_10 = round(np.mean(np.array(hits_at_10)), 3)

            # if use_cuda:
            #     model.cuda()

            _run.log_scalar("training.loss", loss.item(), step=epoch)
            _run.log_scalar("test.mrr", mrr, step=epoch)
            _run.log_scalar("test.hits_at_1", hits_at_1, step=epoch)
            _run.log_scalar("test.hits_at_3", hits_at_3, step=epoch)
            _run.log_scalar("test.hits_at_10", hits_at_10, step=epoch)

            filtered_ = 'filtered' if filtered else 'raw'
            print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
                  f'MRR({filtered_}): {mrr} \t'
                  f'Hits@1({filtered_}): {hits_at_1} \t'
                  f'Hits@3({filtered_}): {hits_at_3} \t'
                  f'Hits@10({filtered_}): {hits_at_10:}')

        else:
            _run.log_scalar("training.loss", loss.item(), step=epoch)
            print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s ')

    print('Training is complete!')

    with torch.no_grad():
        print("Starting final evaluation...")

        # # Note: Evaluation is performed on the CPU due to memory requirements
        # if use_cuda:
        #     model.cpu()

        model.eval()
        mrr_scores, hits_at_1, hits_at_3, hits_at_10 = list(), list(), list(), list()

        graph = torch.tensor(train, dtype=torch.long)

        # Final evaluation is carried out on the entire dataset
        for s, p, o in tqdm.tqdm(test):
            s, p, o = s.item(), p.item(), o.item()
            mrr, hits_at_k = 0, dict({1: 0.0, 3: 0.0, 10: 0.0})
            for corrupt_head in [True, False]:
                correct_triple = (s, p, o)
                if corrupt_head:
                    batch = corrupt_heads(n2i, p, o)
                else:
                    batch = corrupt_tails(s, p, n2i)
                if filtered:
                    batch = filter_triples(batch, all_triples, correct_triple)
                batch = torch.tensor(batch, dtype=torch.long)
                scores, _ = model(graph, batch)
                mrr_, hits_at_k_ = compute_metrics(scores, batch, correct_triple, k=[1, 3, 10])
                mrr += mrr_ * 0.5
                hits_at_k[1] += hits_at_k_[1] * 0.5
                hits_at_k[3] += hits_at_k_[3] * 0.5
                hits_at_k[10] += hits_at_k_[10] * 0.5

            mrr_scores.append(mrr)
            hits_at_1.append(hits_at_k[1])
            hits_at_3.append(hits_at_k[3])
            hits_at_10.append(hits_at_k[10])

        mrr = round(np.mean(np.array(mrr_scores)), 3)
        hits_at_1 = round(np.mean(np.array(hits_at_1)), 3)
        hits_at_3 = round(np.mean(np.array(hits_at_3)), 3)
        hits_at_10 = round(np.mean(np.array(hits_at_10)), 3)

    _run.log_scalar("test.mrr", mrr)
    _run.log_scalar("test.hits_at_1", hits_at_1)
    _run.log_scalar("test.hits_at_3", hits_at_3)
    _run.log_scalar("test.hits_at_10", hits_at_10)

    filtered_ = 'filtered' if filtered else 'raw'
    print(f'[Final Scores] '
          f'Total Epoch {epoch_counter} \t'
          f'MRR({filtered_}): {mrr} \t'
          f'Hits@1({filtered_}): {hits_at_1} \t'
          f'Hits@3({filtered_}): {hits_at_3} \t'
          f'Hits@10({filtered_}): {hits_at_10}')