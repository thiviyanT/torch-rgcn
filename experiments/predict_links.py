from utils.misc import *
from torch_rgcn.models import LinkPredictor, CompressionRelationPredictor
from utils.data import load_link_prediction_data
import torch.nn.functional as F
import torch
import time

""" 
Relational Graph Convolution Network for link prediction. 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 4).
"""


# Create sacred object for experiment tracking
ex = create_experiment(name='R-GCN Link Prediction ', database='link_pred')


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
    eval_batch_size = evaluation["batch_size"] if "batch_size" in evaluation else 16
    eval_verbose = evaluation["verbose"] if "verbose" in evaluation else False

    # Note: Validation dataset will be used as test if this is not a test run
    (n2i, nodes), (r2i, relations), train, test, all_triples = load_link_prediction_data(dataset["name"], use_test_set=final_run)
    true_triples = generate_true_dict(all_triples)

    # Pad the node list to make it divisible by the number of blocks
    if "decomposition" in encoder and encoder["decomposition"]["type"] == 'block':
        added = 0

        if "node_embedding" in encoder:
            block_size = encoder["node_embedding"] / encoder["decomposition"]["num_blocks"]
        else:
            raise ValueError()

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
        model = LinkPredictor
    elif encoder["model"] == 'c-rgcn':
        model = CompressionRelationPredictor
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

    sampling_function = select_sampling(sampling_method)

    epoch_counter = 0


    print("Start training...")
    for epoch in range(1, max_epochs+1):
        epoch_counter += 1
        t1 = time.time()
        optimiser.zero_grad()
        model.train()

        with torch.no_grad():
            if graph_batch_size is None:
                # Use entire graph
                positives = train
                graph_batch_size = len(train)
            else:
                # Randomly sample triples from graph
                positives = sampling_function(train, sample_size=graph_batch_size, entities=n2i)
            positives = torch.tensor(positives, dtype=torch.long, device=device)
            # Generate negative samples triples for training
            negatives = positives.clone()[:, None, :].expand(graph_batch_size, neg_sample_rate, 3).contiguous()
            negatives = negative_sampling(negatives, num_nodes, head_corrupt_prob, device=device)
            batch_idx = torch.cat([positives, negatives], dim=0)

            # Label training data (0 for positive class and 1 for negative class)
            pos_labels = torch.ones(graph_batch_size, 1, dtype=torch.float, device=device)
            neg_labels = torch.zeros(graph_batch_size*neg_sample_rate, 1, dtype=torch.float, device=device)
            train_lbl = torch.cat([pos_labels, neg_labels], dim=0).view(-1)

            graph = positives
            # Apply edge dropout. Edge dropout to self-loops is applied separately inside the RGCN layer.
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
        if epoch % eval_every == 0 and not epoch == max_epochs:
            with torch.no_grad():
                print("Starting evaluation...")
                model.eval()

                graph = torch.tensor(train, dtype=torch.long)

                mrr, hits, ranks = evaluate(
                    model=model,
                    graph=graph,
                    test_set=test,
                    true_triples=true_triples,
                    num_nodes=num_nodes,
                    batch_size=eval_batch_size,
                    verbose=eval_verbose,
                    filter_candidates=filtered)

                hits_at_1, hits_at_3, hits_at_10 = hits

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
        model.eval()

        graph = torch.tensor(train, dtype=torch.long)

        mrr, hits, ranks = evaluate(
            model=model,
            graph=graph,
            test_set=test,
            true_triples=true_triples,
            num_nodes=num_nodes,
            batch_size=eval_batch_size,
            verbose=eval_verbose,
            filter_candidates=filtered)

        hits_at_1, hits_at_3, hits_at_10 = hits

        _run.log_scalar("test.mrr", mrr, step=epoch_counter)
        _run.log_scalar("test.hits_at_1", hits_at_1, step=epoch_counter)
        _run.log_scalar("test.hits_at_3", hits_at_3, step=epoch_counter)
        _run.log_scalar("test.hits_at_10", hits_at_10, step=epoch_counter)

        filtered_ = 'filtered' if filtered else 'raw'
        print(f'[Final Scores] '
              f'Total Epoch {epoch_counter} \t'
              f'MRR({filtered_}): {mrr} \t'
              f'Hits@1({filtered_}): {hits_at_1} \t'
              f'Hits@3({filtered_}): {hits_at_3} \t'
              f'Hits@10({filtered_}): {hits_at_10}')
