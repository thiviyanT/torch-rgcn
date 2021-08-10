from utils.misc import create_experiment
from torch_rgcn.models import NodeClassifier, EmbeddingNodeClassifier
from utils.data import load_node_classification_data
from sklearn.metrics import accuracy_score
from statistics import stdev
import torch.nn as nn
import torch
import time

""" 
Relational Graph Convolution Network for node classification. 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 3).
"""

# Create sacred object for experiment tracking
ex = create_experiment(name='R-GCN Node Classification', database='node_class')


@ex.capture
def train_model(dataset,
                training,
                rgcn,
                evaluation,
                repeat,
                _run):

    assert training is not None, "Training configuration is not specified!"

    # Set default values
    repeat = f"_{repeat}"
    epochs = training["epochs"] if "epochs" in training else 50
    nhid = rgcn["hidden_size"] if "hidden_size" in rgcn else 16
    nlayers = rgcn["num_layers"] if "num_layers" in rgcn else 2
    decomposition = rgcn["decomposition"] if "decomposition" in rgcn else None
    layer1_l2_penalty = rgcn["layer1_l2_penalty"] if "layer1_l2_penalty" in rgcn else 0.0
    nemb = rgcn["node_embeddings"] if "node_embeddings" in rgcn else 10
    node_embedding_l2_penalty = rgcn["node_embedding_l2_penalty"] if "node_embedding_l2_penalty" in rgcn else 0.0
    final_run = evaluation["final_run"] if "final_run" in evaluation else False

    # Load Data
    # Note: Validation dataset will be used as test if this is not a test run
    triples, (n2i, i2n), (r2i, i2r), train, test = load_node_classification_data(
        dataset["name"], use_test_set=final_run, prune=dataset["prune"])

    # Check for available GPUs
    use_cuda = training["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Convert train and test datasets to torch tensors
    train_idx = [n2i[name] for name, _ in train.items()]
    train_lbl = [cls for _, cls in train.items()]
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    train_lbl = torch.tensor(train_lbl, dtype=torch.long, device=device)

    test_idx = [n2i[name] for name, _ in test.items()]
    test_lbl = [cls for _, cls in test.items()]
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
    test_lbl = torch.tensor(test_lbl, dtype=torch.long, device=device)

    classes = set([int(l) for l in test_lbl] + [int(l) for l in train_lbl])
    num_classes = len(classes)
    num_nodes = len(n2i)
    num_relations = len(r2i)

    if rgcn["model"] == 'rgcn':
        model = NodeClassifier
    elif rgcn["model"] == 'e-rgcn':
        model = EmbeddingNodeClassifier
    else:
        raise NotImplementedError(f'\'{rgcn["model"]}\' model has not been implemented!')

    model = model(
        triples=triples,
        nnodes=num_nodes,
        nrel=num_relations,
        nclass=num_classes,
        nhid=nhid,
        nlayers=nlayers,
        decomposition=decomposition,
        nemb=nemb)

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

    print("Starting training...")
    for epoch in range(1, epochs+1):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()

        classes = model()[train_idx, :]
        loss = criterion(classes, train_lbl)

        # Apply l2 penalty on first layer weights
        if layer1_l2_penalty > 0.0:
            if decomposition is not None and decomposition['type'] == 'basis':
                layer1_l2 = model.rgc1.bases.pow(2).sum() + model.rgc1.comps.pow(2).sum()
            elif decomposition is not None and decomposition['type'] == 'block':
                layer1_l2 = model.rgc1.blocks.pow(2).sum()
            else:
                layer1_l2 = model.rgc1.weights.pow(2).sum()
            loss = loss + layer1_l2_penalty * layer1_l2

        # Apply l2 penalty on node embeddings
        if node_embedding_l2_penalty > 0.0:
            if rgcn["model"] == 'e-rgcn':
                node_embedding_l2 = model.node_embeddings.pow(2).sum()
                loss = loss + node_embedding_l2_penalty * node_embedding_l2
            else:
                raise ValueError(f"Cannot apply L2-regularisation on node embeddings for {rgcn['model']} model")

        t2 = time.time()
        loss.backward()
        optimiser.step()
        t3 = time.time()

        # Evaluate
        with torch.no_grad():
            model.eval()
            classes = model()[train_idx, :].argmax(dim=-1)
            train_accuracy = accuracy_score(classes.cpu(), train_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU

            classes = model()[test_idx, :].argmax(dim=-1)
            test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU

        _run.log_scalar(f"training.loss{repeat}", loss.item(), step=epoch)
        _run.log_scalar(f"training.accuracy{repeat}", train_accuracy, step=epoch)
        _run.log_scalar(f"test.accuracy{repeat}", test_accuracy, step=epoch)
        print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
              f'Train Accuracy: {train_accuracy:.2f} Test Accuracy: {test_accuracy:.2f}')

    print('Training is complete!')

    print("Starting evaluation...")
    model.eval()
    classes = model()[test_idx, :].argmax(dim=-1)
    test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
    _run.log_scalar(f"test.accuracy{repeat}", test_accuracy)
    print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')
    return test_accuracy


@ex.automain
def repeat(_run, repeats=1):
    """ Repeats experiments and reports average and standard deviation """
    test_accuracies = []
    for i in range(1, repeats+1):
        test_accuracy = train_model(repeat=i)
        test_accuracies.append(test_accuracy)

    avg = sum(test_accuracies)/len(test_accuracies)  # Average
    std = stdev(test_accuracies) if len(test_accuracies) != 1 else 0  # Standard Deviation
    ste = std / (len(test_accuracies)**0.5)  # Standard Error

    avg = round(avg, 2)
    ste = round(ste, 2)

    _run.log_scalar(f"test.accuracy", avg)
    _run.log_scalar(f"test.accuracy_ste", ste)
    _run.log_scalar(f"repeats", repeats)

    print(f'[Summary] Test Accuracy: {avg:.2f} -/+ {ste:.2f} { f"({repeats} runs)"  if repeats > 1 else ""}')
