from misc import create_experiment
from rgcn.models import NodeClassifier, EmbeddingNodeClassifier, GlobalNodeClassifier
from data import load_node_classification_data
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import time

""" 
Relational Graph Convolution Network for node classification. 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 3).
"""

# Create sacred object for experiment tracking
ex = create_experiment(name='R-GCN Node Classification', database='node_class')


@ex.automain
def train_model(dataset,
                training,
                rgcn,
                evaluation,
                _run):

    assert training is not None, "Training configuration is not specified!"

    # Set default values
    epochs = training["epochs"] if "epochs" in training else 50
    nhid = rgcn["hidden_size"] if "hidden_size" in rgcn else 16
    nlayers = rgcn["num_layers"] if "num_layers" in rgcn else 2
    decomposition = rgcn["decomposition"] if "decomposition" in rgcn else None
    layer1_l2_penalty = rgcn["layer1_l2_penalty"] if "layer1_l2_penalty" in rgcn else 0.0
    test_run = evaluation["test_run"] if "test_run" in evaluation else False

    # Load Data
    # Note: Validation dataset will be used as test if this is not a test run
    triples, (n2i, i2n), (r2i, i2r), train, test = load_node_classification_data(dataset, use_test_set=test_run)

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
    elif rgcn["model"] == 'g-rgcn':
        model = GlobalNodeClassifier
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

    print("Starting training...")
    for epoch in range(0, epochs+1):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()

        classes = model()[train_idx, :]
        loss = criterion(classes, train_lbl)

        if layer1_l2_penalty > 0.0:
            # Apply l2 penalty on first layer weights
            if decomposition['type'] == 'basis':
                layer1_l2 = model.rgc1.bases.pow(2).sum() + model.rgc1.comps.pow(2).sum()
            elif decomposition['type'] == 'block':
                layer1_l2 = model.rgc1.blocks.pow(2).sum()
            else:
                layer1_l2 = model.rgc1.weights.pow(2).sum()
            loss = loss + layer1_l2_penalty * layer1_l2

        t2 = time.time()
        loss.backward()
        optimiser.step()
        t3 = time.time()

        # Evaluate
        with torch.no_grad():
            model.eval()
            classes = model()[train_idx, :].argmax(dim=-1)
            train_accuracy = accuracy_score(classes.cpu(), train_lbl.cpu())  # Note: Accuracy is always computed on CPU

            classes = model()[test_idx, :].argmax(dim=-1)
            test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu())  # Note: Accuracy is always computed on CPU

        _run.log_scalar("training.loss", loss.item(), step=epoch)
        _run.log_scalar("training.accuracy", train_accuracy, step=epoch)
        _run.log_scalar("test.accuracy", test_accuracy, step=epoch)
        print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
              f'Train Accuracy: {train_accuracy:.2f} Test Accuracy: {test_accuracy:.2f}')

    print('Training is complete!')

    print("Starting final evaluation...")
    model.eval()
    classes = model()[test_idx, :].argmax(dim=-1)
    test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu())  # Note: Accuracy is always computed on CPU
    _run.log_scalar("test.accuracy", test_accuracy)
    print(f'[Final Evaluation] Test Accuracy: {test_accuracy:.2f}')
