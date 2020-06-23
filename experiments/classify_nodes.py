from misc import create_experiment
from rgcn.models import NodeClassifier
from data import load_node_classification_data
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import time

""" 
Relational Graph Convolution Network for node classification. 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 3).
"""

# Create sacred experiment for experiment tracking
ex = create_experiment(name='R-GCN Node Classification', database='node_class')


@ex.config
def exp_config():
    """ Declare default configs for node classification experiment """
    model = 'standard-nc'  # Default model to be used
    dataset = 'aifb'  # Name of the dataset
    epochs = 50  # Number of training epochs
    learn_rate = 1e-3  # Learning rate for optimiser
    hid_units = 16  # Size of the hidden layer
    use_cuda = True  # If true, model training is performed on GPU if they are available
    optimiser = 'adam'  # Type of learning optimiser
    decomposition = None  # Weight decomposition (type of decomposition, number of basis/blocks)
    edge_dropout = None  # Edge dropout rates (general, self-loop)
    weight_decay = 0.0  # Weight decay
    test_run = False  # If true, test set is used for evaluation. If False, validation set is used for evaluation.


@ex.automain
def train_model(model,
                dataset,
                hid_units,
                optimiser,
                learn_rate,
                weight_decay,
                decomposition,
                epochs,
                use_cuda,
                test_run,
                _run):

    # Note: Validation dataset will be used as test if this is not a test run
    triples, (n2i, i2n), (r2i, i2r), train, test = load_node_classification_data(dataset, use_test_set=test_run)

    # Check for available GPUs
    use_cuda = use_cuda and torch.cuda.is_available()
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
    num_nodes = len(n2i.values())
    num_relations = len(triples)

    if model == 'standard-nc':
        model = NodeClassifier
    else:
        raise NotImplementedError(f'\'{model}\' model has not been implemented!')

    model = model(
        triples=triples,
        nnodes=num_nodes,
        nrel=num_relations,
        nclass=num_classes,
        nhid=hid_units,
        decomposition=decomposition,
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

    print("Starting training...")
    for epoch in range(epochs):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()
        output = model()[train_idx, :]
        loss = criterion(output, train_lbl)
        train_accuracy = accuracy_score(output.argmax(dim=-1), train_lbl)
        loss.backward()
        optimiser.step()
        t2 = time.time()

        _run.log_scalar("training.loss", loss.item(), step=epoch)
        _run.log_scalar("training.accuracy", train_accuracy, step=epoch)
        print(f'Epoch {epoch}: Train Accuracy: {train_accuracy} Loss: {loss.item()} Time: {(t2 - t1):.3f}')

    print('Training is complete!')

    # Test model on test data
    print("Starting evaluation on test data...")
    test_output = output[test_idx, :]
    test_accuracy = accuracy_score(test_output.argmax(dim=-1), test_lbl)
    _run.log_scalar("test.accuracy", test_accuracy, step=epochs)
    print(f'Evaluation: Test Accuracy: {test_accuracy}')