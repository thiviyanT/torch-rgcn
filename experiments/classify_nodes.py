from misc import create_experiment
from rgcn.models import NodeClassifier
from data import load_node_classification_data
import torch

""" 
Relational Graph Convolution Network for node classification. 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 3).
"""

# TODO: Create sparse matrix multiplication!
# TODO: Row-wise normalisation!
# TODO: Rewrite data loader - Low priority

# Create sacred experiment observer
ex = create_experiment(name='R-GCN Node Classification', database='node_class')


@ex.config
def exp_config():
    """ Declare default configurations for node classification experiment """
    dataset = 'aifb'  # Name of the dataset
    epochs = 1000  # Number of message passing iterations
    learn_rate = 1e-3  # Learning rate for optimiser
    hid_units = 16  # Size of the hidden layer
    weight_reg = 'block'  # Type of weight regularisation (basis decomposition or block diagonal decomposition)
    num_bases = 10  # Number of bases for basis decomposition
    use_cuda = True  # If true, model training is performed on GPU if they are available
    optimiser = 'adam'  # Type of learning optimiser
    weight_decay = 0.0  # Weight decay
    dropout = 0.6  # Dropout rate for RGCN


@ex.automain
def train_model(dataset,
                hid_units,
                optimiser,
                learn_rate,
                weight_decay,
                weight_reg,
                num_bases,
                dropout,
                epochs,
                use_cuda,
                _run):

    # Load Train and Test data  # TODO: Check why node classification data do not have validation sets
    edges, (n2i, i2n), (r2i, i2r), train, test = load_node_classification_data(dataset)

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
    num_relations = len(edges)

    # Declare model here
    model = NodeClassifier(
        edges,
        num_nodes,
        num_relations,
        nclass=num_classes,
        nhid=hid_units,
        decomp=weight_reg,
        nbases=num_bases,
        dropout=dropout,
        device=device)

    if use_cuda:
        model.cuda()

    if optimiser == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    elif optimiser == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    else:
        raise NameError(f"'{optimiser}' optimiser has not been recognised!")

    print("Start training...")
    for epoch in range(epochs):
        print("Training")
        optimizer.zero_grad()
        model.train()

        if use_cuda:
            pass

        out = model(train_idx)
        loss = 0

        # Log loss
        _run.log_scalar("training.loss", loss.item(), step=epoch)

    print('Training is complete!')

    print("Start testing:")
    # Test model on test data