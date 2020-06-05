from misc import create_experiment, negative_sampling
from rgcn.models import LinkPredictor
from data import load_link_prediction_data
import torch.nn.functional as F
import torch

""" 
Relational Graph Convolution Network for link prediction. 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 4).
"""


# Create sacred experiment observer
ex = create_experiment(name='R-GCN Link Prediction', database='link_pred')

@ex.config
def exp_config():
    """ Declare default configurations for link prediction experiment """
    dataset = 'FB15k-237'  # Name of the dataset
    epochs = 1000  # Number of message passing iterations
    learn_rate = 1e-3  # Learning rate for optimiser
    hid_units = 16  # Size of the hidden layer
    latent_size = 128  # Dimension of the latent space
    weight_reg = 'block'  # Type of weight regularisation (basis decomposition or block diagonal decomposition)
    num_blocks = 1  # Number of blocks to be used for block diagonal decomposition
    neg_sample_rate = 1  # Number of negative samples generated for every positive sample
    optimiser = 'adam'  # Type of learning optimiser
    weight_decay = 0.0  # Weight decay
    dropout = 0.6  # Dropout rate for RGCN
    edge_dropout = 0.4  # Edge dropout rate for all edges except self-loops
    edge_dropout_self_loop = 0.2  # Edge dropout for self-loop edges only
    eval_every = 10  # Evaluate every n epochs
    use_cuda = True  # If true, model training is performed on GPU if they are available


@ex.automain
def train(dataset,
          hid_units,
          latent_size,
          optimiser,
          learn_rate,
          weight_decay,
          weight_reg,
          num_blocks,
          neg_sample_rate,
          dropout,
          edge_dropout,
          edge_dropout_self_loop,
          epochs,
          eval_every,
          use_cuda,
          _run):

    # Load Train, Validation and Test data
    train, test, (n2i, i2n), (r2i, i2r) = load_link_prediction_data(dataset)

    # Check for available GPUs
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    num_nodes = len(n2i)
    num_relations = len(r2i)

    triples = [[n2i[st[0]], r2i[st[1]], n2i[st[2]]] for st in train]
    triples = torch.tensor(triples, dtype=torch.long, device='cpu')

    # Declare model here
    model = LinkPredictor(
        triples,
        num_nodes,
        num_relations,
        nhid=hid_units,
        out=latent_size,
        decomp=weight_reg,
        nblocks=num_blocks,
        dropout=dropout,
        edge_dropout=edge_dropout,
        edge_dropout_self_loop=edge_dropout_self_loop,
        device=device)

    if use_cuda:
        model.cuda()

    if optimiser == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    elif optimiser == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    else:
        raise NameError(f"'{optimiser}' optimiser has not been recognised!")

    # NOTE: Evaluation is done on the CPU due to insufficient GPU memory
    test = [[n2i[st[0]], r2i[st[1]], n2i[st[2]]] for st in test]
    test = torch.tensor(test, dtype=torch.long, device='cpu')

    print("Start training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        model.train()

        # Generate negative samples triples for training
        pos_train = [[n2i[st[0]], r2i[st[1]], n2i[st[2]]] for st in train]
        neg_train = negative_sampling(pos_train, n2i, neg_sample_rate)
        train_idx = pos_train + neg_train
        train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)

        # Label training data (0 for positive class and 1 for negative class)
        pos_labels = torch.ones(len(pos_train), 1, dtype=torch.long, device=device)
        neg_labels = torch.zeros(len(neg_train), 1, dtype=torch.long, device=device)
        train_lbl = torch.cat([pos_labels, neg_labels], dim=0)

        # Train model on training data
        output = model(train_idx)
        loss = F.binary_cross_entropy_with_logits(output, train_lbl)
        _run.log_scalar("training.loss", loss.item(), step=epoch)

        # Evaluate link prediction model every n epochs
        if epoch % eval_every == 0:
            mrr, hits_at_1, hits_at_3, hits_at_10 = 0.0
            _run.log_scalar("test.mrr", mrr, step=epoch)
            _run.log_scalar("test.hits_at_1", hits_at_1, step=epoch)
            _run.log_scalar("test.hits_at_3", hits_at_3, step=epoch)
            _run.log_scalar("test.hits_at_10", hits_at_10, step=epoch)

    print('Training is complete!')

    print("Starting final evaluation...")
    mrr, hits_at_1, hits_at_3, hits_at_10 = 0.0, 0.0, 0.0, 0.0
    _run.log_scalar("test.mrr", mrr, step=epoch)
    _run.log_scalar("test.hits_at_1", hits_at_1, step=epoch)
    _run.log_scalar("test.hits_at_3", hits_at_3, step=epoch)
    _run.log_scalar("test.hits_at_10", hits_at_10, step=epoch)