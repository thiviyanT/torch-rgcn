from torch_rgcn.utils import block_diag, stack_matrices, sum_sparse, drop_edges
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn
import math
import torch


class RelationalGraphConvolution(Module):
    """ Relational Graph Convolution (RGC) Layer (as described in https://arxiv.org/abs/1703.06103)"""
    def __init__(self,
                 triples=None,
                 num_nodes=None,
                 num_relations=None,
                 in_features=None,
                 out_features=None,
                 edge_dropout=None,
                 edge_dropout_self_loop=None,
                 bias=True,
                 decomposition=None,
                 vertical_stacking=False,
                 reset_mode='xavier'):
        super(RelationalGraphConvolution, self).__init__()

        assert (triples is not None or num_nodes is not None or num_relations is not None or out_features is not None), \
            "The following must be specified: triples, number of nodes, number of relations and output dimension!"

        # If featureless, use number of nodes instead as input dimension
        in_dim = in_features if in_features is not None else num_nodes
        out_dim = out_features

        # Unpack arguments
        weight_decomp = decomposition['type'] if decomposition is not None and 'type' in decomposition else None
        num_bases = decomposition['num_bases'] if decomposition is not None and 'num_bases' in decomposition else None
        num_blocks = decomposition['num_blocks'] if decomposition is not None and 'num_blocks' in decomposition else None

        self.triples = triples
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decomp = weight_decomp
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.vertical_stacking = vertical_stacking
        self.edge_dropout = edge_dropout
        self.edge_dropout_self_loop = edge_dropout_self_loop

        # Instantiate weights
        if self.weight_decomp is None:
            self.weights = Parameter(torch.FloatTensor(num_relations, in_dim, out_dim))
        elif self.weight_decomp == 'basis':
            # Weight Regularisation through Basis Decomposition
            assert num_bases > 0, \
                'Number of bases should be set to higher than zero for basis decomposition!'
            self.bases = Parameter(torch.FloatTensor(num_bases, in_dim, out_dim))
            self.comps = Parameter(torch.FloatTensor(num_relations, num_bases))
        elif self.weight_decomp == 'block':
            # Weight Regularisation through Block Diagonal Decomposition
            assert self.num_blocks > 0, \
                'Number of blocks should be set to a value higher than zero for block diagonal decomposition!'
            assert in_dim % self.num_blocks == 0 and out_dim % self.num_blocks == 0,\
                f'For block diagonal decomposition, input dimensions ({in_dim}, {out_dim}) must be divisible ' \
                f'by number of blocks ({self.num_blocks})'
            self.blocks = nn.Parameter(
                torch.FloatTensor(num_relations, self.num_blocks, in_dim // self.num_blocks, out_dim // self.num_blocks))
        else:
            raise NotImplementedError(f'{self.weight_decomp} decomposition has not been implemented')

        # Instantiate biases
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else: 
            self.register_parameter('bias', None)
            
        self.reset_parameters(reset_mode)
    
    def reset_parameters(self, reset_mode='xavier'):
        """ Initialise biases and weights (xavier or uniform) """

        if reset_mode == 'xavier':
            if self.weight_decomp == 'block':
                nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))
            elif self.weight_decomp == 'basis':
                nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        elif reset_mode == 'uniform':
            stdv = 1.0 / math.sqrt(self.weights.size(1))
            if self.weight_decomp == 'block':
                self.blocks.data.uniform_(-stdv, stdv)
            elif self.weight_decomp == 'basis':
                self.bases.data.uniform_(-stdv, stdv)
                self.comps.data.uniform_(-stdv, stdv)
            else:
                self.weights.data.uniform_(-stdv, stdv)

            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise NotImplementedError(f'{reset_mode} parameter initialisation method has not been implemented')

    def forward(self, features=None):
        """ Perform a single pass of message propagation """

        assert (features is None) == (self.in_features is None), \
            "Layer has not been properly configured to take in features!"

        in_dim = self.in_features if self.in_features is not None else self.num_nodes
        triples = self.triples
        out_dim = self.out_features
        edge_dropout = self.edge_dropout
        weight_decomp = self.weight_decomp
        num_nodes = self.num_nodes
        num_relations = self.num_relations
        vertical_stacking = self.vertical_stacking

        # Apply edge dropout
        if edge_dropout is not None and self.training:

            assert 'general' in edge_dropout and 'self_loop' in edge_dropout, \
                'General and self-loop edge dropouts must be specified!'
            assert type(edge_dropout['general']) is float and 0.0 <= edge_dropout['general'] <= 1.0, \
                "Edge dropout rates must between 0.0 and 1.0!"

            general_edo = edge_dropout['general']
            self_loop_edo = edge_dropout['self_loop']

            triples = drop_edges(triples, num_nodes, general_edo, self_loop_edo)

        # Choose weights
        if weight_decomp is None:
            weights = self.weights
        elif weight_decomp == 'basis':
            weights = torch.einsum('rb, bio -> rio', self.comps, self.bases)
        elif weight_decomp == 'block':
            weights = block_diag(self.blocks)
        else:
            raise NotImplementedError(f'{weight_decomp} decomposition has not been implemented')

        # Determine whether to use cuda or not
        if weights.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        # Stack adjacency matrices (vertically/horizontally)
        adj_indices, adj_size = stack_matrices(
            triples,
            num_nodes,
            num_relations,
            vertical_stacking=vertical_stacking,
            device=device
        )

        num_triples = adj_indices.size(0)
        vals = torch.ones(num_triples, dtype=torch.float, device=device)

        # Apply normalisation (vertical-stacking -> row-wise rum & horizontal-stacking -> column-wise sum)
        sums = sum_sparse(adj_indices, vals, adj_size, row_normalisation=vertical_stacking, device=device)
        if not vertical_stacking:
            # Rearrange column-wise normalised value to reflect original order (because of transpose-trick)
            n = (len(vals) - num_nodes) // 2
            sums = torch.cat([sums[n:2 * n], sums[:n], sums[2 * n:]], dim=0)
        vals = vals / sums

        # Construct adjacency matrix
        if device == 'cuda':
            adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)
        else:
            adj = torch.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)

        assert weights.size() == (num_relations, in_dim, out_dim)

        if self.in_features is None:
            # Featureless
            output = torch.mm(adj, weights.view(num_relations * in_dim, out_dim))
        elif self.vertical_stacking:
            # Adjacency matrix vertically stacked
            af = torch.spmm(adj, features)
            af = af.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, af)
        else:
            # Adjacency matrix horizontally stacked
            fw = torch.einsum('ni, rio -> rno', features, weights).contiguous()
            output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))

        assert output.size() == (self.num_nodes, out_dim)
        
        if self.bias is not None:
            output = torch.add(output, self.bias)
        
        return output