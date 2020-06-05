import torch
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from rgcn.utils import block_diag
import math

# TODO: Edge dropout (apply different values for self-loops and other values)


class RelationalGraphConvolution(Module):
    """ Relational Graph Convolution Network (r-GCN) Layer (from https://arxiv.org/abs/1703.06103)"""
    def __init__(self,
                 num_nodes,
                 num_relations,
                 in_features=None,
                 out_features=16,
                 num_bases=-1,
                 num_blocks=-1,
                 weight_reg=None,
                 bias=True,
                 horizontal_stack=True,
                 reset_mode='xavier'):
        super(RelationalGraphConvolution, self).__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.in_features = in_features
        self.out_features = out_features
        self.weight_reg = weight_reg
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.horizontal_stack = horizontal_stack

        # If featureless, use number of nodes instead as input dimension
        in_dim = in_features if in_features is not None else num_nodes
        out_dim = out_features

        if self.weight_reg is None:
            self.weight = Parameter(torch.FloatTensor(num_relations, in_dim, out_dim))
        elif self.weight_reg == 'basis':
            # Weight Regularisation through Basis Decomposition
            assert self.num_bases > 0, \
                'Number of bases should be set to higher than zero for basis decomposition!'

            self.bases = Parameter(torch.FloatTensor(num_bases, in_dim, out_dim))
            self.comp = Parameter(torch.FloatTensor(num_relations, num_bases))
        elif self.weight_reg == 'block':
            # Weight Regularisation through Block Diagonal Decomposition
            assert self.num_blocks > 0, \
                'Number of blocks should be set to a value higher than zero for block diagonal decomposition!'
            assert in_dim % num_blocks == 0 and out_dim % num_blocks == 0,\
                f'For block diagonal decomposition, input dimensions ({in_dim}, {out_dim}) must be divisible ' \
                f'by number of blocks ({num_blocks})'

            self.blocks = nn.Parameter(
                torch.FloatTensor(num_relations, num_blocks, in_dim // num_blocks, out_dim // num_blocks))
        else:
            raise NotImplementedError(f'{weight_reg} decomposition has not been implemented')

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else: 
            self.register_parameter('bias', None)
            
        self.reset_parameters(reset_mode)
    
    def reset_parameters(self, reset_mode='xavier'):
        """ Initialise biases and weights (xavier or uniform) """

        if reset_mode == 'xavier':
            if self.weight_reg == 'block':
                nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))
            elif self.weight_reg == 'basis':
                nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.comp, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        elif reset_mode == 'uniform':
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            if self.weight_reg == 'block':
                self.blocks.data.uniform_(-stdv, stdv)
            elif self.weight_reg == 'basis':
                self.bases.data.uniform_(-stdv, stdv)
                self.comp.data.uniform_(-stdv, stdv)
            else:
                self.weight.data.uniform_(-stdv, stdv)

            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise NotImplementedError(f'{reset_mode} parameter initialisation method has not been implemented')

    def forward(self, adj, features=None):
        """ Message Propagation """

        assert (features is None) == (self.in_features is None)

        in_dim = self.in_features if self.in_features is not None else self.num_nodes
        out_dim = self.out_features

        if self.weight_reg is None:
            weights = self.weights
        elif self.weight_reg == 'basis':
            # TODO: Write a non-einsum version of this
            weights = torch.einsum('rb, bio -> rio', self.comps, self.bases)
        elif self.weight_reg == 'block':
            # TODO: Rewrite this using my own implementation
            weights = block_diag(self.blocks)
        else:
            raise NotImplementedError(f'{self.weight_reg} decomposition has not been implemented')

        assert weights.size() == (self.num_relations, in_dim, out_dim)

        if self.in_features is None:
            # Featureless - Input is the identity matrix
            output = torch.mm(adj, weights.view(self.num_relations * in_dim, out_dim))
        elif self.horizontal_stack:
            # Adjacency matrix horizontally stacked
            features = features[None, :, :].expand(self.num_relations, self.num_nodes, in_dim)
            # TODO: Write a non-einsum version of this
            fw = torch.einsum('rni, rio -> rno', features, weights).contiguous()
            output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))
        else:
            # Adjacency matrix vertically stacked
            output = torch.spmm(adj, features)
            output = output.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, output)

        # Note: An explicit sum operation is not required since it is free using matrix multiplication
        # Combine representations of different relations using permutation-invariant SUM operation

        assert output.size() == (self.num_nodes, out_dim)
        
        if self.bias is not None:
            output = torch.add(output, self.bias)
        
        return output
