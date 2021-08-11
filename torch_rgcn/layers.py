from torch_rgcn.utils import *
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn
import math
import torch


class DistMult(Module):
    """ DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf) """
    def __init__(self,
                 indim,
                 outdim,
                 num_nodes,
                 num_rel,
                 w_init='standard-normal',
                 w_gain=False,
                 b_init=None):
        super(DistMult, self).__init__()
        self.w_init = w_init
        self.w_gain = w_gain
        self.b_init = b_init

        # Create weights & biases
        self.relations = nn.Parameter(torch.FloatTensor(indim, outdim))
        if b_init:
            self.sbias = Parameter(torch.FloatTensor(num_nodes))
            self.obias = Parameter(torch.FloatTensor(num_nodes))
            self.pbias = Parameter(torch.FloatTensor(num_rel))
        else:
            self.register_parameter('sbias', None)
            self.register_parameter('obias', None)
            self.register_parameter('pbias', None)

        self.initialise_parameters()

    def initialise_parameters(self):
        """
        Initialise weights and biases

        Options for initialising weights include:
            glorot-uniform - glorot (aka xavier) initialisation using a uniform distribution
            glorot-normal - glorot (aka xavier) initialisation using a normal distribution
            schlichtkrull-uniform - schlichtkrull initialisation using a uniform distribution
            schlichtkrull-normal - schlichtkrull initialisation using a normal distribution
            normal - using a standard normal distribution
            uniform - using a uniform distribution

        Options for initialising biases include:
            ones - setting all values to one
            zeros - setting all values to zero
            normal - using a standard normal distribution
            uniform - using a uniform distribution
        """
        # Weights
        init = select_w_init(self.w_init)
        if self.w_gain:
            gain = nn.init.calculate_gain('relu')
            init(self.relations, gain=gain)
        else:
            init(self.relations)

        # Checkpoint 6
        # print('min', torch.min(self.relations))
        # print('max', torch.max(self.relations))
        # print('mean', torch.mean(self.relations))
        # print('std', torch.std(self.relations))
        # print('size', self.relations.size())

        # Biases
        if self.b_init:
            init = select_b_init(self.b_init)
            init(self.sbias)
            init(self.pbias)
            init(self.obias)

    def s_penalty(self, triples, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = split_spo(triples)

        s, p, o = nodes[s_index, :], self.relations[p_index, :], nodes[o_index, :]

        return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()

    def forward(self, triples, nodes):
        """ Score candidate triples """

        s_index, p_index, o_index = split_spo(triples)

        s, p, o = nodes[s_index, :], self.relations[p_index, :], nodes[o_index, :]

        scores = (s * p * o).sum(dim=-1)

        if self.b_init:
            scores = scores + (self.sbias[s_index] + self.pbias[p_index] + self.obias[o_index])

        return scores


class RelationalGraphConvolutionNC(Module):
    """
    Relational Graph Convolution (RGC) Layer for Node Classification
    (as described in https://arxiv.org/abs/1703.06103)
    """
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
                 diag_weight_matrix=False,
                 reset_mode='glorot_uniform'):
        super(RelationalGraphConvolutionNC, self).__init__()

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
        self.diag_weight_matrix = diag_weight_matrix
        self.edge_dropout = edge_dropout
        self.edge_dropout_self_loop = edge_dropout_self_loop

        # If this flag is active, the weight matrix is a diagonal matrix
        if self.diag_weight_matrix:
            self.weights = torch.nn.Parameter(torch.empty((self.num_relations, self.in_features)), requires_grad=True)
            self.out_features = self.in_features
            self.weight_decomp = None
            bias = False

        # Instantiate weights
        elif self.weight_decomp is None:
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
    
    def reset_parameters(self, reset_mode='glorot_uniform'):
        """ Initialise biases and weights (glorot_uniform or uniform) """

        if reset_mode == 'glorot_uniform':
            if self.weight_decomp == 'block':
                nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))
            elif self.weight_decomp == 'basis':
                nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        elif reset_mode == 'schlichtkrull':
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

        assert (features is None) == (self.in_features is None), "in_features not provided!"

        in_dim = self.in_features if self.in_features is not None else self.num_nodes
        triples = self.triples
        out_dim = self.out_features
        edge_dropout = self.edge_dropout
        weight_decomp = self.weight_decomp
        num_nodes = self.num_nodes
        num_relations = self.num_relations
        vertical_stacking = self.vertical_stacking
        general_edge_count = int((triples.size(0) - num_nodes)/2)
        self_edge_count = num_nodes

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

        # Stack adjacency matrices either vertically or horizontally
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
            n = general_edge_count
            i = self_edge_count
            sums = torch.cat([sums[n:2 * n], sums[:n], sums[-i:]], dim=0)

        vals = vals / sums

        # Construct adjacency matrix
        if device == 'cuda':
            adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)
        else:
            adj = torch.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)

        if self.diag_weight_matrix:
            assert weights.size() == (num_relations, in_dim)
        else:
            assert weights.size() == (num_relations, in_dim, out_dim)

        if self.in_features is None:
            # Message passing if no features are given
            output = torch.mm(adj, weights.view(num_relations * in_dim, out_dim))
        elif self.diag_weight_matrix:
            fw = torch.einsum('ij,kj->kij', features, weights)
            fw = torch.reshape(fw, (self.num_relations * self.num_nodes, in_dim))
            output = torch.mm(adj, fw)
        elif self.vertical_stacking:
            # Message passing if the adjacency matrix is vertically stacked
            af = torch.spmm(adj, features)
            af = af.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, af)
        else:
            # Message passing if the adjacency matrix is horizontally stacked
            fw = torch.einsum('ni, rio -> rno', features, weights).contiguous()
            output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))

        assert output.size() == (self.num_nodes, out_dim)
        
        if self.bias is not None:
            output = torch.add(output, self.bias)
        
        return output


class RelationalGraphConvolutionLP(Module):
    """
    Relational Graph Convolution (RGC) Layer for Link Prediction
    (as described in https://arxiv.org/abs/1703.06103)
    """

    def __init__(self,
                 num_nodes=None,
                 num_relations=None,
                 in_features=None,
                 out_features=None,
                 edge_dropout=None,
                 edge_dropout_self_loop=None,
                 decomposition=None,
                 vertical_stacking=False,
                 w_init='glorot-normal',
                 w_gain=False,
                 b_init=None):
        super(RelationalGraphConvolutionLP, self).__init__()

        assert (num_nodes is not None or num_relations is not None or out_features is not None), \
            "The following must be specified: number of nodes, number of relations and output dimension!"

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # If featureless, use number of nodes instead as feature input dimension
        in_dim = in_features if in_features is not None else num_nodes
        out_dim = out_features

        # Unpack arguments
        weight_decomp = decomposition['type'] if decomposition is not None and 'type' in decomposition else None
        num_bases = decomposition['num_bases'] if decomposition is not None and 'num_bases' in decomposition else None
        num_blocks = decomposition['num_blocks'] if decomposition is not None and 'num_blocks' in decomposition else None

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.in_features = in_dim
        self.out_features = out_dim
        self.weight_decomp = weight_decomp
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.vertical_stacking = vertical_stacking
        self.edge_dropout = edge_dropout
        self.edge_dropout_self_loop = edge_dropout_self_loop
        self.w_init = w_init
        self.w_gain = w_gain
        self.b_init = b_init

        # Create weight parameters
        if self.weight_decomp is None:
            self.weights = Parameter(torch.FloatTensor(num_relations, in_dim, out_dim).to(device))
        elif self.weight_decomp == 'basis':
            # Weight Regularisation through Basis Decomposition
            assert num_bases > 0, \
                'Number of bases should be set to higher than zero for basis decomposition!'
            self.bases = Parameter(torch.FloatTensor(num_bases, in_dim, out_dim).to(device))
            self.comps = Parameter(torch.FloatTensor(num_relations, num_bases).to(device))
        elif self.weight_decomp == 'block':
            # Weight Regularisation through Block Diagonal Decomposition
            assert self.num_blocks > 0, \
                'Number of blocks should be set to a value higher than zero for block diagonal decomposition!'
            assert in_dim % self.num_blocks == 0 and out_dim % self.num_blocks == 0, \
                f'For block diagonal decomposition, input dimensions ({in_dim}, {out_dim}) must be divisible ' \
                f'by number of blocks ({self.num_blocks})'
            self.blocks = nn.Parameter(
                torch.FloatTensor(num_relations - 1, self.num_blocks, in_dim // self.num_blocks,
                                  out_dim // self.num_blocks).to(device))
            self.blocks_self = nn.Parameter(torch.FloatTensor(in_dim, out_dim).to(device))
        else:
            raise NotImplementedError(f'{self.weight_decomp} decomposition has not been implemented')

        # Create bias parameters
        if b_init:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.initialise_weights()
        if self.bias is not None:
            self.initialise_biases()

    def initialise_biases(self):
        """
        Initialise bias parameters using one of the following methods:
            ones - setting all values to one
            zeros - setting all values to zero
            normal - using a standard normal distribution
            uniform - using a uniform distribution
        """

        b_init = self.b_init
        init = select_b_init(b_init)
        init(self.bias)

    def initialise_weights(self):
        """
        Initialise weights parameters using one of the following methods:
            glorot-uniform - glorot (aka xavier) initialisation using a uniform distribution
            glorot-normal - glorot (aka xavier) initialisation using a normal distribution
            schlichtkrull-uniform - schlichtkrull initialisation using a uniform distribution
            schlichtkrull-normal - schlichtkrull initialisation using a normal distribution
            normal - using a standard normal distribution
            uniform - using a uniform distribution
        """

        w_init = self.w_init
        w_gain = self.w_gain

        # Add scaling factor according to non-linearity function used
        if w_gain:
            gain = nn.init.calculate_gain('relu')
        else:
            gain = 1.0

        # Select appropriate initialisation method
        init = select_w_init(w_init)

        if self.weight_decomp == 'block':
            schlichtkrull_normal_(self.blocks, shape=[(self.num_relations-1)//2, self.in_features//self.num_blocks], gain=gain)
            # Checkpoint 3
            # print('min', torch.min(self.blocks))
            # print('max', torch.max(self.blocks))
            # print('mean', torch.mean(self.blocks))
            # print('std', torch.std(self.blocks))
            # print('size', self.blocks.size())
            schlichtkrull_normal_(self.blocks_self, shape=[(self.num_relations-1)//2, self.in_features//self.num_blocks], gain=gain)
            # Checkpoint 4
            # print('min', torch.min(self.blocks_self))
            # print('max', torch.max(self.blocks_self))
            # print('mean', torch.mean(self.blocks_self))
            # print('std', torch.std(self.blocks_self))
            # print('size', self.blocks_self.size())
        elif self.weight_decomp == 'basis':
            init(self.bases, gain=gain)
            init(self.comps, gain=gain)
        else:
            init(self.weights, gain=gain)


    def forward(self, triples, features=None):
        """ Perform a single pass of message propagation """

        assert (features is None) == (self.in_features is None), "in_features not given"

        in_dim = self.in_features if self.in_features is not None else self.num_nodes
        out_dim = self.out_features
        num_nodes = self.num_nodes
        num_relations = self.num_relations
        vertical_stacking = self.vertical_stacking
        original_num_relations = int((self.num_relations-1)/2)  # Count without inverse and self-relations
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        triples = triples.to(device)
        features = features.to(device)

        # Apply weight decomposition
        if self.weight_decomp is None:
            weights = self.weights
        elif self.weight_decomp == 'basis':
            weights = torch.einsum('rb, bio -> rio', self.comps, self.bases)
        elif self.weight_decomp == 'block':
            pass
        else:
            raise NotImplementedError(f'{self.weight_decomp} decomposition has not been implemented')

        # Define edge dropout rate for self-loops
        if self.training and self.edge_dropout["self_loop_type"] != 'schlichtkrull-dropout':
            self_loop_keep_prob = 1 - self.edge_dropout["self_loop"]
        else:
            self_loop_keep_prob = 1

        with torch.no_grad():
            # Add inverse relations
            inverse_triples = generate_inverses(triples, original_num_relations)
            # Add self-loops to triples
            self_loop_triples = generate_self_loops(
                triples, num_nodes, original_num_relations, self_loop_keep_prob, device=device)
            triples_plus = torch.cat([triples, inverse_triples, self_loop_triples], dim=0)

        # Stack adjacency matrices either vertically or horizontally
        adj_indices, adj_size = stack_matrices(
            triples_plus,
            num_nodes,
            num_relations,
            vertical_stacking=vertical_stacking,
            device=device
        )

        num_triples = adj_indices.size(0)
        vals = torch.ones(num_triples, dtype=torch.float, device=device)

        assert vals.size(0) == (triples.size(0) + inverse_triples.size(0) + self_loop_triples.size(0))

        # Apply normalisation (vertical-stacking -> row-wise rum & horizontal-stacking -> column-wise sum)
        sums = sum_sparse(adj_indices, vals, adj_size, row_normalisation=vertical_stacking, device=device)
        if not vertical_stacking:
            # Rearrange column-wise normalised value to reflect original order (because of transpose-trick)
            n = triples.size(0)
            i = self_loop_triples.size(0)
            sums = torch.cat([sums[n : 2*n], sums[:n], sums[-i:]], dim=0)
        vals = vals / sums

        # Construct adjacency matrix
        if device == 'cuda':
            adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)
        else:
            adj = torch.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)

        if self.in_features is None:
            # Message passing if no features are given
            if self.weight_decomp == 'block':
                weights = block_diag(self.blocks)
                weights = torch.cat([weights, self.blocks_self], dim=0)
            output = torch.mm(adj, weights.view(num_relations * in_dim, out_dim))
        elif self.vertical_stacking:
            # Message passing if the adjacency matrix is vertically stacked
            if self.weight_decomp == 'block':
                weights = block_diag(self.blocks)
                weights = torch.cat([weights, self.blocks_self], dim=0)
            af = torch.spmm(adj, features)
            af = af.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, af)
        else:
            # Message passing if the adjacency matrix is horizontally stacked
            if self.weight_decomp == 'block':
                n = features.size(0)
                r = num_relations - 1
                input_block_size = in_dim // self.num_blocks
                output_block_size = out_dim // self.num_blocks
                num_blocks = self.num_blocks
                block_features = features.view(n, num_blocks, input_block_size)
                fw = torch.einsum('nbi, rbio -> rnbo', block_features, self.blocks).contiguous()
                assert fw.shape == (r, n, num_blocks, output_block_size), f"{fw.shape}, {(r, n, num_blocks, output_block_size)}"
                fw = fw.view(r, n, out_dim)
                self_fw = torch.einsum('ni, io -> no', features, self.blocks_self)[None, :, :]
                if self.training and self.edge_dropout["self_loop_type"] == 'schlichtkrull-dropout':
                    self_fw = nn.functional.dropout(self_fw, p=self.edge_dropout["self_loop"], training=True,inplace=False)
                fw = torch.cat([fw, self_fw], dim=0)
                output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))
            else:
                fw = torch.einsum('ni, rio -> rno', features, weights).contiguous()
                output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))

        assert output.size() == (self.num_nodes, out_dim)

        if self.bias is not None:
            output = torch.add(output, self.bias)

        # Checkpoint 5
        # print('min', torch.min(output))
        # print('max', torch.max(output))
        # print('mean', torch.mean(output))
        # print('std', torch.std(output))
        # print('size', output.size())

        return output
