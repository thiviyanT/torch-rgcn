from torch_rgcn.layers import RelationalGraphConvolution
from torch_rgcn.utils import add_inverse_and_self
import torch.nn.functional as F
from torch import nn
import torch
import copy


######################################################################################
# Models for Experiment Reproduction
######################################################################################


class RelationPredictor(nn.Module):
    """ Relation Prediction via RGCN encoder and DistMult decoder """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 dropout=.0,
                 encoder_config=None,
                 decoder_config=None):
        super(RelationPredictor, self).__init__()

        nhid = encoder_config["hidden_size"] if "hidden_size" in encoder_config else None
        nemb = encoder_config["embedding_size"] if "embedding_size" in encoder_config else None
        rgcn_layers = encoder_config["num_layers"] if "num_layers" in encoder_config else 2
        edge_dropout = encoder_config["edge_dropout"] if "edge_dropout" in encoder_config else None
        decomposition = encoder_config["decomposition"] if "decomposition" in encoder_config else None

        assert (triples is not None or nnodes is not None or nrel is not None or nemb is not None), \
            "The following must be specified: triples, number of nodes, number of relations and output dimension!"
        assert 0 < rgcn_layers < 3, "Only supports the following number of RGCN layers: 1 and 2."

        self.num_nodes = nnodes
        self.num_rels = nrel
        self.dropout = dropout
        self.rgcn_layers = rgcn_layers

        if rgcn_layers == 1:
            nhid = nemb

        if rgcn_layers == 2:
            assert nhid is not None, "Requested two layers but hidden_size not specified!"

        triples = torch.tensor(triples, dtype=torch.long)
        with torch.no_grad():
            self.register_buffer('triples', triples)
            # Add inverse relations and self-loops to triples
            self.register_buffer('triples_plus', add_inverse_and_self(triples, nnodes, nrel))

        self.rgc1 = RelationalGraphConvolution(
            triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False
        )
        if rgcn_layers == 2:
            self.rgc2 = RelationalGraphConvolution(
                triples=self.triples_plus,
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid,
                out_features=nemb,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=True
            )

        # Decoder
        self.relations = nn.Parameter(torch.FloatTensor(nrel, nemb))

        # Initialise Parameters
        nn.init.xavier_uniform_(self.relations)

    def distmult_score(self, triples, nodes, relations):
        """ Simple DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf) """

        s = triples[:, 0]
        p = triples[:, 1]
        o = triples[:, 2]
        s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

        scores = (s * p * o).sum(dim=1)

        return scores.view(-1)

    def forward(self, triples):
        """ Embed relational graph and then compute score """

        x = self.rgc1()

        if self.rgcn_layers == 2:
            x = F.relu(x)
            x = self.rgc2(features=x)

        scores = self.distmult_score(triples, x, self.relations)
        return scores


class NodeClassifier(nn.Module):
    """ Node classification with R-GCN message passing """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 nlayers=2,
                 nclass=None,
                 edge_dropout=None,
                 decomposition=None,
                 nemb=None):
        super(NodeClassifier, self).__init__()

        self.nlayers = nlayers

        assert (triples is not None or nnodes is not None or nrel is not None or nclass is not None), \
            "The following must be specified: triples, number of nodes, number of relations and number of classes!"
        assert 0 < nlayers < 3, "Only supports the following number of RGCN layers: 1 and 2."

        if nlayers == 1:
            nhid = nclass

        if nlayers == 2:
            assert nhid is not None, "Number of hidden layers not specified!"

        triples = torch.tensor(triples, dtype=torch.long)
        with torch.no_grad():
            self.register_buffer('triples', triples)
            # Add inverse relations and self-loops to triples
            self.register_buffer('triples_plus', add_inverse_and_self(triples, nnodes, nrel))

        self.rgc1 = RelationalGraphConvolution(
            triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False
        )
        if nlayers == 2:
            self.rgc2 = RelationalGraphConvolution(
                triples=self.triples_plus,
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid,
                out_features=nclass,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=True
            )

    def forward(self):
        """ Embed relational graph and then compute class probabilities """
        x = self.rgc1()

        if self.nlayers == 2:
            x = F.relu(x)
            x = self.rgc2(features=x)

        return x


######################################################################################
# RGCN Extensions
######################################################################################


class CompressionRelationPredictor(RelationPredictor):
    """ Relation prediction model with a bottleneck architecture within the encoder """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 dropout=.0,
                 encoder_config=None,
                 decoder_config=None):

        # Hack: Sacred config object is immutable, so we its data!
        encoder_config = copy.deepcopy(encoder_config)

        # Declare variables for bottleneck architecture
        embedding_size = encoder_config["embedding_size"]
        compression_size = encoder_config["hidden_size"]
        # Manipulate RGCN input
        nfeat = encoder_config["hidden_size"]  # Configure RGCN to accept compressed node embedding as feature matrix
        encoder_config["embedding_size"] = encoder_config["hidden_size"]  # Set RGCN output dimension

        super(CompressionRelationPredictor, self)\
            .__init__(triples, nnodes, nrel, nfeat, dropout, encoder_config, decoder_config)

        # Encoder
        self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, embedding_size))
        self.encoding_layer = torch.nn.Linear(embedding_size, compression_size)
        self.decoding_layer = torch.nn.Linear(compression_size, embedding_size)
        # Decoder
        self.relations = nn.Parameter(torch.FloatTensor(nrel, embedding_size))

        # Initialise Parameters
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.relations)

    def forward(self, triples):
        """ Embed relational graph and then compute class probabilities """

        nodes = self.encoding_layer(self.node_embeddings)

        x = self.rgc1(features=nodes)

        if self.rgcn_layers == 2:
            x = F.relu(x)
            x = self.rgc2(features=x)

        x = self.node_embeddings + self.decoding_layer(x)

        scores = self.distmult_score(triples, x, self.relations)
        return scores


class EmbeddingNodeClassifier(NodeClassifier):
    """ Node classification model with node embeddings as the feature matrix """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 nlayers=2,
                 nclass=None,
                 edge_dropout=None,
                 decomposition=None,
                 nemb=None):

        assert nemb is not None, "Size of node embedding not specified!"
        nfeat = nemb  # Configure RGCN to accept node embeddings as feature matrix

        super(EmbeddingNodeClassifier, self)\
            .__init__(triples, nnodes, nrel, nfeat, nhid, nlayers, nclass, edge_dropout, decomposition)

        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))

        # Initialise Parameters
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self):
        """ Embed relational graph and then compute class probabilities """
        x = self.rgc1(self.node_embeddings)

        if self.nlayers == 2:
            x = F.relu(x)
            x = self.rgc2(features=x)

        return x


class GlobalNodeClassifier(NodeClassifier):
    """ Node classification model with global readouts """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 nlayers=2,
                 nclass=None,
                 edge_dropout=None,
                 decomposition=None,
                 nemb=None):

        assert nemb is not None, "Size of node embedding not specified!"
        nfeat = nemb  # Configure RGCN to accept node embeddings as feature matrix

        super(GlobalNodeClassifier, self)\
            .__init__(triples, nnodes, nrel, nfeat, nhid, nlayers, nclass, edge_dropout, decomposition)

        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))

        # Initialise Parameters
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self):
        """ Embed relational graph and then compute class probabilities """

        x = self.node_embeddings

        x = x + x.mean(dim=0, keepdim=True)

        x = self.rgc1(features=x)

        x = x + x.mean(dim=0, keepdim=True)

        if self.nlayers == 2:
            x = F.relu(x)
            x = self.rgc2(features=x)

        return x
