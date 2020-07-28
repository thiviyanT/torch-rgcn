from torch_rgcn.layers import RelationalGraphConvolution, RelationalGraphConvolutionRP
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
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 encoder_config=None,
                 decoder_config=None):
        super(RelationPredictor, self).__init__()

        nemb = encoder_config["node_embedding"] if "node_embedding" in encoder_config else None
        nhid1 = encoder_config["hidden1_size"] if "hidden1_size" in encoder_config else None
        nhid2 = encoder_config["hidden2_size"] if "hidden2_size" in encoder_config else None
        rgcn_layers = encoder_config["num_layers"] if "num_layers" in encoder_config else 2
        edge_dropout = encoder_config["edge_dropout"] if "edge_dropout" in encoder_config else None
        decomposition = encoder_config["decomposition"] if "decomposition" in encoder_config else None

        assert (nnodes is not None or nrel is not None or nhid1 is not None), \
            "The following must be specified: number of nodes, number of relations and output dimension!"
        assert 0 < rgcn_layers < 3, "Only supports the following number of convolution layers: 1 and 2."

        self.num_nodes = nnodes
        self.num_rels = nrel
        self.rgcn_layers = rgcn_layers
        self.nemb = nemb

        if nemb is not None:
            self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))
            nn.init.xavier_uniform_(self.node_embeddings)
            nfeat = self.nemb

        self.rgc1 = RelationalGraphConvolutionRP(
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid1,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False
        )
        if rgcn_layers == 2:
            self.rgc2 = RelationalGraphConvolutionRP(
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid1,
                out_features=nhid2,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=True
            )

        # Decoder
        self.relations = nn.Parameter(torch.FloatTensor(nrel, nhid2 if rgcn_layers == 2 else nhid1))
        nn.init.xavier_uniform_(self.relations)

    def distmult_score(self, triples, nodes, relations):
        """ Simple DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf) """

        s = triples[:, 0]
        p = triples[:, 1]
        o = triples[:, 2]
        s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

        scores = (s * p * o).sum(dim=1)

        return scores.view(-1)

    def forward(self, graph, batch):
        """ Embed relational graph and then compute score """

        if self.nemb is not None:
            x = self.node_embeddings
            x = self.rgc1(graph, features=x)
        else:
            x = self.rgc1(graph)

        if self.rgcn_layers == 2:
            x = F.relu(x)
            x = self.rgc2(graph, features=x)

        scores = self.distmult_score(batch, x, self.relations)
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


class CompressionRelationPredictor(nn.Module):
    """ Relation prediction model with a bottleneck architecture within the encoder and DistMult decoder """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 encoder_config=None,
                 decoder_config=None):
        super(CompressionRelationPredictor, self).__init__()

        # Declare variables
        nhid = encoder_config["hidden1_size"] if "hidden1_size" in encoder_config else None
        nemb = encoder_config["embedding_size"] if "embedding_size" in encoder_config else None
        rgcn_layers = encoder_config["num_layers"] if "num_layers" in encoder_config else 2
        edge_dropout = encoder_config["edge_dropout"] if "edge_dropout" in encoder_config else None
        decomposition = encoder_config["decomposition"] if "decomposition" in encoder_config else None
        rgcn_layers = encoder_config["num_layers"] if "num_layers" in encoder_config else 2
        self.rgcn_layers = rgcn_layers

        assert 0 < rgcn_layers < 3, "Only supports the following number of convolution layers: 1 and 2."

        # Encoder
        self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))
        self.encoding_layer = torch.nn.Linear(nemb, nhid)
        self.rgc1 = RelationalGraphConvolutionRP(
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nhid,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False
        )
        if rgcn_layers == 2:
            self.rgc2 = RelationalGraphConvolutionRP(
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid,
                out_features=nhid,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=True
            )
        self.decoding_layer = torch.nn.Linear(nhid, nemb)
        # Decoder
        self.relations = nn.Parameter(torch.FloatTensor(nrel, nemb))

        # Initialise Parameters
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.relations)

    def distmult_score(self, triples, nodes, relations):
        """ Simple DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf) """

        s = triples[:, 0]
        p = triples[:, 1]
        o = triples[:, 2]
        s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

        scores = (s * p * o).sum(dim=1)

        return scores.view(-1)

    def forward(self, graph, all_triples):
        """ Embed relational graph and then compute class probabilities """

        x = self.node_embeddings

        x = self.encoding_layer(x)

        x = self.rgc1(graph, features=x)

        if self.rgcn_layers == 2:
            x = F.relu(x)
            x = self.rgc2(graph, features=x)

        x = self.node_embeddings + self.decoding_layer(x)

        scores = self.distmult_score(all_triples, x, self.relations)
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
