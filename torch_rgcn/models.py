from torch_rgcn.layers import RelationalGraphConvolution, RelationalGraphConvolutionRP, DistMult
from torch_rgcn.utils import add_inverse_and_self, select_w_init
import torch.nn.functional as F
from torch import nn
import torch

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

        # Encoder config
        nemb = encoder_config["node_embedding"] if "node_embedding" in encoder_config else None
        nhid1 = encoder_config["hidden1_size"] if "hidden1_size" in encoder_config else None
        nhid2 = encoder_config["hidden2_size"] if "hidden2_size" in encoder_config else None
        rgcn_layers = encoder_config["num_layers"] if "num_layers" in encoder_config else 2
        edge_dropout = encoder_config["edge_dropout"] if "edge_dropout" in encoder_config else None
        decomposition = encoder_config["decomposition"] if "decomposition" in encoder_config else None
        encoder_w_init = encoder_config["weight_init"] if "weight_init" in encoder_config else None
        encoder_gain = encoder_config["include_gain"] if "include_gain" in encoder_config else False
        encoder_b_init = encoder_config["bias_init"] if "bias_init" in encoder_config else None

        # Decoder config
        decoder_l2_type = decoder_config["l2_penalty_type"] if "l2_penalty_type" in decoder_config else None
        decoder_l2 = decoder_config["l2_penalty"] if "l2_penalty" in decoder_config else None
        decoder_w_init = decoder_config["weight_init"] if "weight_init" in decoder_config else None
        decoder_gain = decoder_config["include_gain"] if "include_gain" in decoder_config else False
        decoder_b_init = decoder_config["bias_init"] if "bias_init" in decoder_config else None

        assert (nnodes is not None or nrel is not None or nhid1 is not None), \
            "The following must be specified: number of nodes, number of relations and output dimension!"
        assert 0 < rgcn_layers < 3, "Only supports the following number of convolution layers: 1 and 2."

        self.num_nodes = nnodes
        self.num_rels = nrel
        self.rgcn_layers = rgcn_layers
        self.nemb = nemb

        self.decoder_l2_type = decoder_l2_type
        self.decoder_l2 = decoder_l2

        if nemb is not None:
            self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))
            self.node_embeddings_bias = nn.Parameter(torch.zeros(1, nemb))
            init = select_w_init(encoder_w_init)
            init(self.node_embeddings)

        if nfeat is None:
            nfeat = self.nemb

        self.rgc1 = RelationalGraphConvolutionRP(
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid1,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False,
            w_init=encoder_w_init,
            w_gain=encoder_gain,
            b_init=encoder_b_init
        )
        if rgcn_layers == 2:
            self.rgc2 = RelationalGraphConvolutionRP(
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid1,
                out_features=nhid2,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=False,
                w_init=encoder_w_init,
                w_gain=encoder_gain,
                b_init=encoder_b_init
            )

        # Decoder
        out_feat = nemb  # TODO sort this out
        self.scoring_function = DistMult(nrel, out_feat, nnodes, nrel, decoder_w_init, decoder_gain, decoder_b_init)

    def compute_penalty(self, batch, x):
        """ Compute L2 penalty for decoder """
        if self.decoder_l2 == 0.0:
            return 0

        # TODO Clean this up
        if self.decoder_l2_type == 'schlichtkrull-l2':
            return self.scoring_function.s_penalty(batch, x)
        else:
            return self.scoring_function.relations.pow(2).sum()

    def forward(self, graph, triples):
        """ Embed relational graph and then compute score """

        if self.nemb is not None:
            x = self.node_embeddings + self.node_embeddings_bias
            x = torch.nn.functional.relu(x)
            x = self.rgc1(graph, features=x)
        else:
            x = self.rgc1(graph)

        if self.rgcn_layers == 2:
            x = F.relu(x)
            x = self.rgc2(graph, features=x)

        scores = self.scoring_function(triples, x)
        penalty = self.compute_penalty(triples, x)
        return scores, penalty


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
    """ Relation prediction model with a bottleneck architecture within the encoder and DistMult decoder """
    def __init__(self,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 encoder_config=None,
                 decoder_config=None):

        nhid = encoder_config["hidden1_size"] if "hidden1_size" in encoder_config else None
        nemb = encoder_config["node_embedding"] if "node_embedding" in encoder_config else None
        nfeat = nhid

        super(CompressionRelationPredictor, self) \
            .__init__(nnodes, nrel, nfeat, encoder_config, decoder_config)

        self.encoding_layer = torch.nn.Linear(nemb, nhid)
        self.decoding_layer = torch.nn.Linear(nhid, nemb)

    def forward(self, graph, triples):
        """ Embed relational graph and then compute score """

        x = self.node_embeddings + self.node_embeddings_bias
        x = torch.nn.functional.relu(x)

        x = self.encoding_layer(x)

        x = self.rgc1(graph, features=x)

        if self.rgcn_layers == 2:
            x = F.relu(x)
            x = self.rgc2(graph, features=x)

        x = self.node_embeddings + self.decoding_layer(x)

        scores = self.scoring_function(triples, x)
        penalty = self.compute_penalty(triples, x)
        return scores, penalty


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
        init = select_w_init('glorot-uniform')
        init(self.node_embeddings)

    def forward(self):
        """ Embed relational graph and then compute class probabilities """
        x = self.rgc1(self.node_embeddings)

        if self.nlayers == 2:
            x = F.relu(x)
            x = self.rgc2(features=x)

        return x


class NoHiddenEmbeddingNodeClassifier(NodeClassifier):
    """ No-hidden Node classification model with node embeddings as the feature matrix """
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

        assert nlayers == 2, "For this model only 2 layers are normally configured (for now)"
        nhid = nemb

        super(NoHiddenEmbeddingNodeClassifier, self)\
            .__init__(triples, nnodes, nrel, nfeat, nhid, 1, nclass, edge_dropout, decomposition)

        # This model has a custom first layer
        self.rgcn_no_hidden = RelationalGraphConvolution(triples=self.triples_plus,
                                                         num_nodes=nnodes,
                                                         num_relations=nrel * 2 + 1,
                                                         in_features=nfeat,
                                                         out_features=nhid,
                                                         edge_dropout=edge_dropout,
                                                         decomposition=decomposition,
                                                         vertical_stacking=False,
                                                         no_hidden=True)

        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))

        # Initialise Parameters
        nn.init.kaiming_normal_(self.node_embeddings, mode='fan_in')

    def forward(self):
        """ Embed relational graph and then compute class probabilities """
        x = self.rgcn_no_hidden(self.node_embeddings)

        # Normally there will be checked if the desired number of layers is 2, but this model implies it (for now).
        x = F.relu(x)
        x = self.rgc1(features=x)

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
        init = select_w_init('glorot-uniform')
        init(self.node_embeddings)

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
