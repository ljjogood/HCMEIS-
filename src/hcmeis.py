import sys
from torch_geometric.utils import to_dense_batch, to_dense_adj
from layers import AttentionModule, TenorNetworkModule
import torch
import torch.nn as nn
from model import Base_GNN
import torch.nn.functional as F


class HCMEIS(torch.nn.Module):

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(HCMEIS, self).__init__()

        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.feature_count = self.args.tensor_neurons + self.args.filters_3
        self.convolution_1 = Base_GNN(self.number_labels, self.args.filters_1)
        self.bn1 = nn.BatchNorm1d(self.args.filters_1)
        self.convolution_2 = Base_GNN(self.args.filters_1, self.args.filters_2)
        self.bn2 = nn.BatchNorm1d(self.args.filters_2)
        self.convolution_3 = Base_GNN(self.args.filters_2, self.args.filters_3, get_loss=True)
        self.bn3 = nn.BatchNorm1d(self.args.filters_3)

        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.tensor_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.tensor_neurons,
                                                      4)
        self.scoring_layer = torch.nn.Linear(4, 1)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights and set experience of different school experts.
        """
        nn.init.xavier_normal_(self.fully_connected_first.weight)
        nn.init.xavier_normal_(self.fully_connected_second.weight)
        nn.init.xavier_normal_(self.scoring_layer.weight)

        if self.args.expert == 0:
            # experience of primary physicians
            self.map_dic = {0: 0.0, 1: 0.120, 2: 0.204, 3: 0.186, 4: 0.244, 5: 0.147, 6: 0.039, 7: 0.057}
        elif self.args.expert == 1:
            # experience of intermediate physicians
            self.map_dic = {0: 0.0, 1: 0.114, 2: 0.210, 3: 0.328, 4: 0.151, 5: 0.062, 6: 0.025, 7: 0.105}
        elif self.args.expert == 2:
            # experience of senior physicians
            self.map_dic = {0: 0.0, 1: 0.139, 2: 0.239, 3: 0.275, 4: 0.117, 5: 0.086, 6: 0.050, 7: 0.091}
        elif self.args.expert == 3:
            # experience of specific-school physicians
            self.map_dic = {0: 0.0, 1: 0.3, 2: 0.2, 3: 0.15, 4: 0.1, 5: 0.05, 6: 0.1, 7: 0.1}
        else:
            sys.stderr.write(f"error")
            sys.exit()

    def calculate_node_scores(
            self, abstract_features_1, abstract_features_2, batch_1, batch_2
    ):
        """
        Calculate node-level scoring embeddings.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs.
        :param batch_2: Batch vector for target graphs.
        :return scores_emb: Scoring embeddings between nodes.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()
        num_nodes = max(N1, N2)

        pad_nodes1 = num_nodes - N1
        pad_nodes2 = num_nodes - N2

        abstract_features_1_padded = F.pad(
            abstract_features_1,
            pad=(0, 0, 0, pad_nodes1),
            mode='constant',
            value=0
        )

        abstract_features_2_padded = F.pad(
            abstract_features_2,
            pad=(0, 0, 0, pad_nodes2),
            mode='constant',
            value=0
        )

        scores_emb = torch.matmul(
            abstract_features_1_padded, abstract_features_2_padded.permute([0, 2, 1])
        ).detach()
        scores_emb = scores_emb.unsqueeze(1)
        scores_emb = scores_emb.view(scores_emb.size(0), -1)

        linear_layer = nn.Linear(in_features=num_nodes * num_nodes, out_features=self.args.filters_3)

        scores_emb = torch.sigmoid(linear_layer(scores_emb))

        return scores_emb

    def convolutional_pass(self, edge_index, features, H):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :param H: Label mapping of expert scoring.
        :return features: Absstract feature matrix.
        :return sup_loss: Self-supervised graph attention loss.
        """
        features = self.convolution_1(x=features, edge_index=edge_index, labels=H)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)

        features = self.convolution_2(x=features, edge_index=edge_index, labels=H)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)

        features, sup_loss = self.convolution_3(x=features, edge_index=edge_index, labels=H)
        features = torch.nn.functional.relu(features)

        return features, sup_loss

    def process_matrix(self, edge_index, label_matrix):
        """
        Redundant label processing.
        :param edge_index: Edge indices for graphs.
        :param label_matrix: Label matrix for graphs.
        :return label_scores : Label score after processing.
        """

        # Combining data
        combined = list(zip(edge_index[0], label_matrix[0])) + list(zip(edge_index[1], label_matrix[1]))

        # Deduplication
        unique_labels = {}
        for label, score in combined:
            if label not in unique_labels:
                unique_labels[label] = score
            else:
                unique_labels[label] = max(unique_labels[label], score)

        # Sorting
        sorted_scores = sorted(unique_labels.items(), key=lambda x: x[0])
        _, label_scores = zip(*sorted_scores)

        return label_scores

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Dataset.
        :return score: Similarity score.
        :return sup_loss: Self-supervised graph attention loss.
        """
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        edge_classify1 = data['g1'].edge_classify
        edge_classify2 = data['g2'].edge_classify
        features_1 = data["g1"].x
        features_2 = data["g2"].x

        matrix_1 = edge_classify1.H
        matrix_2 = edge_classify2.H

        score1 = self.process_matrix(edge_index_1.tolist(), matrix_1.tolist())
        score2 = self.process_matrix(edge_index_2.tolist(), matrix_2.tolist())
        node_matrix_1 = torch.tensor(score1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        node_matrix_2 = torch.tensor(score2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        node_matrix_1 = node_matrix_1.unsqueeze(0).repeat(matrix_1.size(0), 1)
        node_matrix_2 = node_matrix_2.unsqueeze(0).repeat(matrix_2.size(0), 1)

        node_matrix_1 = torch.cat((matrix_1, node_matrix_1), dim=1)
        node_matrix_2 = torch.cat((matrix_2, node_matrix_2), dim=1)

        # Obtain the maximum key value
        max_key = max(self.map_dic.keys())
        map_tensor = torch.zeros(max_key + 1)
        for k, v in self.map_dic.items():
            map_tensor[k] = v

        # Label-mapping
        H1_pre = map_tensor[node_matrix_1]
        H2_pre = map_tensor[node_matrix_2]

        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(
            data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(
            data["g2"].num_nodes)

        # Node-level embeddings
        abstract_features_1, sup_loss1 = self.convolutional_pass(edge_index_1, features_1, H1_pre)
        abstract_features_2, sup_loss2 = self.convolutional_pass(edge_index_2, features_2, H2_pre)
        Node_Level_scores = self.calculate_node_scores(abstract_features_1, abstract_features_2, batch_1, batch_2)

        # Graph-level embeddings
        features_pooled1 = self.attention(abstract_features_1, batch_1)
        features_pooled2 = self.attention(abstract_features_2, batch_2)
        Graph_Level_scores = self.tensor_network(features_pooled1, features_pooled2)

        # Obtain final graph-graph scores
        scores = torch.cat((Graph_Level_scores, Node_Level_scores), dim=1)
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.sigmoid(self.scoring_layer(scores)).view(-1)

        return scores, sup_loss1, sup_loss2