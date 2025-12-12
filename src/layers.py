
from utils import scatter_
import torch

class AttentionModule(torch.nn.Module):
    """
    DI-KGSP Attention Module to get the graph-level embeddings.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param size: Dimension size for scatter_
        :param batch: Batch vector, which assigns each node to a specific example
        :return representation: A graph level representation matrix.
        """
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter_('mean', x, batch, dim_size=size)
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return scatter_('add', weighted, batch, dim_size=size)

class TenorNetworkModule(torch.nn.Module):
    """
    DI-KGSP Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.args.filters_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = len(embedding_1)
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.args.filters_3, -1))
        scoring = scoring.view(batch_size, self.args.filters_3, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.args.filters_3, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias.view(-1))
        return scores
