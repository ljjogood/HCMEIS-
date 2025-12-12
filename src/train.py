import json
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader, Batch
import torch.nn.functional as F
from utils import tab_printer
from param_parser import parameter_parser
from dikgsp import DIKGSP


class DIKGSPTrainer(object):
    """
    DI-KGSP model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.initial_dataset()
        self.setup_model()

    def setup_model(self):
        """
        Introducing the DI-KGSP model.
        """
        self.model = DIKGSP(self.args, self.number_of_labels).to(self.device)


    def initial_dataset(self):
        """
        Dataset initialization.
        """
        print("\nPreparing dataset.\n")

        with open('./data/dataset.json', 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)

        with open('./data/sim_matrix.json', 'r', encoding='utf-8') as f:
            self.sim_matrix = json.load(f)

        with open('./data/features_dict.json', 'r', encoding='utf-8') as f:
            Global_Labels = json.load(f)

        random.seed(42)
        random.shuffle(self.dataset)

        train_size = int(len(self.dataset) * 0.7)
        train_simple = self.dataset[:train_size]
        self.training_graphs = train_simple

        self.global_labels = {val: index for index, val in enumerate(Global_Labels)}
        self.number_of_labels = len(self.global_labels)
        self.patient_id = self.index_to_id(self.dataset)


    def index_to_id(self, graphs):
        """
        Found the patient ID in graphs.
        :param graphs: patient subgraphs.
        :return: patient_ids: patient IDs.
        """
        patient_ids = {patient['patient_index']: patient['patient_id'] for patient in graphs}
        return patient_ids

    def create_batches(self, graphs):
        """
        Create batches of graphs.
        :param graphs: patient subgraphs.
        :return: batches of graphs.
        """
        dataset = []
        for graph in graphs:
            edge_index = torch.from_numpy(np.array(graph['edge_index'], dtype=np.int64).T).type(torch.long).to(
                self.device)
            x = torch.tensor(graph['features']).to(self.device)
            patient_index = torch.tensor(graph['patient_index']).to(self.device)
            node_classify = torch.tensor(graph['F_classify']).to(self.device)
            edge_classify = torch.tensor(graph['edge_class']).to(self.device)
            dataset.append(Data(x=x, edge_index=edge_index, y=patient_index, node_classify=node_classify,
                                edge_classify=edge_classify))
        source_batches = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        target_batches = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        return list(zip(source_batches, target_batches))

    def transfer_to_torch(self, data):
        """
         Convert graph pair data to PyTorch-compatible dictionary with patient similarity targets.
        :param data: tuple of two pyg `Data` objects
        :return new_data: dictionary containing processed graph pair and similarity targets.
        """
        new_data = dict()
        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        index_list = list(zip(data[0].y, data[1].y))
        targets = []
        for index in index_list:
            patient1_id = self.patient_id[index[0].item()]
            patient2_id = self.patient_id[index[1].item()]
            target = self.sim_matrix[patient1_id][patient2_id]
            targets.append(target)

        new_data['targets'] = torch.tensor(targets).to(self.device)
        return new_data

    def process_batch(self, batch):
        """
        Execute core training pipeline for a single batch of graph pairs.
        :param batch: input training batch.
        :return: loss value of training.
        """
        self.optimizer.zero_grad()
        data = self.transfer_to_torch(batch)
        targets = data["targets"]
        prediction,sup_loss1,sup_loss2 = self.model(data)
        loss1 = torch.nn.functional.mse_loss(prediction, targets, reduction='sum')
        weight_mse = F.softplus(self.weight_mse)
        weight_sup1 = F.softplus(self.weight_sup1)
        weight_sup2 = F.softplus(self.weight_sup2)
        loss = weight_mse*loss1 + weight_sup1*sup_loss1 + weight_sup2* sup_loss2
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        train_set = self.training_graphs
        self.weight_mse = self.model.weight_mse
        self.weight_sup1 = self.model.weight_sup1
        self.weight_sup2 = self.model.weight_sup2
        all_params = list(self.model.parameters()) + [self.weight_mse, self.weight_sup1, self.weight_sup2]
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        losses = []

        for _ in epochs:
            self.model.train()
            train_batches = self.create_batches(train_set)
            main_index = 0
            loss_sum = 0

            for index, batch in tqdm(enumerate(train_batches), total=len(train_batches), desc="train_Batches"):
                loss_score = self.process_batch(batch)
                loss_sum += loss_score/len(batch[0].y)
                main_index += 1
            loss = loss_sum / main_index
            losses.append(loss)
            epochs.set_description(
                "Epoch (train_Loss=%g)"
                % (round(loss, 5)))

        plt.plot(range(1, self.args.epochs + 1,10), losses[::10], label='Training Loss', color='blue')
        plt.title('model_train')
        plt.show()

    def save_model(self, save_path=None):
        """
        Save the model's parameters.
        :param save_path: path to save the model.
        """
        if save_path is None:
            expert_names = {
                0: "DIKGSP_Basic_Expert",
                1: "DIKGSP_Middle_Expert",
                2: "DIKGSP_High_Expert",
                3: "DIKGSP_SGP_Expert"
            }
            model_name = expert_names.get(self.args.expert, "DIKGSP_Default_Expert")
            save_path = f'./models/{model_name}.pth'

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }

        torch.save(checkpoint, save_path)
        print(f"\n✅ Model is saved in：{save_path}")

    def load_model(self, load_path):
        """
        Load training state
        :param load_path: Path to the saved model file
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"❌ can't find the model file：{load_path}")

        checkpoint = torch.load(load_path, map_location=self.device,weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ already loaded the model params：{load_path}")


def main():
    """
    Fitting a DI-KGSP model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = DIKGSPTrainer(args)
    trainer.fit()
    trainer.save_model()

if __name__ == '__main__':
    main()