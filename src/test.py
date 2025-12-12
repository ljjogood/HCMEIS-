import json
import math
import sys
import random
import numpy as np
from tqdm import tqdm, trange
import torch
from torch_geometric.data import Data, DataLoader, Batch

def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def index_to_id(graphs):
    """
    Found the patient ID in graphs.
    :param graphs: patient subgraphs.
    :return: patient_ids: patient IDs.
    """
    patient_ids = {patient['patient_index']: patient['patient_id'] for patient in graphs}
    return patient_ids

def transfer_to_torch(data,patient_id,sim_matrix):
    """
     Convert graph pair data to PyTorch-compatible dictionary with patient similarity targets.
    :param data: tuple of two pyg `Data` objects
    :return new_data: dictionary containing processed graph pair and similarity targets.
    """
    device = set_device()
    new_data = dict()
    new_data["g1"] = data[0]
    new_data["g2"] = data[1]
    index_list = list(zip(data[0].y, data[1].y))
    targets = []
    for index in index_list:
        patient1_id = patient_id[index[0].item()]
        patient2_id = patient_id[index[1].item()]
        target = sim_matrix[patient1_id][patient2_id]
        targets.append(target)

    new_data['targets'] = torch.tensor(targets).to(device)
    return new_data

def create_Data(graphs):
    """
    Convert dictionary-format patient subgraphs to PyG Data dataset.
    :param graphs: patient subgraphs.
    :return dataset: list of PyG `Data` objects.
    """
    device = set_device()
    dataset = []
    for graph in graphs:
        edge_index = torch.from_numpy(np.array(graph['edge_index'], dtype=np.int64).T).type(torch.long).to()
        x = torch.tensor(graph['features']).to(device)
        patient_index = torch.tensor(graph['patient_index']).to(device)
        node_classify = torch.tensor(graph['F_classify']).to(device)
        edge_classify = torch.tensor(graph['edge_class']).to(device)
        dataset.append(Data(x=x, edge_index=edge_index, y=patient_index, node_classify=node_classify,
                            edge_classify=edge_classify))
    return dataset

def found_ground_truth(ground_truth, threshold):
    """
    Find the indices of patients whose similarity exceeds the threshold.
    :param ground_truth: actually relevant patients.
    :param threshold: similarity threshold.
    :return: indices of patients whose similarity exceeds the threshold.
    """
    ground_truth_index = [index for index, value in enumerate(ground_truth) if value >= threshold]
    return ground_truth_index

def hit(predict_list, ground_list):
    """
    The method to compute Hit@10.
    :param ground_list: list of relevant items (ground truth).
    :param predict_list: list of predicted items (ordered by predicted relevance).
    :return: evaluation result.
    """
    last_idx = sys.maxsize
    for idx, item in enumerate(predict_list):
        if item in ground_list:
            last_idx = idx
            break
    result = np.zeros(len(predict_list), dtype=np.float32)
    result[last_idx:] = 1.0
    return result

def precision(predict_list, ground_list):
    """
    The method to compute P@10.
    :param ground_list: List of relevant items (ground truth).
    :param predict_list: List of predicted items (ordered by predicted relevance).
    :return: evaluation result.
    """
    top_k_predictions = predict_list
    relevant_count = sum(1 for item in top_k_predictions if item in ground_list)
    precision = relevant_count / len(predict_list)

    return precision

def AP(predict_list, ground_list):
    """
    The method to compute AP@10.
    :param ground_list: List of relevant items (ground truth).
    :param predict_list: List of predicted items (ordered by predicted relevance).
    :return: evaluation result.
    """
    cu_hits = 0
    cu_precision = 0
    for i, pred_item in enumerate(predict_list, start=1):
        if pred_item in ground_list:
            cu_hits += 1
            cu_precision += cu_hits / i
    if cu_hits > 0:
        return cu_precision / cu_hits
    else:
        return 0

def dcg_at_n(predict_list, ground_list, n):
    """
    Calculate Discounted Cumulative Gain at n (DCG@n).
    :param predict_list: List of predicted items (ordered by predicted relevance).
    :param ground_list: List of relevant items (ground truth).
    :param n: Number of top results to consider.
    :return: DCG@n value.
    """
    dcg = 0.0
    for i, item in enumerate(predict_list[:n]):
        if item in ground_list:
            # rel(k) is set to 1 if the item is relevant, otherwise 0
            dcg += 1 / math.log2(i + 2)  # i+2 because rank starts at 1, and log2(1) is 0
    return dcg

def idcg_at_n(ground_list, n):
    """
    Calculate Ideal Discounted Cumulative Gain at n (IDCG@n).
    :param ground_list: List of relevant items (ground truth).
    :param n: Number of top results to consider.
    :return: IDCG@n value.
    """
    ideal_list = sorted(ground_list[:n])
    idcg = 0.0
    for i in range(len(ideal_list)):
        idcg += 1 / math.log2(i + 2)
    return idcg

def ndcg(predict_list, ground_list, n=10):
    """
    Calculate Normalized Discounted Cumulative Gain at 10 (NDCG@10).
    :param predict_list: List of predicted items (ordered by predicted relevance).
    :param ground_list: List of relevant items (ground truth).
    :param n: Number of top results to consider.
    :return: NDCG@10 value.
    """
    dcg = dcg_at_n(predict_list, ground_list, n)
    idcg = idcg_at_n(ground_list, n)
    return dcg / idcg if idcg > 0 else 0.0

def print_evaluation(hit_k,precision_k,map_k,ndcg_k):
    """
    Printing the model's evaluation results on the test dataset.
    """
    print(f"hit@10:{hit_k}")
    print(f"p@10:{precision_k}")
    print(f"map@10:{map_k}")
    print(f"ndcg@10:{ndcg_k}")

def test():
    """
    Scoring on the test set.
    """
    print("\n\nModel evaluation.\n")

    with open('./data/dataset_7d.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open('./data/sim_matrix_new.json', 'r', encoding='utf-8') as f:
        sim_matrix = json.load(f)

    model = torch.load('./models/Ai_SGP_Expert.pth',weights_only=False)

    random.seed(42)
    random.shuffle(dataset)

    train_size = int(len(dataset) * 0.7)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    patient_id = index_to_id(dataset)

    predictions = np.empty((len(test_dataset), len(train_dataset)))
    ground_truth = np.empty((len(test_dataset), len(train_dataset)))
    test_graphs = create_Data(test_dataset)
    train_graphs = create_Data(train_dataset)
    hit_k_list = []
    precision_k_list = []
    map_k_list = []
    ndcg_k_list = []
    for i, graph in tqdm(enumerate(test_graphs), total=len(test_graphs), desc="testing"):
        source_batch = Batch.from_data_list([graph] * len(train_graphs))
        target_batch = Batch.from_data_list(train_graphs)
        data = transfer_to_torch((source_batch, target_batch), patient_id, sim_matrix)
        target = data["targets"]
        prediction, _, _ = model(data)
        ground_truth[i] = target.cpu()
        predictions[i] = prediction.cpu().detach().numpy()

    for i, row in enumerate(ground_truth):
        pre = predictions[i]
        tar = row
        # set the similarity threshold is 0.5
        ground_truth = found_ground_truth(tar, 0.5)
        if len(ground_truth) >= 10:
            predict = pre.argsort()[::-1][:10]
        elif len(ground_truth) == 0:
            continue
        else:
            predict = pre.argsort()[::-1][:len(ground_truth)]
        hit_i_k = hit(predict, ground_truth)[-1]
        precision_i_k = precision(predict, ground_truth)
        map_i_k = AP(predict, ground_truth)
        ndcg_i_k = ndcg(predict, ground_truth)
        hit_k_list.append(hit_i_k)
        precision_k_list.append(precision_i_k)
        map_k_list.append(map_i_k)
        ndcg_k_list.append(ndcg_i_k)

    hit_k = np.round(np.average(np.array(hit_k_list)), 4)
    precision_k = np.round(np.average(np.array(precision_k_list)), 4)
    map_k = np.round(np.average(np.array(map_k_list)), 4)
    ndcg_k = np.round(np.average(np.array(ndcg_k_list)), 4)
    print_evaluation(hit_k, precision_k, map_k, ndcg_k)

if __name__ == '__main__':
    test()