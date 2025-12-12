import json
import sys
from py2neo import Graph
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from param_parser import parameter_parser
from train import DIKGSPTrainer

graph = Graph("address", auth=("UserName","PassWord"), name='Neo4j')

class patient_retrival:
    """
    Find similar patients for the query patient.
    """
    def __init__(self, model, inquiry_patient,inquiry_patientID, dataset):
        self.model = model
        self.dataset = dataset
        self.inquiry_patient = inquiry_patient
        self.inquiry_patientID = inquiry_patientID
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.process_data()

    def process_data(self):
        edge_index = torch.from_numpy(np.array(self.inquiry_patient['edge_index'], dtype=np.int64).T).type(
            torch.long).to(
            self.device)
        x = torch.tensor(self.inquiry_patient['features']).to(self.device)
        F_classify = torch.tensor(self.inquiry_patient['F_classify']).to(self.device)
        edge_class = torch.tensor(self.inquiry_patient['edge_class']).to(self.device)
        self.patientIDs = self.index_to_id(self.dataset)
        self.inquiry_patient = Data(x=x, edge_index=edge_index, node_classify=F_classify, edge_classify=edge_class)
        self.P_index, self.dataset = self.create_Data(self.dataset)

    def index_to_id(self, graphs):
        patient_ids = {patient['patient_index']: patient['patient_id'] for patient in graphs}
        return patient_ids

    def create_Data(self, graphs):
        dataset = []
        P_index = []
        for graph in graphs:
            edge_index = torch.from_numpy(np.array(graph['edge_index'], dtype=np.int64).T).type(torch.long).to(
                self.device)
            x = torch.tensor(graph['features']).to(self.device)
            patient_index = torch.tensor(graph['patient_index']).to(self.device)
            node_classify = torch.tensor(graph['F_classify']).to(self.device)
            edge_classify = torch.tensor(graph['edge_class']).to(self.device)
            dataset.append(Data(x=x, edge_index=edge_index, y=patient_index, node_classify=node_classify,
                                edge_classify=edge_classify))
            P_index.append(patient_index.item())
        return P_index, dataset

    def transfer_to_torch(self, data):
        new_data = dict()
        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        return new_data

    def similar_dic(self):
        """
        The model predicts similarity scores.
        :return: similar patients.
        """
        patients_similarity = {}
        source_batch = Batch.from_data_list([self.inquiry_patient] * len(self.dataset))
        target_batch = Batch.from_data_list(self.dataset)
        data = self.transfer_to_torch((source_batch, target_batch))
        similarity_list,_,_ = self.model(data)
        patient_to_index = list(zip(self.P_index, similarity_list.tolist()))
        for patient in patient_to_index:
            p_id = self.patientIDs[patient[0]]
            similarity = patient[1]
            patients_similarity[p_id] = similarity

        return patients_similarity

    def retrieval(self):
        patients_similarity = self.similar_dic()
        sorted_keys = sorted(patients_similarity, key=patients_similarity.get, reverse=True)
        return sorted_keys[:3]

    def prescription(self):
        """
        Retrieve from the graph database, filter similar patients,
        and obtain both the medication information of the similar patients,
        and the actual medication information of the query patient.
        :return retrieval_patients: similar patients.
        :return herbs: the herbs of similar patients.
        :return herb_physician: true herbs.
        """
        retrieval_patients = self.retrieval()
        similar_patient = retrieval_patients[0]
        query1 = 'match(p:患者)-[r:服用]->(q:药物) where p.患者编号="%s" return q.name AS 草药, r.剂量 AS 剂量' % (
            similar_patient)
        query2 = 'match(p:患者)-[r:服用]->(q:药物) where p.患者编号="%s" return q.name AS 草药, r.剂量 AS 剂量' % (
            self.inquiry_patientID)
        results = graph.run(query1)
        origin = graph.run(query2)
        herb_physician = []
        for o in origin:
            herb_physician.append(o['草药'])
        herbs = []
        for record in results:
            herbs.append(record['草药'])
        return retrieval_patients, len(set(herbs)), herbs,herb_physician


def IO(herbs_herbs):
    """
    Evaluate the IO value of the model-predicted drugs.
    :param herbs_herbs: ground-truth and predicted herb pairs
    :return: the value of IO.
    """
    ingredient_overlap = 0
    for herbs in herbs_herbs:
        ingredient_overlap += len(herbs[0] & herbs[1])/len(herbs[1])
    return ingredient_overlap/len(herbs_herbs)


def main():
    file_path_1 = './data/patient_subgroup.json'

    with open('./data/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(file_path_1, 'r', encoding='utf-8') as f:
        subgraphs = json.load(f)

    args = parameter_parser()
    trainer = DIKGSPTrainer(args)
    load_path = './models/DIKGSP_SGP_Expert.pth'
    trainer.load_model(load_path)

    trainer.model.eval()

    herbs_herbs = []
    for patient in subgraphs:
        patientID = patient['patient_id']
        similar_instances = patient_retrival(model=trainer.model, inquiry_patient=patient,inquiry_patientID = patientID, dataset=dataset)
        patients, numbers, herbs,herbs_physician = similar_instances.prescription()
        physician_herbs_dose = herbs_physician
        re_herbs_dose = herbs
        herbs_herbs.append((set(re_herbs_dose),set(physician_herbs_dose)))
    Io = round(IO(herbs_herbs),4)
    print(Io)




if __name__ == '__main__':
    main()
