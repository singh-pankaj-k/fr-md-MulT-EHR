from torch_geometric.data import HeteroData
import torch
from tqdm import tqdm
import pickle
import yaml
from pathlib import Path
try:
    from .utils import ordered_yaml
except (ImportError, ValueError):
    from utils import ordered_yaml

from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.tasks import (
    drug_recommendation_mimic3_fn as DrugRecommendationMIMIC3,
    drug_recommendation_mimic4_fn as DrugRecommendationMIMIC4,
    length_of_stay_prediction_mimic3_fn as LengthOfStayPredictionMIMIC3,
    length_of_stay_prediction_mimic4_fn as LengthOfStayPredictionMIMIC4,
    mortality_prediction_mimic3_fn as MortalityPredictionMIMIC3,
    mortality_prediction_mimic4_fn as MortalityPredictionMIMIC4,
    readmission_prediction_mimic3_fn as ReadmissionPredictionMIMIC3,
    readmission_prediction_mimic4_fn as ReadmissionPredictionMIMIC4
)


class GraphConstructor:
    def __init__(self, config_graph):
        """
        Initializes the GraphConstructor with the given configuration.

        Args:
            config_graph (dict): Configuration dictionary containing dataset name, paths, and dev mode flag.
        """
        self.config_graph = config_graph
        self.dataset_name = config_graph["dataset_name"]
        self.cache_path = config_graph["processed_path"]
        self.graph_path = config_graph["graph_output_path"]
        self.dev = config_graph.get("dev", False)

        self.dataset = None
        self.graph = None
        self.mappings = None

    def load_mimic(self):
        """
        Loads the MIMIC dataset (III or IV) using pyhealth based on the configuration.

        Raises:
            NotImplementedError: If the dataset path does not contain 'mimiciii' or 'mimiciv'.
        """
        print("\nRunning load_mimic()...")
        # Get mimic dataset from
        raw_path = self.config_graph["raw"]

        if "mimiciii" in raw_path:
            self.dataset = MIMIC3Dataset(
                root=raw_path,
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                dev=self.dev,
            )
        elif "mimiciv" in raw_path:
            self.dataset = MIMIC4Dataset(
                root=raw_path,
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                dev=self.dev,
            )
        else:
            raise NotImplementedError

    def construct_graph(self):
        """
        Orchestrates the construction of the heterogeneous graph.
        It retrieves graph data, initializes HeteroData, and sets edge indices.
        """
        print("\nRunning construct_graph()...")
        # Construct graph with the loaded datasets and tables
        graph_data = self.get_graph_data()
        data = HeteroData()

        # Set node counts for each node type based on mappings
        if self.mappings is not None:
            for ntype, mapping in self.mappings.items():
                data[ntype].num_nodes = len(mapping)

        # Add hetero edges
        for (head, rel, tail), (src, dst) in graph_data.items():
            edge_index = torch.stack([src, dst], dim=0)
            data[(head, rel, tail)].edge_index = edge_index

        self.graph = data

    def initialize_features(self):
        """
        Initializes node features of the graph.
        Currently a placeholder for future feature initialization logic.
        """
        print("\nRunning initialize_features()...")

        # Naive approach: randomly initialize the features

    def get_graph_data(self):
        """
        Iterates through the dataset to collect nodes and edges for the graph.

        Returns:
            dict: A dictionary mapping (head, rel, tail) tuples to tensors of edge indices.
        """
        # TODO: load gender and sex as attribute nodes to patients

        # Dictionaries of indices
        patients_dict = {}
        visits_set = set()
        diagnosis_set = set()
        procedures_set = set()
        prescriptions_set = set()
        labevents_set = set()

        patient_visit_edges = []
        visit_diagnosis_edges = []
        visit_procedure_edges = []
        visit_prescription_edges = []
        visit_labevent_edges = []

        # In pyhealth 1.1.6, we use self.dataset.patients.values()
        for patient in tqdm(self.dataset.patients.values()):
            patient_id = patient.patient_id
            if patient_id not in patients_dict:
                patients_dict[patient_id] = len(patients_dict)

            self._process_patient_admissions(
                patient, visits_set, diagnosis_set, procedures_set,
                prescriptions_set, labevents_set, patient_visit_edges,
                visit_diagnosis_edges, visit_procedure_edges,
                visit_prescription_edges, visit_labevent_edges
            )

        # Convert sets to dicts
        visits_dict = self.set_to_dict(visits_set)
        diagnosis_dict = self.set_to_dict(diagnosis_set)
        procedures_dict = self.set_to_dict(procedures_set)
        prescriptions_dict = self.set_to_dict(prescriptions_set)
        labevents_dict = self.set_to_dict(labevents_set)

        # Load graph indices
        graph_indices = self._map_edges_to_indices(
            patients_dict, visits_dict, diagnosis_dict, procedures_dict,
            prescriptions_dict, labevents_dict, patient_visit_edges,
            visit_diagnosis_edges, visit_procedure_edges,
            visit_prescription_edges, visit_labevent_edges
        )

        # Create graph data
        graph_data = self._build_graph_data_tensors(*graph_indices)

        # Save mappings
        self.mappings = {
            "patient": patients_dict,
            "visit": visits_dict,
            "diagnosis": diagnosis_dict,
            "procedure": procedures_dict,
            "prescription": prescriptions_dict,
            "labevent": labevents_dict
        }

        with open(f'{self.graph_path}{self.dataset_name}_entity_mapping.pkl', 'wb') as outp:
            pickle.dump(self.mappings, outp, pickle.HIGHEST_PROTOCOL)

        return graph_data

    def _process_patient_admissions(self, patient, visits_set, diagnosis_set, procedures_set,
                                   prescriptions_set, labevents_set, patient_visit_edges,
                                   visit_diagnosis_edges, visit_procedure_edges,
                                   visit_prescription_edges, visit_labevent_edges):
        """
        Processes all admissions for a single patient.
        """
        for visit_id, visit in patient.visits.items():
            visits_set.add(visit_id)
            patient_visit_edges.append((patient.patient_id, visit_id))

            # Load diagnoses
            if "diagnoses_icd" in visit.event_list_dict:
                for ev in visit.event_list_dict["diagnoses_icd"]:
                    code = ev.code
                    diagnosis_set.add(code)
                    visit_diagnosis_edges.append((visit_id, code))

            # Load procedures
            if "procedures_icd" in visit.event_list_dict:
                for ev in visit.event_list_dict["procedures_icd"]:
                    code = ev.code
                    procedures_set.add(code)
                    visit_procedure_edges.append((visit_id, code))

            # Load prescriptions
            if "prescriptions" in visit.event_list_dict:
                for ev in visit.event_list_dict["prescriptions"]:
                    code = ev.code
                    prescriptions_set.add(code)
                    visit_prescription_edges.append((visit_id, code))

            # Load labevents
            if "labevents" in visit.event_list_dict:
                for ev in visit.event_list_dict["labevents"]:
                    code = ev.code
                    labevents_set.add(code)
                    visit_labevent_edges.append((visit_id, code))

    def _map_edges_to_indices(self, patients_dict, visits_dict, diagnosis_dict, procedures_dict,
                             prescriptions_dict, labevents_dict, patient_visit_edges,
                             visit_diagnosis_edges, visit_procedure_edges,
                             visit_prescription_edges, visit_labevent_edges):
        """
        Maps raw ID edges to integer index edges using the provided dictionaries.
        """
        pv_idx = [(patients_dict[p], visits_dict[v]) for (p, v) in patient_visit_edges]
        vd_idx = [(visits_dict[v], diagnosis_dict[d]) for (v, d) in visit_diagnosis_edges]
        vl_idx = [(visits_dict[v], labevents_dict[l]) for (v, l) in visit_labevent_edges]
        vp_idx = [(visits_dict[v], procedures_dict[p]) for (v, p) in visit_procedure_edges]
        vr_idx = [(visits_dict[v], prescriptions_dict[r]) for (v, r) in visit_prescription_edges]
        return pv_idx, vd_idx, vl_idx, vp_idx, vr_idx

    def _build_graph_data_tensors(self, pv_idx, vd_idx, vl_idx, vp_idx, vr_idx):
        """
        Converts indexed edge lists into PyTorch tensors and returns the graph data dictionary.
        """
        graph_data = {}

        def update_edges(head, rel, tail, edges):
            if not edges:
                graph_data[(head, rel, tail)] = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                return
            graph_data.update({
                (head, rel, tail): (
                    torch.tensor([e[0] for e in edges], dtype=torch.long),
                    torch.tensor([e[1] for e in edges], dtype=torch.long)
                )
            })

        update_edges("patient", "makes", "visit", pv_idx)
        update_edges("visit", "diagnosed", "diagnosis", vd_idx)
        update_edges("visit", "prescribed", "prescription", vr_idx)
        update_edges("visit", "treated", "procedure", vp_idx)
        update_edges("visit", "occurs", "labevent", vl_idx)

        return graph_data

    def set_tasks(self):
        """
        Extracts task labels (mortality, drug recommendation, LOS, readmission) from the dataset
        and saves them to a pickle file.
        """
        print("\nRunning set_tasks()...")

        mort_pred_samples, drug_rec_samples, los_samples, readm_samples = self.get_sample_datasets()
        vm = self.mappings["visit"]

        # Assign labels for each task
        # PyHealth task functions usually return labels with 'label' key
        mort_pred = self._extract_task_labels(mort_pred_samples, vm, "label")
        drug_rec = self._extract_task_labels(drug_rec_samples, vm, "drugs")
        los = self._extract_task_labels(los_samples, vm, "label")
        readm = self._extract_task_labels(readm_samples, vm, "label")

        # Get all tokens for drugs from output_processors
        # For SampleEHRDataset in newer PyHealth, we can try to get it from the samples
        all_drugs = set()
        for s in drug_rec_samples:
            for d in s["drugs"]:
                all_drugs.add(d)
        all_drugs = list(all_drugs)

        labels = {
            "mort_pred": mort_pred,
            "drug_rec": drug_rec,
            "all_drugs": all_drugs,
            "los": los,
            "readm": readm
        }

        self.save_labels(labels)

    def _extract_task_labels(self, samples, visit_mapping, label_key):
        """
        Extracts labels for a specific task and maps them to visit indices.

        Args:
            samples: List of samples from pyhealth task.
            visit_mapping (dict): Mapping from visit_id to index.
            label_key (str): The key in the sample dictionary representing the label.

        Returns:
            dict: Mapping from visit index to label.
        """
        task_labels = {}
        for s in samples:
            visit_id = s.get("visit_id") or s.get("admission_id")
            if visit_id in visit_mapping:
                task_labels[visit_mapping[visit_id]] = s[label_key]
        return task_labels

    def get_sample_datasets(self):
        """
        Retrieves sample datasets for all prediction tasks using pyhealth.

        Returns:
            tuple: (mort_pred_samples, drug_rec_samples, los_samples, readm_samples)
        """
        if "mimic3" in self.dataset_name:
            mort_pred_samples = self.dataset.set_task(task_fn=MortalityPredictionMIMIC3)
            drug_rec_samples = self.dataset.set_task(task_fn=DrugRecommendationMIMIC3)
            los_samples = self.dataset.set_task(task_fn=LengthOfStayPredictionMIMIC3)
            readm_samples = self.dataset.set_task(task_fn=ReadmissionPredictionMIMIC3)
        elif "mimic4" in self.dataset_name:
            mort_pred_samples = self.dataset.set_task(task_fn=MortalityPredictionMIMIC4)
            drug_rec_samples = self.dataset.set_task(task_fn=DrugRecommendationMIMIC4)
            los_samples = self.dataset.set_task(task_fn=LengthOfStayPredictionMIMIC4)
            readm_samples = self.dataset.set_task(task_fn=ReadmissionPredictionMIMIC4)
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        return mort_pred_samples, drug_rec_samples, los_samples, readm_samples

    def get_mimic_dataset(self):
        """
        Loads the cached pyhealth MIMIC dataset.

        Returns:
            MIMIC3Dataset or MIMIC4Dataset: The loaded dataset.
        """
        with open(f'{self.cache_path}{self.dataset_name}', 'rb') as inp:
            unp = pickle.Unpickler(inp)
            mimic_ds = unp.load()

        return mimic_ds

    def save_mimic_dataset(self, mimic_ds=None):
        """
        Saves the pyhealth MIMIC dataset to cache.

        Args:
            mimic_ds: The dataset to save. Defaults to self.dataset.
        """
        print("\nRunning save_mimic_dataset()...")
        if mimic_ds is None:
            mimic_ds = self.dataset
        # Save a copy to cache
        with open(f'{self.cache_path}{self.dataset_name}_dataset.pkl', 'wb') as outp:
            pickle.dump(mimic_ds, outp, pickle.HIGHEST_PROTOCOL)

    def save_graph(self):
        """Saves the constructed HeteroData graph to a pickle file."""
        print("\nRunning save_graph()...")
        print(f'{self.graph_path}{self.dataset_name}_graph.pkl')
        with open(f'{self.graph_path}{self.dataset_name}_graph.pkl', 'wb') as outp:
            pickle.dump(self.graph, outp, pickle.HIGHEST_PROTOCOL)

    def load_graph(self):
        """Loads the HeteroData graph from a pickle file."""
        with open(f'{self.graph_path}{self.dataset_name}', 'rb') as inp:
            unp = pickle.Unpickler(inp)
            g = unp.load()

        return g

    def save_labels(self, labels):
        """Saves the task labels to a pickle file."""
        with open(f'{self.graph_path}{self.dataset_name}_labels.pkl', 'wb') as outp:
            pickle.dump(labels, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def set_to_dict(s):
        """
        Converts a set of unique elements into a dictionary mapping each element to a unique index.

        Args:
            s (set): The set of elements.

        Returns:
            dict: Mapping from element to index.
        """
        return {e: i for i, e in enumerate(s)}


def main():
    opt_path = "construct_graph/MIMIC4.yml"
    opt_path = Path("./configs") / opt_path
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    graph_constructor = GraphConstructor(config)

    graph_constructor.load_mimic()
    graph_constructor.construct_graph()
    graph_constructor.set_tasks()
    graph_constructor.initialize_features()
    graph_constructor.save_graph()
    graph_constructor.save_mimic_dataset()


if __name__ == '__main__':
    main()
