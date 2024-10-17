import os
import json
from beir.datasets.data_loader import GenericDataLoader
import pickle as pkl
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import re

PROMPT1 = """Give me a QUERY that could be answered by the following text:"""

class prepared_data:
    def __init__(self, instruction, output, input="", system="", history=[]):
        self.instruction = instruction
        self.input = input
        self.output = output
        self.system = system
        self.history = history

    def to_dict(self):
        return vars(self)
    
class json_data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.json_dev = []
        self.json_train = []
        self.corpus_train = {}
        self.queries_train = {}
        self.qrels_train = {}
        self.corpus_test = {}
        self.queries_test = {}
        self.qrels_test = {}
    
    def load_dataset(self):
        self.corpus_train, self.queries_train, self.qrels_train = pkl.load(open(f"iso_entreprise/dataset_iso.pkl", "rb"))
        self.corpus_test, self.queries_test, self.qrels_test = pkl.load(open(f"iso_entreprise/dataset_iso.pkl", "rb"))

    def augment_data(self):
        aug = naw.ContextualWordEmbsAug(model_path='flaubert/flaubert_base_uncased', action="insert", aug_p=0.2)
        aug2 = naw.ContextualWordEmbsAug(model_path='flaubert/flaubert_base_uncased', action="substitute", aug_p=0.2)
        aug3 = naw.SynonymAug(aug_src='wordnet', lang='fra', aug_p=0.2)
        seen_docs = []
        for query_id, docs in self.qrels_train.items():
            for doc_id in docs:
                seen_docs.append(doc_id)
                text = self.corpus_train[doc_id]
                text = re.sub(r'\d+', '', text) # remove numbers from text
                self.json_train.append(prepared_data(PROMPT1, self.queries_train[query_id], aug.augment(text)[0], "", []).to_dict())
                self.json_train.append(prepared_data(PROMPT1, self.queries_train[query_id], aug2.augment(text)[0], "", []).to_dict())
                self.json_train.append(prepared_data(PROMPT1, self.queries_train[query_id], aug3.augment(text)[0], "", []).to_dict())
        # for doc_id in self.corpus_train.keys():
        #     if doc_id not in seen_docs:
        #         text = self.corpus_train[doc_id]
        #         text = re.sub(r'\d+', '', text) # remove numbers from text
        #         self.json_train.append(prepared_data(PROMPT1, "NONE", aug.augment(text)[0], "", []).to_dict())
        #         self.json_train.append(prepared_data(PROMPT1, "NONE", aug2.augment(text)[0], "", []).to_dict())
        #         self.json_train.append(prepared_data(PROMPT1, "NONE", aug3.augment(text)[0], "", []).to_dict())

        
        
    def prepare_data(self):
        seen_docs = []
        # for query_id, docs in self.qrels_train.items():
        #     for doc_id in docs:
        #         self.json_train.append(prepared_data(PROMPT1, self.queries_train[query_id], self.corpus_train[doc_id], "", []).to_dict())
        for query_id, docs in self.qrels_test.items():
            for doc_id in docs:
                seen_docs.append(doc_id)
                self.json_dev.append(prepared_data(PROMPT1, self.queries_test[query_id], self.corpus_test[doc_id], "", []).to_dict())
        # for doc_id in self.corpus_test.keys():
        #     if doc_id not in seen_docs:
        #         self.json_dev.append(prepared_data(PROMPT1, "NONE", self.corpus_test[doc_id], "", []).to_dict())


if __name__ == '__main__':
    import nltk
    nltk.download('omw-1.4')
    dataset = 'iso'
    jd = json_data(dataset)
    jd.load_dataset()
    jd.augment_data()
    jd.prepare_data()
    with open(f"iso_entreprise/{dataset}_dev.json", "w", encoding="utf-8") as f:
        json.dump(jd.json_dev, f, indent=2, ensure_ascii=False)
    with open(f"iso_entreprise/{dataset}_train.json", "w", encoding="utf-8") as f:
        json.dump(jd.json_train, f, indent=2, ensure_ascii=False)
