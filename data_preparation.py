import os
import json
from beir.datasets.data_loader import GenericDataLoader

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
        data_path = f"{self.dataset}"
        self.corpus_train, self.queries_train, self.qrels_train = GenericDataLoader(data_path).load(split="train")
        self.corpus_test, self.queries_test, self.qrels_test = GenericDataLoader(data_path).load(split="test")
        
    def prepare_data(self):
        for query_id, docs in self.qrels_train.items():
            for doc_id in docs:
                self.json_train.append(prepared_data(PROMPT1, self.queries_train[query_id], self.corpus_train[doc_id]['text'], "", []).to_dict())
        for query_id, docs in self.qrels_test.items():
            for doc_id in docs:
                self.json_dev.append(prepared_data(PROMPT1, self.queries_test[query_id], self.corpus_test[doc_id]['text'], "", []).to_dict())

if __name__ == '__main__':
    dataset = 'hal'
    jd = json_data(dataset)
    jd.load_dataset()
    jd.prepare_data()
    if not os.path.exists(dataset):
        os.makedirs(dataset)
    with open(f"{dataset}/{dataset}_dev.json", "w", encoding="utf-8") as f:
        json.dump(jd.json_dev, f, indent=2, ensure_ascii=False)
    with open(f"{dataset}/{dataset}_train.json", "w", encoding="utf-8") as f:
        json.dump(jd.json_train, f, indent=2, ensure_ascii=False)
