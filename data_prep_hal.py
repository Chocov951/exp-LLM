import json
import os
import random

def write_jsonl(data, output_file, corpus=True):
    with open(output_file, 'w', encoding='utf-8') as f:
        for id, content in data.items():
            if corpus:
                entry = {
                    "_id": id,
                    "title": content.get("title", ""),
                    "text": content.get("text", ""),
                    "metadata": {}
                }
            else:
                entry = {
                    "_id": id,
                    "text": content,
                    "metadata": {}
                }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def write_qrels_tsv(qrels_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("query-id\tcorpus-id\tscore\n")
        # Write data
        for query_id, doc_scores in qrels_dict.items():
            for doc_id, score in doc_scores.items():
                f.write(f"{query_id}\t{doc_id}\t{score}\n")

def process_files(file1, file2):
    # Initialize output structures
    corpus = {}
    queries = {}
    qrels = {}

    # Read and process both input files
    input_files = [file1, file2]
    for i, input_file in enumerate(input_files):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Process each document
            for doc in data:
                if 'en' not in doc:
                    doc_id = doc['docid']
                    
                    # Add to corpus with empty title and abstract as text
                    corpus[f"d{doc_id}"] = {
                        "title": "",
                        "text": doc['abstract']
                    }
                    
                    if i == 0:                
                        # Add to queries mapping docid to title
                        queries[f"q{doc_id}"] = doc['title']
                        
                        # Add to qrels mapping query ID to doc ID with relevance 1
                        qrels[f"q{doc_id}"] = {f"d{doc_id}": 1}

    # Split qrels randomly (80% train, 20% test)
    print(f"Total queries: {len(qrels)}")
    qrels_items = list(qrels.items())
    random.shuffle(qrels_items)
    split_point = 400
    
    train_qrels = dict(qrels_items[:split_point])
    print(f"Train queries: {len(train_qrels)}")
    test_qrels = dict(qrels_items[split_point:])
    print(f"Test queries: {len(test_qrels)}")

    print(f"Total documents: {len(corpus)}")

    # Write outputs to files
    write_jsonl(corpus, 'hal/corpus.jsonl')
    write_jsonl(queries, 'hal/queries.jsonl', False)
    
    # Write qrels in TSV format
    # if qrels directory does not exist, create it
    if not os.path.exists('hal/qrels'):
        os.makedirs('hal/qrels')
    write_qrels_tsv(train_qrels, 'hal/qrels/train.tsv')
    write_qrels_tsv(test_qrels, 'hal/qrels/test.tsv')

# Process the files
process_files('hal/hal_extract_24.json',
             'hal/hal_extract_21-23.json')