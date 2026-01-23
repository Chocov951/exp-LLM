import csv
import gzip
import tarfile
import json
import time
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModel
import torch

from datasets import Dataset
import string
import pickle as pkl
from beir.datasets.data_loader import GenericDataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import os
from scipy.spatial.distance import cosine
from pyserini.search.lucene import LuceneSearcher
from codecarbon import OfflineEmissionsTracker

SEPARATOR = "### Answer:\n"

def preprocess(txt):
    return set(txt.translate(str.maketrans('', '', string.punctuation)).lower().split())

def create_model_config(model_name):
    if model_name == 'qwen3':
        model_path = 'models/Qwen2.5-3B-Instruct'
        max_length = 32768
        quant=True
    elif model_name == 'qwen14':
        model_path = 'models/Qwen2.5-14B-Instruct'
        max_length = 32768
        quant=True
    elif model_name == 'qwen32':
        model_path = 'models/Qwen2.5-32B-Instruct'
        max_length = 32768
        quant=True
    elif model_name == 'qwen72':
        model_path = 'models/Qwen2.5-72B-Instruct'
        max_length = 32768
        quant=True
    elif model_name == 'calme':
        model_path = 'models/calme-3.1-instruct-78b'
        max_length = 32768
        quant=True
    else:
        raise ValueError(f"Model {model_name} not supported")
    # Configure the quantization settings
    if quant:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    else:
        quantization_config = None
        
    return model_path, max_length, quantization_config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model to use", default="mistral",
                        choices=["qwen3", "qwen14", "qwen32", "qwen72"], type=str)
    parser.add_argument("--dataset", help="dataset to test", type=str, default="scifact")
    parser.add_argument("--bm25_topk", type=int, default=100, help="Topk BM25 (or BERT) to input to the LLM")
    parser.add_argument("--bert_model", type=str, default="BAAI/bge-m3", help="BERT model to use")
    parser.add_argument("--reject_number", type=int, default=5, help="Number of passages to keep in the reject phase")
    return parser.parse_args()

def estimate_tokens(text):
    return len(text.split()) * 1.3  # Estimation approximative

def split_queries_by_max_length(texts, max_length, max_query_length, prompts):
    current_batch = []
    current_length = 0
    all_batches = []
    prompt_length = (len(prompts['S-prompt'])+len(prompts['U-prompt-1'])+len(prompts['U-prompt-2'])+len(prompts['U-prompt-3'])) * 1.3
    text_lengths = [len(text.split()) * 1.3 for text in texts]
    print(f"Taille moyenne des passages : {sum(text_lengths)/len(text_lengths):.2f} tokens")
    prompt_length += max_query_length
    max_length -= prompt_length
    max_length -= 20
    print(f"Max length pour les queries : {max_length}")

    for text, tokens_length in zip(texts, text_lengths):        
        if tokens_length > max_length:
            print(f"Warning: Text with {tokens_length} tokens exceeds max_length of {max_length}")
            continue
            
        if current_length + tokens_length > max_length:
            if current_batch:
                all_batches.append(current_batch)
            current_batch = [text]
            current_length = tokens_length
        else:
            current_batch.append(text)
            current_length += tokens_length
    
    if current_batch:
        all_batches.append(current_batch)
        
    return all_batches, max(text_lengths)

def get_top_k_bm25(query, bm25, k):
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = scores.argsort()[-k:][::-1]
    top_k_scores = scores[top_k_indices]
    return top_k_indices, top_k_scores

def compute_bm25_rundict(queries, k):
    lucene_bm25_searcher = LuceneSearcher('datasets/trec/lucene-inverted.msmarco-v1-passage.20221004.252b5e')
    q_test = {}
    rundict = {}
    for qid, query in tqdm(queries.items()):
        hits = lucene_bm25_searcher.search(query['query'], k)
        rundict[qid] = {}
        q_test[qid] = {'query': query['query'], 'passages': {}}
        for i in range(len(hits)):
            doc_id = hits[i].docid
            score = hits[i].score
            rundict[qid][doc_id] = score
            q_test[qid]['passages'][doc_id] = hits[i].lucene_document.get('raw')
    return rundict, q_test

def get_mean_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    mean_embedding = last_hidden_state.mean(dim=1)
    return mean_embedding

def compute_similarity_rundict(queries, corpus, model, tokenizer):
    rundict = {}
    print("\nComputing corpus embeddings...")
    corpus_embeddings = {doc_id: get_mean_embedding(text, model, tokenizer) for doc_id, text in tqdm(corpus.items())}
    
    print("\nComputing query + similarity scores...")
    for query_id, query_text in tqdm(queries.items()):
        query_embedding = get_mean_embedding(query_text, model, tokenizer)
        similarities = {}
        for doc_id, doc_embedding in corpus_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding).item()
            similarities[str(doc_id)] = similarity
        rundict[str(query_id)] = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    
    return rundict

def get_bert_embeddings(texts, model, tokenizer, device='cuda', batch_size=1024):
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
            
        # Libérer la mémoire CUDA
        del encoded, outputs
        torch.cuda.empty_cache()
    
    return embeddings

def compute_bert_rundict(queries, corpus, model_name):
    jz_model_name = "models--" + model_name.replace("/","--")
    jz_path = os.path.join("/linkhome/rech/genlir01/uep39vh/.cache/huggingface/hub/", jz_model_name, "snapshots/")
    jz_path = os.path.join(jz_path, os.listdir(jz_path)[0])
    tokenizer = AutoTokenizer.from_pretrained(jz_path)
    model = AutoModel.from_pretrained(jz_path).to('cuda')
    rundict = {}
    if os.path.exists(f'datasets/corpus_embeddings_trec.pkl'):
        with open(f'datasets/corpus_embeddings_trec.pkl', 'rb') as f:
            corpus_embeddings = pkl.load(f)
    else:
        print("\nComputing corpus embeddings...")
        corpus_texts = [corpus[did] for did in corpus]
        corpus_embeddings = get_bert_embeddings(corpus_texts, model, tokenizer)
        # save corpus embeddings in a pickle file
        # with open(f'datasets/corpus_embeddings_trec.pkl', 'wb') as f:
        #     pkl.dump(corpus_embeddings, f)
    
    print("\nComputing query + similarity scores...")
    for qid, query in tqdm(queries.items(), desc="Computing query embeddings"):
        query_embedding = get_bert_embeddings([query], model, tokenizer)[0]
        
        similarities = []
        for i, doc_embedding in enumerate(corpus_embeddings):
            sim = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((list(corpus.keys())[i], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Only keep top 1000 results
        similarities = similarities[:1000]
        rundict[str(qid)] = {str(did): score for did, score in similarities}
    
    return rundict

if __name__ == '__main__':
    args = get_args()
    print(args)

    model_path, max_length, quantization_config = create_model_config(args.model_name)
    
    # Set torch configurations for faster loading
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Loading model {model_path}...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quantization_config,
        max_length=max_length,
    )
    
    # # Move model to GPU explicitly if needed
    # if torch.cuda.is_available():
    #     model = model.cuda()
    
    # Create chat pipeline instead of text-generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    print("Model loaded in {:.2f}s".format(time.time() - start_time))

    # Load dataset    
    if args.dataset == 'trec19' or args.dataset == 'trec20':
        corpus_test, queries_test, qrels_test, qrels_test_bin = {}, {}, {}, {}
        year = args.dataset[-2:]
        # with tarfile.open("datasets/trec/collection.tar.gz", "r:gz") as tar:
        #     for member in tar:
        #         if member.isfile() and member.name.endswith(".tsv"):
        #             with tar.extractfile(member) as f:
        #                 for line in f:
        #                     pid, passage = line.decode("utf-8").strip().split("\t", 1)
        #                     pid = str(pid)
        #                     corpus_test[pid] = passage
        with gzip.open(f"datasets/trec/{year}/msmarco-passagetest20{year}-top1000.tsv.gz", "rt", encoding="utf-8") as f:
            for line in f:
                qid, pid, query, passage = line.strip().split("\t", 3)
                qid = str(qid)
                pid = str(pid)
                if qid not in queries_test:
                    queries_test[qid] = {'query': query,
                                         'passages': {}}
                queries_test[qid]['passages'][pid] = passage
        with open(f"datasets/trec/{year}/20{year}qrels-pass.txt", "r", encoding="utf-8") as f:
            for line in f:
                qid, _, docid, rating = line.strip().split()
                qid = str(qid)
                docid = str(docid)
                rating = int(rating)
                if qid not in qrels_test:
                    qrels_test[qid] = {}
                qrels_test[qid][docid] = rating
                if rating > 1:
                    if qid not in qrels_test_bin:
                        qrels_test_bin[qid] = {}
                    qrels_test_bin[qid][docid] = rating
    else:
        try:
            corpus_test, queries_test, qrels_test = GenericDataLoader(f"datasets/{args.dataset}").load(split="test")
        except:
            try:
                corpus_test, queries_test, qrels_test = GenericDataLoader(f"datasets/../datasets/{args.dataset}").load(split="test")
            except:
                print("Dataset not found")
                exit(1)
        corpus_test = {k: v['text'] for k, v in corpus_test.items()}
    
    # Load prompts
    with open(f"prompts.json", encoding='utf-8') as f:
        prompts = json.load(f)
        if args.dataset == 'trec19' or args.dataset == 'trec20':
            prompts = prompts['trec']
        else:
            prompts = prompts[args.dataset]
        # Replace --NUMBER with reject_number in prompts
        prompts['S-prompt'] = prompts['S-prompt'].replace('--NUMBER', str(args.reject_number))
        prompts['U-prompt-3'] = prompts['U-prompt-3'].replace('--NUMBER', str(args.reject_number))

    # Remplacer le calcul de la moyenne par le découpage en batchs
    max_query_length = max(estimate_tokens(query['query']) for query in queries_test.values())

    gen_res_time = 0
    gen_res_count = 0
    gen_rerank_time = 0
    gen_rerank_count = 0

    # Initialize the tracker
    emissions_folder = 'emissions'
    os.makedirs(emissions_folder, exist_ok=True)
    # Rerank
    emissions_file_name = f"emissions_rerank_{args.dataset}_{args.model_name}_rerank_window_topk{args.bm25_topk}_cache.csv"
    tracker_rerank = OfflineEmissionsTracker(country_iso_code="FRA", tracking_mode="process", on_csv_write="append",
                                      project_name="LLM-Rerank", log_level="warning", output_dir=emissions_folder, output_file=emissions_file_name)

    def generate_responses(input, passages, corpus_texts):
        responses = {}
        global gen_res_time
        plist_s = {str(i): passage for i, passage in enumerate(passages)}
        try:
            messages=[
                {"role": "system", "content": prompts['S-prompt']},
                {"role": "user", "content": f"{prompts['U-prompt-1']}{plist_s}\n\n{prompts['U-prompt-2']}{input}\n\n{prompts['U-prompt-3']}"}
            ]
            start_time = time.time()
            outputs = pipe(messages, max_new_tokens=8192, num_return_sequences=1, do_sample=False)
            gen_res_time += time.time() - start_time
            llm_rep = outputs[0]['generated_text'][-1]['content']
        except Exception as e:
            print(e)
            llm_rep = {}

        print(f"- Generated response : {llm_rep}")
        try:
            llm_rep = llm_rep.split("{")[1].split("}")[0]
            llm_rep = '{' + llm_rep + '}'
            llm_rep = llm_rep.replace("'",'"')
            llm_rep = json.loads(llm_rep)
        except Exception as e:
            print(e)
            llm_rep = {}

        if len(llm_rep) > 0:
            for rep_id, score in llm_rep.items():
                if rep_id in plist_s:
                    rep = plist_s[rep_id]
                    responses[corpus_texts.index(rep)] = score
                    
        return responses
    
    def generate_rerank(query, docs):
        responses = {}
        global gen_rerank_time

        try:
            messages=[
                {"role": "system", "content": prompts['rerank-S-prompt']},
                {"role": "user", "content": f"{prompts['rerank-U-prompt-1']}{docs}\n\n{prompts['rerank-U-prompt-2']}{query}\n\n{prompts['rerank-U-prompt-3']}"}
            ]
            start_time = time.time()
            outputs = pipe(messages, max_new_tokens=8192, num_return_sequences=1, do_sample=False)
            gen_rerank_time += time.time() - start_time
            llm_rep = outputs[0]['generated_text'][-1]['content']
        except Exception as e:
            print(e)
            llm_rep = '[]'
        
        print(f"Generated response : {llm_rep}")
        llm_rep = llm_rep.split("[")[1].split("]")[0].replace(' ','').split(",")
        for i, id in enumerate(llm_rep):
            id = id.replace("'",'').replace('"','')
            try:
                doc_id = int(id)
                responses[str(doc_id)] = len(llm_rep) - i
            except:
                continue
        return responses

    # Compute or load cosine similarity
    if os.path.exists(f'rundicts/rundict_{args.dataset}_bm25.json'):
        with open(f'rundicts/rundict_{args.dataset}_bm25.json', 'r', encoding='utf-8') as f:
            bm25_rundict = json.load(f)
        with open(f'rundicts/queries_test_{args.dataset}_bm25.json', 'r', encoding='utf-8') as f:
            queries_test = json.load(f)
        print("\nCosine similarity loaded\n")
    else:
        bm25_rundict, queries_test = compute_bm25_rundict(queries_test, 1000)
        print("\nCosine similarity done\n")
        # Save cosine rundict
        with open(f'rundicts/rundict_{args.dataset}_bm25.json', 'w', encoding='utf-8') as f:
            json.dump(bm25_rundict, f, indent=4)
        with open(f'rundicts/queries_test_{args.dataset}_bm25.json', 'w', encoding='utf-8') as f:
            json.dump(queries_test, f, indent=4)

    # # Only one time :
    # true_bm25_rundict = {}
    # for qid, docs in bm25_rundict.items():
    #     for docid, score in docs.items():
    #         if qid not in true_bm25_rundict:
    #             true_bm25_rundict[qid] = {}
    #         true_bm25_rundict[qid][corpus_ids[int(docid)]] = score
    # bm25_rundict = true_bm25_rundict
    # with open(f'rundicts/rundict_{args.dataset}_bm25.json', 'w', encoding='utf-8') as f:
    #         json.dump(bm25_rundict, f, indent=4)

    if os.path.exists(f'rundicts/rundict_{args.dataset}_{args.model_name}_rerank_window_topk{args.bm25_topk}_cache.json'):
        with open(f'rundicts/rundict_{args.dataset}_{args.model_name}_rerank_window_topk{args.bm25_topk}_cache.json', 'r', encoding='utf-8') as f:
            rundict = json.load(f)
            rundict_rerank = rundict.copy()
    else:
        rundict = {}
        queries_in_qrels = [query_id for query_id in qrels_test.keys()]
        tracker_rerank.start()
        for tracker_index, qkey in tqdm(enumerate(queries_in_qrels)):
            qtext = queries_test[qkey]['query']
            corpus_test = queries_test[qkey]['passages']
            corpus_texts = list(corpus_test.values())
            corpus_ids = list(corpus_test.keys())
            responses = {}                
            # Top k cosine similarity / bm25
            top_k_passages_ids = list(bm25_rundict[str(qkey)].keys())[:args.bm25_topk]
            top_k_passages =  {doc_id: corpus_test[doc_id] for doc_id in top_k_passages_ids}

            reranked_passages_ids = top_k_passages_ids.copy() 
            nb_passages = len(reranked_passages_ids)
            for window in range(0, nb_passages-10, 10):
                window_passages = reranked_passages_ids[nb_passages-window-20:nb_passages-window]
                passages = {doc_id: corpus_test[doc_id] for doc_id in window_passages}
                responses = generate_rerank(qtext, passages)
                new_responses = [r for r in responses.keys() if r in reranked_passages_ids]
                # replace the selected window passages with the reranked ones
                reranked_passages_ids[nb_passages-window-20:nb_passages-window] = new_responses
            gen_rerank_count += 1
            print("--------------------")

            qid = str(qkey)
            
            for i, did in enumerate(reranked_passages_ids):
                if qid not in rundict:
                    rundict[qid] = {}
                if did not in rundict[qid]:
                    rundict[qid][did] = len(reranked_passages_ids) - i
                else:
                    rundict[qid][did] += len(reranked_passages_ids) - i
        tracker_rerank.stop()
        rundict_rerank = rundict.copy()
        # Save rerank rundict
        with open(f'rundicts/rundict_{args.dataset}_{args.model_name}_rerank_window_topk{args.bm25_topk}_cache.json', 'w', encoding='utf-8') as f:
            json.dump(rundict_rerank, f, indent=4)

    # Print results:
    metrics = ['recall@1', 'recall@2', 'recall@3', 'recall@4', 'recall@5',
               'ndcg@1', 'ndcg@2', 'ndcg@3', 'ndcg@4', 'ndcg@5', 'ndcg@10',
               'precision@1', 'precision@2', 'precision@3', 'precision@4', 'precision@5',]
    from ranx import Qrels, Run, evaluate

    results_bm25 = evaluate(Qrels(qrels_test), Run(bm25_rundict), metrics+['recall@10', 'recall@20', 'recall@40', 'recall@50', 'recall@60', 'recall@80', 'recall@100', 'recall@1000'], make_comparable=True)
    print('Bert Rerank :\n',results_bm25)
    print('\n---------------------------------------------\n')
    if args.reject_number != 0:
        results = evaluate(Qrels(qrels_test), Run(rundict), metrics, make_comparable=True)
        print('Reject :\n',results)
        print('\n---------------------------------------------\n')
    results2 = evaluate(Qrels(qrels_test), Run(rundict_rerank), metrics, make_comparable=True)
    print('LLM Rerank :\n',results2)

    # Qrels bin :
    results_bm25_bin = evaluate(Qrels(qrels_test_bin), Run(bm25_rundict), metrics+['recall@10', 'recall@20', 'recall@40', 'recall@50', 'recall@60', 'recall@80', 'recall@100', 'recall@1000'], make_comparable=True)
    print('Bert Rerank BIN :\n',results_bm25_bin)
    print('\n---------------------------------------------\n')
    if args.reject_number != 0:
        results_bin = evaluate(Qrels(qrels_test_bin), Run(rundict), metrics, make_comparable=True)
        print('Reject BIN :\n',results_bin)
        print('\n---------------------------------------------\n')
    results2_bin = evaluate(Qrels(qrels_test_bin), Run(rundict_rerank), metrics, make_comparable=True)
    print('LLM Rerank BIN :\n',results2_bin)

    gen_res_avg_time = gen_res_time/gen_res_count if gen_res_count != 0 else 0
    gen_rerank_avg_time = gen_rerank_time/gen_rerank_count if gen_rerank_count != 0 else 0

    print(f"Total generation time for responses: {gen_res_time:.2f}s for {gen_res_count} queries")
    print(f"Average generation time for responses: {gen_res_avg_time:.2f}s")
    print(f"Total generation time for rerank: {gen_rerank_time:.2f}s for {gen_rerank_count} queries")
    print(f"Average generation time for rerank: {gen_rerank_avg_time:.2f}s")

    with open(f'emissions/emissions_rerank_{args.dataset}_{args.model_name}_rerank_window_topk{args.bm25_topk}_cache.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        tracker_rerank = next(reader)
        tracker_rerank = {key: value for key, value in tracker_rerank.items()}
        
    # Save results in a json file
    results = {
        'bm25': results_bm25,
        'reject': results if args.reject_number != 0 else None,
        'rerank': results2,
        'bm25_bin': results_bm25_bin,
        'reject_bin': results_bin if args.reject_number != 0 else None,
        'rerank_bin': results2_bin,
        'gen_res_time': gen_res_time,
        'gen_res_count': gen_res_count,
        'gen_res_avg_time': gen_res_avg_time,
        'gen_rerank_time': gen_rerank_time,
        'gen_rerank_count': gen_rerank_count,
        'gen_rerank_avg_time': gen_rerank_avg_time,
        'tracker_rerank': {key: float(value)/gen_rerank_count for key, value in tracker_rerank.items() if key in ['duration', 'emissions', 'energy_consumed']},
    }
    with open(f'paper_res/results_{args.dataset}_{args.model_name}_rerank_window_topk{args.bm25_topk}_cache.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)