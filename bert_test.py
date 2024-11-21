import os
import torch
from transformers import AutoTokenizer, AutoModel
from beir.datasets.data_loader import GenericDataLoader
from ranx import Qrels, Run, evaluate
from tqdm import tqdm
import string
from scipy.spatial.distance import cosine
import gc
from rank_bm25 import BM25Okapi
import numpy as np

def preprocess(txt):
    return set(txt.translate(str.maketrans('', '', string.punctuation)).lower().split())

def get_bert_embeddings(texts, model, tokenizer, device='cuda', batch_size=32):
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
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

def evaluate_model(model_name, corpus, queries, qrels_test):
    # Initialiser le modèle et le tokenizer
    jz_model_name = "models--" + model_name.replace("/","--")
    jz_path = os.path.join("/linkhome/rech/genlir01/uep39vh/.cache/huggingface/hub/", jz_model_name, "snapshots/")
    jz_path = os.path.join(jz_path, os.listdir(jz_path)[0])
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(jz_path)
        model = AutoModel.from_pretrained(jz_path).to('cuda')
        model.eval()

        # Préparer le rundict
        rundict = {}
        
        # Obtenir les embeddings pour le corpus
        corpus_texts = [corpus[did]['text'] for did in corpus]
        corpus_embeddings = get_bert_embeddings(corpus_texts, model, tokenizer, batch_size=16)
        
        # Pour chaque requête
        for qid, query in tqdm(queries.items(), desc=f"Évaluation de {model_name}"):
            query_embedding = get_bert_embeddings([query], model, tokenizer)[0]
            
            similarities = []
            for i, doc_embedding in enumerate(corpus_embeddings):
                sim = 1 - cosine(query_embedding, doc_embedding)
                similarities.append((list(corpus.keys())[i], sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            rundict[str(qid)] = {str(did): score for did, score in similarities}

        # Évaluer avec ranx
        qrels = Qrels(qrels_test)
        run = Run(rundict)
        results = evaluate(qrels, run, ['recall@1', 'recall@2', 'recall@3', 'recall@4', 'recall@5',
                                      'ndcg@1', 'ndcg@2', 'ndcg@3', 'ndcg@4', 'ndcg@5',
                                      'precision@1'],
                                      make_comparable=True)
        
        # Nettoyage explicite
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    except Exception as e:
        print(f"Erreur lors de l'évaluation de {model_name}: {str(e)}")
        # Nettoyage en cas d'erreur
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()
        raise e

def evaluate_bm25(corpus, queries, qrels_test):
    # Prétraiter et tokenizer le corpus
    corpus_texts = [corpus[did]['text'] for did in corpus]
    tokenized_corpus = [preprocess(doc) for doc in corpus_texts]
    
    # Initialiser BM25
    bm25 = BM25Okapi([list(doc) for doc in tokenized_corpus])
    
    # Préparer le rundict
    rundict = {}
    
    # Pour chaque requête
    for qid, query in tqdm(queries.items(), desc="Évaluation de BM25"):
        tokenized_query = preprocess(query)
        
        # Calculer les scores BM25
        scores = bm25.get_scores(list(tokenized_query))
        
        # Créer les paires document-score
        similarities = list(zip(corpus.keys(), scores))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        rundict[str(qid)] = {str(did): score for did, score in similarities}
    
    # Évaluer avec ranx
    qrels = Qrels(qrels_test)
    run = Run(rundict)
    results = evaluate(qrels, run, ['recall@1', 'recall@2', 'recall@3', 'recall@4', 'recall@5',
                                  'ndcg@1', 'ndcg@2', 'ndcg@3', 'ndcg@4', 'ndcg@5',
                                  'precision@1'],
                                  make_comparable=True)
    
    return results

def main():
    # Liste des modèles à tester
    models = [
        "BAAI/bge-m3",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "Alibaba-NLP/gte-multilingual-base",
        "dangvantuan/sentence-camembert-base",
        "tomaarsen/xlm-roberta-base-multilingual-en-ar-fr-de-es-tr-it"
    ]

    # Charger le dataset
    dataset = "hal"
    corpus, queries, qrels_test = GenericDataLoader(f"datasets/{dataset}").load(split="test")

    # Évaluer BM25 d'abord
    print("\nÉvaluation de BM25")
    try:
        bm25_results = evaluate_bm25(corpus, queries, qrels_test)
        results_by_model = {"BM25": bm25_results}
        print("Résultats pour BM25:")
        print(bm25_results)
    except Exception as e:
        print(f"Erreur lors de l'évaluation de BM25: {str(e)}")
        results_by_model = {}

    # Évaluer chaque modèle BERT
    for model_name in models:
        print(f"\nÉvaluation du modèle: {model_name}")
        try:
            results = evaluate_model(model_name, corpus, queries, qrels_test)
            results_by_model[model_name] = results
            print(f"Résultats pour {model_name}:")
            print(results)
        except Exception as e:
            print(f"Erreur lors de l'évaluation de {model_name}: {str(e)}")

    # Afficher un récapitulatif comparatif
    print("\nRécapitulatif des performances:")
    for model_name, results in results_by_model.items():
        print(f"\n{model_name}:")
        print(f"NDCG@1: {results['ndcg@1']:.4f}")
        print(f"Recall@1: {results['recall@1']:.4f}")
        print(f"Precision@1: {results['precision@1']:.4f}")

if __name__ == "__main__":
    main()