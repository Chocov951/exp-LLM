import json
import gzip
import argparse
from ranx import Qrels, Run, evaluate
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--reject_number', type=str, help='Number of queries to reject for the reject model')
parser.add_argument('--bm25_topk', type=int, default=100, help='Top k passages to retrieve with BM25')
args = parser.parse_args()

corpus_test, queries_test, qrels_test, qrels_test_bin = {}, {}, {}, {}
year = 20
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

with open(f'rundicts/rundict_trec{year}_bm25.json', 'r', encoding='utf-8') as f:
    bm25_rundict = json.load(f)
with open(f'rundicts/queries_test_trec{year}_bm25.json', 'r', encoding='utf-8') as f:
    queries_test = json.load(f)
with open(f'rundicts/rundict_trec{year}_qwen72_rerank_window_topk{args.bm25_topk}.json', 'r', encoding='utf-8') as f:
    rundict_rerank = json.load(f)
# with open(f'rundicts/rundict_trec{year}_qwen72_reject{args.reject_number}_topk{args.bm25_topk}.json', 'r', encoding='utf-8') as f:
    rundict = rundict_rerank

gen_res_count = 54
gen_rerank_count = 54
gen_res_time = 0.0
gen_rerank_time = 0.0

# Open csv file with tracked emissions data
# with open(f'emissions/emissions_reject_trec{year}_qwen72_reject{args.reject_number}_topk{args.bm25_topk}.csv', 'r', encoding='utf-8') as f:
#     reader = csv.DictReader(f, delimiter=',')
#     tracker_reject = next(reader)
#     tracker_reject = {key: value for key, value in tracker_reject.items()}
with open(f'emissions/emissions_rerank_trec{year}_qwen72_rerank_window_topk{args.bm25_topk}.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    tracker_rerank = next(reader)
    tracker_rerank = {key: value for key, value in tracker_rerank.items()}

 # Print results:
metrics = ['recall@1', 'recall@2', 'recall@3', 'recall@4', 'recall@5',
            'ndcg@1', 'ndcg@2', 'ndcg@3', 'ndcg@4', 'ndcg@5', 'ndcg@10',
            'precision@1', 'precision@2', 'precision@3', 'precision@4', 'precision@5',]

results_bm25 = evaluate(Qrels(qrels_test), Run(bm25_rundict), metrics+['recall@10', 'recall@20', 'recall@40', 'recall@50', 'recall@60', 'recall@80', 'recall@100', 'recall@1000'], make_comparable=True)
print('Bert Rerank :\n',results_bm25)
print('\n---------------------------------------------\n')
results = evaluate(Qrels(qrels_test), Run(rundict), metrics, make_comparable=True)
print('Reject :\n',results)
print('\n---------------------------------------------\n')
results2 = evaluate(Qrels(qrels_test), Run(rundict_rerank), metrics, make_comparable=True)
print('LLM Rerank :\n',results2)

# Qrels bin :
results_bm25_bin = evaluate(Qrels(qrels_test_bin), Run(bm25_rundict), metrics+['recall@10', 'recall@20', 'recall@40', 'recall@50', 'recall@60', 'recall@80', 'recall@100', 'recall@1000'], make_comparable=True)
print('Bert Rerank BIN :\n',results_bm25_bin)
print('\n---------------------------------------------\n')
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
    # 'tracker_reject': {key: float(value)/gen_res_count for key, value in tracker_reject.items() if key in ['duration', 'emissions', 'energy_consumed']},
    'tracker_rerank': {key: float(value)/gen_rerank_count for key, value in tracker_rerank.items() if key in ['duration', 'emissions', 'energy_consumed']},
}
with open(f'paper_res/results_trec{year}_qwen72_rerank_window_topk{args.bm25_topk}_carbon.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)