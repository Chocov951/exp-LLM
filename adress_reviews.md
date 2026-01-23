# Fastest Complementary Experiments Plan (Reviewer-Oriented)

## 1. First-stage retrieval alternatives (minimal but convincing)

**Keep existing**
- BM25 (baseline)

**Add ONE strong dense retriever (no LLM involved)**
- Use **Contriever** (or DPR / BGE if already available)
  - Paper: *Contriever* (Izacard et al.)
  - Setup:
    - Retrieve top-100 / top-200
    - Feed directly into final reranker (no filter)
- Purpose:
  - Answer: *“Why not just use a better retriever instead of LLM filtering?”*

**What to compare**
- BM25 → final rerank  
- Dense retriever → final rerank  
- BM25 → LLM filter → final rerank (FIRE-LLM)

**Instead**
- Write in limitations : We dont want to redo all exp with contriever but we could to see the impact of top100 in final ndcg at 10

---

## 2. Replace LLM pre-filter with a small cross-encoder (key reviewer request)

**Add ONE small cross-encoder as “filter”**
- Model: **RankT5 / MiniLM / DistilBERT cross-encoder**
  - Canonical reference: 
    - *Nogueira & Cho, Passage Re-Ranking with BERT* 
    - *RankT5 : Fine-Tuning T5 for Text Ranking with Ranking Losses*
- Setup:
  - BM25 top-200
  - Cross-encoder scores → keep top-20 / top-30
  - Same final reranker as FIRE-LLM

**Purpose**
- Directly tests:
  - “LLM pre-filter vs small cross-encoder intermediate reranking”
  - Reviewer 1 + Reviewer 3 core criticism

---

## 3. LLM reranking alternatives (fast to plug in)

### 3.1 RankGPT (sorting-style baseline)
- Paper: *RankGPT* (Sun et al.)
- Already in the paper → **extend it**
- Use for:
  - Full sorting
  - Depth experiments (top-10 / 20 / 50 / 100)
- Purpose:
  - Efficiency comparison against FIRE-LLM
  - Reviewer 2a

### 3.2 FIRST (fast LLM reranking)
- Paper: **FIRST** (arXiv:2406.15657)
- Key idea:
  - Use **first-token logits only**
  - ~50% faster inference
- Setup:
  - Replace final LLM reranker with FIRST
  - Same inputs, same candidate sets
- Purpose:
  - Show FIRE-LLM compatibility with faster LLM reranking
  - Demonstrate efficiency frontier

### 3.3 RankZephyr (open-source strong baseline)
- Paper: **RankZephyr** (arXiv:2312.02724)
- Model:
  - 7B open-source listwise reranker
- Setup:
  - Drop-in replacement for final reranker
- Purpose:
  - Broader comparison
  - Avoid “only proprietary LLM” criticism

---

## 4. Minimal depth experiment (Reviewer 2a)

**Question**
> What if the user wants more than top-10?

**Implement**
- Target depths: {10, 20, 50, 100}
- For each method:
  - RankGPT: normal iterative sorting
  - FIRE-LLM: iterative filtering + reranking on remaining candidates

**Report**
- NDCG@10 / NDCG@20
- Recall@50 / Recall@100
- Runtime + number of LLM calls

**Outcome**
- Show FIRE-LLM efficiency gains persist as depth increases

---

## 5. Statistical significance (Reviewer 2b)

**Do the minimum**
- Paired significance test:
  - FIRE-LLM vs RankGPT
  - NDCG@10 per query
- Datasets:
  - TREC-DL 2019
  - TREC-DL 2020

**Report**
- p-values
- Mark significant / non-significant differences in tables

---

## 6. Critical ablation study on filtering (Reviewer 2c)

### Ablation A — Is filtering needed?
- BM25 → final rerank (no filter)

### Ablation B — Does filter *ordering* matter?
Use same filtered IDs:
- B1: LLM filter score order (current)
- B2: BM25 order
- B3: Random order

### Ablation C — Does filtering help downstream?
For each variant:
- Filter precision / recall vs qrels
- Final NDCG@10
- Runtime

---

## 7. Pipeline grid (to satisfy Review 3 cleanly)

Evaluate only these **6 pipelines**:
1. BM25 → final rerank  
2. Dense retriever → final rerank  
3. BM25 → small cross-encoder → final rerank  
4. BM25 → LLM filter → final rerank (FIRE-LLM)  
5. BM25 → LLM filter → FIRST  
6. BM25 → LLM filter → RankGPT  

**This setup:**
- Covers all reviewer concerns  
- Keeps implementation scope reasonable  
- Strongly justifies the FIRE-LLM design choices



## PLUS : 

- **Regarder si ndcg20 et ndcg50**