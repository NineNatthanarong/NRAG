# Dense embeddings vs NRAG — BEIR (scifact · nfcorpus · fiqa)

Cost-fair comparison on three BEIR datasets. **NRAG = pure lexical** (`preset="fast"`, BM25 + trigrams + title, no LLM): **$0 per query**.
Dense = OpenRouter `/embeddings` + cosine over L2-normalized vectors, the standard dense setup (docs truncated to 6000 chars). Chunk/doc scores are max-pooled to document ids to match BEIR's document-level judgments.

Run date: 2026-08-15 · methodology: `python benchmarks/dense_bench.py` (cache in `.bench_cache/`).

## scifact

| System | Emb. | Dims | Query cost | nDCG@10 | R@10 | R@100 | MRR | Spend |
|---|---|---|---|---|---|---|---|---|
| NRAG (pure lexical) | ✗ | — | $0/query | 0.7100 | 0.8271 | 0.9363 | 0.6838 | — |
| text-embedding-3-small (dense) | ✔ | 1536 | embed/query | 0.7163 | 0.8536 | 0.9700 | 0.6840 | $0.033 |
| text-embedding-3-large (dense) | ✔ | 3072 | embed/query | 0.7786 | 0.8997 | 0.9767 | 0.7512 | $0.216 |
| bge-m3 (dense) | ✔ | 1024 | embed/query | 0.6436 | 0.7834 | 0.9037 | 0.6130 | $0.019 |
| qwen3-embedding-4b (dense) | ✔ | 2560 | embed/query | 0.7304 | 0.8717 | 0.9733 | 0.6960 | $0.034 |
| normal FTS (BM25 word-only) | ✗ | — | $0/query | 0.6860 | 0.8193 | 0.9216 | 0.6532 | — |
| BM25 (Anserini, published) | ✗ | — | $0/query | 0.6650 | — | — | — | — |
| NRAG + doc2query x2 (prior run) | ✗ | — | $0/query | 0.7291 | — | — | 0.7042 | — |

## nfcorpus

| System | Emb. | Dims | Query cost | nDCG@10 | R@10 | R@100 | MRR | Spend |
|---|---|---|---|---|---|---|---|---|
| NRAG (pure lexical) | ✗ | — | $0/query | 0.3366 | 0.1629 | 0.2599 | 0.5429 | — |
| text-embedding-3-small (dense) | ✔ | 1536 | embed/query | 0.3837 | 0.1873 | 0.3646 | 0.5957 | $0.025 |
| text-embedding-3-large (dense) | ✔ | 3072 | embed/query | 0.4239 | 0.2109 | 0.3999 | 0.6363 | $0.162 |
| bge-m3 (dense) | ✔ | 1024 | embed/query | 0.3153 | 0.1476 | 0.2826 | 0.5307 | $0.014 |
| qwen3-embedding-4b (dense) | ✔ | 2560 | embed/query | 0.3579 | 0.1834 | 0.3471 | 0.5571 | $0.026 |
| normal FTS (BM25 word-only) | ✗ | — | $0/query | 0.3229 | 0.1536 | 0.2479 | 0.5309 | — |
| BM25 (Anserini, published) | ✗ | — | $0/query | 0.3250 | — | — | — | — |

## fiqa

| System | Emb. | Dims | Query cost | nDCG@10 | R@10 | R@100 | MRR | Spend |
|---|---|---|---|---|---|---|---|---|
| NRAG (pure lexical) | ✗ | — | $0/query | 0.2511 | 0.3160 | 0.5486 | 0.3195 | — |
| text-embedding-3-small (dense) | ✔ | 1536 | embed/query | 0.4487 | 0.5222 | 0.7783 | 0.5299 | $0.195 |
| bge-m3 (dense) | ✔ | 1024 | embed/query | 0.4130 | 0.4730 | 0.7180 | 0.5113 | $0.108 |
| qwen3-embedding-4b (dense) | ✔ | 2560 | embed/query | 0.5047 | 0.5952 | 0.8423 | 0.5804 | $0.199 |
| BM25 (Anserini, published) | ✗ | — | $0/query | 0.2360 | — | — | — | — |
| normal FTS (BM25 word-only) | ✗ | — | $0/query | 0.2478 | 0.3089 | 0.5487 | 0.3200 | — |

*Pending: `text-embedding-3-large` on fiqa (not run — API key budget). Rerun with `OPENROUTER_API_KEY=... python benchmarks/dense_bench.py --datasets fiqa --skip-nrag` to fill in.*

## Honest headline

- **scifact:** NRAG 0.7100 vs best dense (text-embedding-3-large (dense)) 0.7786 nDCG@10 — dense leads.
- **nfcorpus:** NRAG 0.3366 vs best dense (text-embedding-3-large (dense)) 0.4239 nDCG@10 — dense leads.
- **fiqa:** NRAG 0.2511 vs best dense (qwen3-embedding-4b (dense)) 0.5047 nDCG@10 — dense leads.

## Raw records

```json
{"dataset": "fiqa", "dims": 2560, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.5803881524418912, "ndcg@10": 0.5047254688960281, "recall@10": 0.5951710455182675, "recall@100": 0.8423278067953998}, "spend": 0.19906765999999998, "system": "qwen3-embedding-4b (dense)"}
{"dataset": "fiqa", "dims": 1536, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.529857213815743, "ndcg@10": 0.4486599004344054, "recall@10": 0.5222144068208882, "recall@100": 0.7782785672831972}, "spend": 0.19470438, "system": "text-embedding-3-small (dense)"}
{"dataset": "fiqa", "dims": 1024, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.5112909970804546, "ndcg@10": 0.41297217471523384, "recall@10": 0.47302813356054096, "recall@100": 0.7180071300441674}, "spend": 0.10780184999999994, "system": "bge-m3 (dense)"}
{"dataset": "fiqa", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": 0.3195389468223928, "ndcg@10": 0.2511369574824167, "recall@10": 0.3159716997679962, "recall@100": 0.5485726165124313}, "spend": 0.0, "system": "NRAG (pure lexical)"}
{"dataset": "fiqa", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": 0.31999456585046926, "ndcg@10": 0.24778282300742194, "recall@10": 0.3089431530403753, "recall@100": 0.5486693733221508}, "spend": 0.0, "system": "normal FTS (BM25 word-only)"}
{"dataset": "fiqa", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": null, "ndcg@10": 0.236, "recall@10": null, "recall@100": null}, "spend": 0.0, "system": "BM25 (Anserini, published)"}
{"dataset": "nfcorpus", "dims": 3072, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.636270625042552, "ndcg@10": 0.4239389411255284, "recall@10": 0.21091235527587032, "recall@100": 0.3998612724203595}, "spend": 0.1617, "system": "text-embedding-3-large (dense)"}
{"dataset": "nfcorpus", "dims": 1536, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.5957390274135678, "ndcg@10": 0.38365155258516703, "recall@10": 0.18733617431629504, "recall@100": 0.36455157056713405}, "spend": 0.0249, "system": "text-embedding-3-small (dense)"}
{"dataset": "nfcorpus", "dims": 2560, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.5571262972609634, "ndcg@10": 0.357902594061004, "recall@10": 0.18340588879214093, "recall@100": 0.34714891919534313}, "spend": 0.0259, "system": "qwen3-embedding-4b (dense)"}
{"dataset": "nfcorpus", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": 0.5428554945237435, "ndcg@10": 0.3366297823734831, "recall@10": 0.16293458963032487, "recall@100": 0.25993305505252534}, "spend": 0.0, "system": "NRAG (pure lexical)"}
{"dataset": "nfcorpus", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": null, "ndcg@10": 0.325, "recall@10": null, "recall@100": null}, "spend": 0.0, "system": "BM25 (Anserini, published)"}
{"dataset": "nfcorpus", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": 0.530920951371953, "ndcg@10": 0.3228802018425997, "recall@10": 0.15359121217453667, "recall@100": 0.24786281784315048}, "spend": 0.0, "system": "normal FTS (BM25 word-only)"}
{"dataset": "nfcorpus", "dims": 1024, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.5306904257850067, "ndcg@10": 0.3152959634865237, "recall@10": 0.14755244045974913, "recall@100": 0.2826362761375877}, "spend": 0.0142, "system": "bge-m3 (dense)"}
{"dataset": "scifact", "dims": 3072, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.751208927905124, "ndcg@10": 0.7786064156081363, "recall@10": 0.8997222222222221, "recall@100": 0.9766666666666667}, "spend": 0.2164, "system": "text-embedding-3-large (dense)"}
{"dataset": "scifact", "dims": 2560, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.6960133440965162, "ndcg@10": 0.7304024223033324, "recall@10": 0.8716666666666667, "recall@100": 0.9733333333333334}, "spend": 0.0343, "system": "qwen3-embedding-4b (dense)"}
{"dataset": "scifact", "dims": "\u2014", "emb": "\u2717", "note": "LLM index-time enrichment, cached; from scifact_results.md", "query_cost": "$0/query", "scores": {"mrr": 0.7042, "ndcg@10": 0.7291, "recall@10": null, "recall@100": null}, "spend": 0.0, "system": "NRAG + doc2query x2 (prior run)"}
{"dataset": "scifact", "dims": 1536, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.6839984563949932, "ndcg@10": 0.716315750657132, "recall@10": 0.8535555555555555, "recall@100": 0.97}, "spend": 0.0333, "system": "text-embedding-3-small (dense)"}
{"dataset": "scifact", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": 0.6837817667687704, "ndcg@10": 0.7099593168236384, "recall@10": 0.8271111111111111, "recall@100": 0.9363333333333332}, "spend": 0.0, "system": "NRAG (pure lexical)"}
{"dataset": "scifact", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": 0.6531898582323372, "ndcg@10": 0.6859662244030473, "recall@10": 0.8193333333333334, "recall@100": 0.9215555555555557}, "spend": 0.0, "system": "normal FTS (BM25 word-only)"}
{"dataset": "scifact", "dims": "\u2014", "emb": "\u2717", "query_cost": "$0/query", "scores": {"mrr": null, "ndcg@10": 0.665, "recall@10": null, "recall@100": null}, "spend": 0.0, "system": "BM25 (Anserini, published)"}
{"dataset": "scifact", "dims": 1024, "emb": "\u2714", "query_cost": "embed/query", "scores": {"mrr": 0.612994127256947, "ndcg@10": 0.6435744069611098, "recall@10": 0.7834444444444444, "recall@100": 0.9036666666666667}, "spend": 0.0195, "system": "bge-m3 (dense)"}
```
