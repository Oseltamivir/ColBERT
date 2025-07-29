import time
import torch
import random
import math
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Set

import os, re
import json
from pathlib import Path
from datetime import datetime

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
import matplotlib.pyplot as plt

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints

import torch.nn.functional as F

def lipo_lambda_loss(scores: torch.Tensor,
                     pos_idx: torch.LongTensor,
                     tau: float = 1.0) -> torch.Tensor:
    probs = F.softmax(scores / tau, dim=1)
    B, L = scores.size()
    pos_prob = probs[torch.arange(B), pos_idx]
    if torch.any(pos_prob <= 0):
        print("Warning: pos_prob has zero or negative values:", pos_prob)
    log_prob_pos = torch.log(pos_prob + 1e-12)     # [B]
    inv_rank_pos = pos_prob.detach()                # treat 1/prob as rank proxy

    # compute λ weights for every doc: λ_j = max(0, inv_rank_pos - inv_rank_j)
    inv_rank_all = probs.detach()                   # proxy ranks
    lambda_weights = (inv_rank_pos[:, None] - inv_rank_all).clamp(min=0)

    loss = - (lambda_weights.sum(dim=1) * log_prob_pos).mean()
    return loss




# --------------------------- EVAL HELPERS ---------------------------------

# Static eval config
EVAL_K_LIST = [40, 50, 60, 70, 80, 90, 100]
EVAL_DATA_DIR = Path("/home/bshan/colbert/data/")  # per your answer
EVAL_DOC_BSIZE = 64  # doc encoding batch size for reranking

_WORD_RE = re.compile(r"\w+")

# Reuse tokenizers to avoid re-instantiation overhead during eval
_QTOK = None
_DTOK = None

def _get_tokenizers(config):
    global _QTOK, _DTOK
    if _QTOK is None or _DTOK is None:
        _QTOK = QueryTokenizer(config)
        _DTOK = DocTokenizer(config)
    return _QTOK, _DTOK


def _canon_text(text: str) -> str:
    return " ".join(text.split()).lower()

def _preprocess(text: str, *, min_len: int = 2, strip_nums: bool = True) -> Set[str]:
    text = text.lower()
    text = text.replace("’", " ").replace("‘", " ")
    text = text.replace("–", " ").replace("—", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = []
    for tok in _WORD_RE.findall(text):
        if strip_nums and tok.isdigit():
            continue
        if len(tok) < min_len:
            continue
        tokens.append(tok)
    return set(tokens)

def _coverage_ratio(gt: str, chunk: str, *, min_len: int = 2) -> float:
    gt_set = _preprocess(gt, min_len=min_len)
    ch_set = _preprocess(chunk, min_len=min_len)
    if not gt_set or not ch_set:
        return 0.0
    inter = len(gt_set & ch_set)
    return inter / min(len(gt_set), len(ch_set))

def _rel_vector(docs_json: List[Dict], gt_texts: List[str], thr: float = 0.95, min_len: int = 2) -> List[int]:
    return [
        int(any(_coverage_ratio(gt, d["doc"]["page_content"], min_len=min_len) >= thr for gt in gt_texts))
        for d in docs_json
    ]

def _ap_at_k(rels: List[int], k: int) -> float:
    hits, sum_prec = 0, 0.0
    cutoff = min(k, len(rels))
    for i in range(1, cutoff + 1):
        if rels[i-1] == 1:
            hits += 1
            sum_prec += hits / i
    return 0.0 if hits == 0 else (sum_prec / hits)


def _dcg(rels: List[int]) -> float:
    return sum((rel / math.log2(i+2)) for i, rel in enumerate(rels))

def _ndcg(rels: List[int]) -> float:
    ideal = sorted(rels, reverse=True)
    idcg  = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0

# ---- Minimal ColBERT-native reranker (no external deps) ------------------

def _tokenize_queries_docs(config, queries: List[str], docs: List[str], device):
    qtok = QueryTokenizer(config)
    dtok = DocTokenizer(config)
    # Tokenize query batch (we do 1 query at a time in eval, but keep general)
    q_inputs = qtok.tensorize(queries)
    d_inputs = dtok.tensorize(docs)
    # Move to device lazily in model.query/doc
    return q_inputs, d_inputs

@torch.inference_mode()
def _rerank_single_query(colbert_module, config, query: str, texts: List[str], device, doc_bsize: int) -> List[Tuple[int, float]]:
    """
    Returns list of (orig_idx, score) sorted by descending score.
    Fast path: encodes *all* docs in a single call if doc_bsize in {0, None} or >= len(texts).
    Uses Q.size(0)==1 path in colbert_score so we don't need to duplicate by nway.
    """
    qtok, dtok = _get_tokenizers(config)

    # Encode query once
    q_ids, q_mask = qtok.tensorize([query])
    Q = colbert_module.query(q_ids.to(device), q_mask.to(device))  # [1, Lq, dim]

    N = len(texts)
    scores_all: List[float] = []
    idxs_all: List[int] = []

    # Decide batch size: "all at once" if 0/None or large enough
    if (not doc_bsize) or (doc_bsize >= N):
        d_ids, d_mask = dtok.tensorize(texts)  # [N, Ld], [N, Ld]
        D, D_mask = colbert_module.doc(d_ids.to(device), d_mask.to(device), keep_dims='return_mask')
        batch_scores = colbert_module.score(Q, D, D_mask)  # [N]
        scores_all = batch_scores.detach().float().cpu().tolist()
        idxs_all = list(range(N))
    else:
        # Fallback: chunked pass
        for start in range(0, N, doc_bsize):
            end = min(start + doc_bsize, N)
            batch_texts = texts[start:end]
            d_ids, d_mask = dtok.tensorize(batch_texts)
            D, D_mask = colbert_module.doc(d_ids.to(device), d_mask.to(device), keep_dims='return_mask')
            batch_scores = colbert_module.score(Q, D, D_mask)  # [B]
            scores_all.extend(batch_scores.detach().float().cpu().tolist())
            idxs_all.extend(range(start, end))

    # Sort by descending score
    order = sorted(range(len(scores_all)), key=lambda i: scores_all[i], reverse=True)
    return [(idxs_all[i], float(scores_all[i])) for i in order]



@torch.inference_mode()
def _evaluate_benchmark(colbert_ddp, config, device, step: int, metrics_file: Path) -> Dict[str, float]:
    model = colbert_ddp.module if hasattr(colbert_ddp, "module") else colbert_ddp
    was_training = model.training
    model.eval()

    rows_all: List[Dict] = []
    ndcgs_allk: List[float] = []
    ndcgs_per_k: Dict[int, List[float]] = {K: [] for K in EVAL_K_LIST}

    # NEW: collect AP@15 specifically for K==40
    map_k40_15_list: List[float] = []

    for K in EVAL_K_LIST:
        in_path = EVAL_DATA_DIR / f"candidates_k{K}.jsonl"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing benchmark file: {in_path}")

        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                case = row["case"]
                query = row["query"]
                gts   = row["ground_truths"]
                docs_j = row["docs"]
                assert len(docs_j) == K

                texts_canon = [_canon_text(d["doc"]["page_content"]) for d in docs_j]
                assert len(texts_canon) == len(set(texts_canon)), f"{case} K{K}: duplicate texts!"

                texts = [d["doc"]["page_content"] for d in docs_j]
                reranked = _rerank_single_query(model, config, query, texts, device, EVAL_DOC_BSIZE)

                reranked_texts = [texts[idx] for idx, _ in reranked]
                rels = [
                    int(any(_coverage_ratio(gt, txt, min_len=2) >= 0.95 for gt in gts))
                    for txt in reranked_texts
                ]

                nd = _ndcg(rels)

                ndcgs_allk.append(nd)
                ndcgs_per_k[K].append(nd)
                rows_all.append({"case": case, "K": K, "nDCG": nd})

                # NEW: compute AP@15 when K==40
                if K == 40:
                    map15 = _ap_at_k(rels, k=15)
                    map_k40_15_list.append(map15)

    mean_ndcg_allk = float(sum(ndcgs_allk) / max(1, len(ndcgs_allk)))
    mean_ndcgs_per_k = {
        f"mean_ndcg_k{K}": float(sum(vals) / max(1, len(vals)))
        for K, vals in ndcgs_per_k.items()
    }

    # NEW: aggregate MAP@15 for K=40
    mean_map_k40_15 = float(sum(map_k40_15_list) / max(1, len(map_k40_15_list)))

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "mean_ndcg_allk": mean_ndcg_allk,
        **mean_ndcgs_per_k,
        # NEW in metrics log
        "mean_map_k40_15": mean_map_k40_15,
        "counts": {
            "allk": len(ndcgs_allk),
            **{f"k{K}": len(vals) for K, vals in ndcgs_per_k.items()},
            "k40_for_map15": len(map_k40_15_list),
        }
    }
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(record) + "\n")

    if was_training:
        model.train()

    # NEW: return mean_map_k40_15 as part of results
    return {"mean_ndcg_allk": mean_ndcg_allk,
            **mean_ndcgs_per_k,
            "mean_map_k40_15": mean_map_k40_15}


def _overwrite_dir(path: Path):
    if path.exists():
        # Remove previous best to truly "overwrite"
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)




def train(config: ColBERTConfig, triples, queries=None, collection=None):
    # --------------------- Setup & Seeding ---------------------
    config.checkpoint = config.checkpoint or 'bert-base-uncased'
    if config.rank < 1:
        config.help()

    # Label smoothing factor (add this to your config; e.g. 0.1)
    eps = getattr(config, 'label_smoothing', 0.2)

    # ------------------ Plot teacher‐score distribution ------------------
    # Flatten all teacher scores from your triples
    orig_scores = [score for triple in triples for (_pid, score) in triple[1:]]
    # Compute what they would look like after smoothing
    smoothed_scores = []
    for triple in triples:
        nway = len(triple) - 1
        for (_pid, score) in triple[1:]:
            smoothed_scores.append((1 - eps) * score + eps / nway)

    plt.figure()
    plt.hist(orig_scores,   bins= fifty if False else 50, alpha=0.5, label='Original')
    plt.hist(smoothed_scores, bins=50, alpha=0.5, label='Smoothed')
    plt.legend()
    plt.title('Teacher score distribution: before vs. after label smoothing')
    plt.savefig("teacher_score_distribution.png")

    # ------------------ Apply smoothing to your triples in place ------------------
    for t_idx, triple in enumerate(triples):
        nway = len(triple) - 1
        for j in range(1, len(triple)):
            pid, score = triple[j]
            triple[j][1] = (1 - eps) * score + eps / nway

    # --------------------- Standard seeding / batch size split ---------------------
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks
    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    # --------------------- Reader (RerankBatcher vs LazyBatcher) ---------------------
    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection,
                                   (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection,
                                 (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError("No collection provided to train()")

    # --------------------- Model & Optimizer ---------------------
    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()
    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert, device_ids=[config.rank], output_device=config.rank, find_unused_parameters=True
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    # --------------------- Eval trackers & paths ---------------------
    best_allk = 0.58

    # NEW: track MAP@15 for K=40 instead of mean_ndcg_k40
    best_k40_15 = 0.505

    eval_logs_dir = Path(config.root) / "eval_logs"
    metrics_file   = eval_logs_dir / "metrics.jsonl"
    best_allk_dir  = Path(config.root) / "best_allk"

    # NEW: directory name reflects the metric
    best_k40_15_dir = Path(config.root) / "best_k40_15_map"


    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = getattr(config, 'warmup_bert', None)
    if warmup_bert is not None:
        # freeze BERT until after warmup
        for p in colbert.bert.parameters():
            p.requires_grad = False

    amp    = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)  # positive index=0

    start_time     = time.time()
    train_loss     = None
    train_loss_mu  = 0.999
    start_batch_idx = 0

    # --------------------- Main training loop ---------------------
    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if warmup_bert is not None and warmup_bert <= batch_idx:
            # unfreeze
            for p in colbert.bert.parameters():
                p.requires_grad = True
            warmup_bert = None

        this_batch_loss = 0.0
        nway_this_seen = None

        for batch in BatchSteps:
            # batch shape: (queries, passages, target_scores, nway_this) or legacy
            if isinstance(batch, tuple) and len(batch) == 4:
                queries_t, passages_t, target_scores, nway_this = batch
                encoding = [queries_t, passages_t]
            elif isinstance(batch, tuple) and len(batch) == 3:
                enc, target_scores, nway_this = batch
                encoding = [enc.to(DEVICE)]
            else:
                # fallback legacy
                queries_t, passages_t, target_scores = batch
                nway_this = getattr(config, 'nway', None)
                encoding = [queries_t, passages_t]

            nway_this_seen = nway_this

            with amp.context():
                # forward
                scores = colbert(*encoding, nway_override=nway_this)  # [B*nway]
                if getattr(config, 'use_ib_negatives', False):
                    scores, ib_loss = scores

                scores = scores.view(-1, nway_this)  # [B, nway]

                if len(target_scores) and not getattr(config, 'ignore_scores', False):
                    # ---- Distillation + label smoothing on teacher probs ----
                    B, _ = scores.size()
                    tau = float(getattr(config, "kd_temperature", 2.0))
                    lam = float(getattr(config, "kd_lambda", 0.9))

                    # student logits -> float32
                    s32 = scores.float()
                    # teacher scores -> tensor -> float32
                    t32 = torch.as_tensor(target_scores, device=scores.device, dtype=torch.float32).view(B, nway_this)

                    # if raw probs, z‑score them
                    if t32.max() <= 1.0 and t32.min() >= 0.0:
                        t32 = (t32 - t32.mean(1, keepdim=True)) / (t32.std(1, keepdim=True) + 1e-6)

                    # shift to avoid overflow
                    t32 = t32 - t32.max(1, keepdim=True).values
                    s32 = s32 - s32.max(1, keepdim=True).values

                    pT = nn.functional.softmax(t32 / tau, dim=-1)
                    # apply label smoothing to teacher distillation targets
                    pT = (1 - eps) * pT + eps / nway_this

                    log_pS = nn.functional.log_softmax(s32 / tau, dim=-1)
                    loss_kd = nn.functional.kl_div(log_pS, pT, reduction="batchmean") * (tau * tau)

                    # CE on raw logits (anchor pos idx=0)
                    loss_ce = nn.CrossEntropyLoss(label_smoothing=eps)(scores, labels[:B])

                    # optional margin
                    m = 0.2
                    pos    = scores[:, 0]
                    neg_max= scores[:, 1:].max(1).values
                    loss_margin = nn.functional.relu(m - (pos - neg_max)).mean() * 0.1

                    loss = lam * loss_kd + (1.0 - lam) * loss_ce + loss_margin
                    if getattr(config, 'use_ib_negatives', False):
                        if config.rank < 1:
                            print('\tKD+CE+IB losses:', loss_kd.item(), loss_ce.item(), ib_loss.item())
                        loss = loss + ib_loss

                else:
                    # ---- Pure CE with label smoothing ----
                    ce = nn.CrossEntropyLoss(label_smoothing=eps)
                    ce_loss = ce(scores, labels[: scores.size(0)])
                    norm = max(1.0, math.log(float(nway_this)))
                    loss = ce_loss / norm

                # accumulation step
                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)
            this_batch_loss += loss.item()

        # optimizer step
        train_loss = this_batch_loss if train_loss is None else (train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss)
        amp.step(colbert, optimizer, scheduler)

        # -------------- periodic eval & checkpointing on rank 0 --------------
        global_step = batch_idx + 1
        if (config.rank in [0, -1] and (global_step == 1 or global_step % 200 == 0)):
            results = _evaluate_benchmark(colbert, config, DEVICE, step=global_step, metrics_file=metrics_file)
            mean_allk = results["mean_ndcg_allk"]
            mean_k40_15 = results["mean_map_k40_15"]

            if mean_allk > best_allk:
                best_allk = mean_allk
                manage_checkpoints(config, colbert, optimizer, global_step, savepath=str(best_allk_dir))

            if mean_k40_15 > best_k40_15:
                best_k40_15 = mean_k40_15
                manage_checkpoints(config, colbert, optimizer, global_step, savepath=str(best_k40_15_dir))

        if config.rank < 1:
            if nway_this_seen is not None:
                print_message(f"[step {batch_idx}] EMA loss={train_loss:.6f} (nway={nway_this_seen})")
            else:
                print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    # --------------------- done ---------------------
    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)
        return ckpt_path

def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
