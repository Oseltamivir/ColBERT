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

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints

# --------------------------- Relevance labelling constants -----------------
# Make these MATCH your notebook to ensure identical MAP/nDCG numbers.
GT_MIN_LEN    = 2
COVERAGE_THR  = 0.95
STRIP_NUMS    = True

# --------------------------- LoSS: LIPO-Lambda -----------------------------
def lipo_lambda_loss(scores: torch.Tensor,
                     pos_idx: torch.LongTensor,
                     tau: float = 1.0) -> torch.Tensor:
    """
    NOTE: The comment says '1/prob as rank proxy' but the code uses prob directly.
    Leaving intact per your request; change if you actually meant 1/prob.
    """
    probs = F.softmax(scores / tau, dim=1)  # [B, L]
    B, L = scores.size()
    pos_prob = probs[torch.arange(B), pos_idx]
    if torch.any(pos_prob <= 0):
        print("Warning: pos_prob has zero or negative values:", pos_prob)
    log_prob_pos = torch.log(pos_prob + 1e-12)     # [B]
    inv_rank_pos = pos_prob.detach()               # proxy (as written)

    inv_rank_all = probs.detach()
    lambda_weights = (inv_rank_pos[:, None] - inv_rank_all).clamp(min=0)

    loss = - (lambda_weights.sum(dim=1) * log_prob_pos).mean()
    return loss

# --------------------------- EVAL CONFIG -----------------------------------
EVAL_K_LIST = [40, 50, 60, 70, 80, 90, 100]
EVAL_DATA_DIR = Path("/home/bshan/colbert/data/")
EVAL_DOC_BSIZE = 64  # doc encoding batch size for reranking

_WORD_RE = re.compile(r"\w+")
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

def _coverage_ratio(gt: str, chunk: str, *, min_len: int = 2, strip_nums: bool = True) -> float:
    gt_set = _preprocess(gt, min_len=min_len, strip_nums=strip_nums)
    ch_set = _preprocess(chunk, min_len=min_len, strip_nums=strip_nums)
    if not gt_set or not ch_set:
        return 0.0
    inter = len(gt_set & ch_set)
    return inter / min(len(gt_set), len(ch_set))

# --------------------------- Metric helpers (MATCH notebook) ---------------
def ap_at_k(rels: List[int], k: int) -> float:
    """AP over the first k items; denominator = #relevant in top-k (pool-based)."""
    sub = rels[:k]
    m = sum(sub)
    if m == 0:
        return 0.0
    cum = 0.0
    hit = 0
    for i, r in enumerate(sub, start=1):
        if r:
            hit += 1
            cum += hit / i
    return cum / m

def ndcg_at_k(rels: List[int], k: int) -> float:
    """nDCG over the first k items."""
    sub = rels[:k]
    if not sub:
        return 0.0
    # DCG
    dcg_val = sum((r / math.log2(i+2)) for i, r in enumerate(sub))
    # IDCG on the same prefix length
    ideal = sorted(sub, reverse=True)
    idcg  = sum((r / math.log2(i+2)) for i, r in enumerate(ideal))
    return (dcg_val / idcg) if idcg > 0 else 0.0

# (Retain old "full-list" nDCG@K for your per-K means)
def _dcg(rels: List[int]) -> float:
    return sum((rel / math.log2(i+2)) for i, rel in enumerate(rels))

def _ndcg_fullK(rels: List[int]) -> float:
    ideal = sorted(rels, reverse=True)
    idcg  = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0

# ---------------------- Minimal ColBERT-native reranker --------------------
def _tokenize_queries_docs(config, queries: List[str], docs: List[str], device):
    qtok = QueryTokenizer(config)
    dtok = DocTokenizer(config)
    q_inputs = qtok.tensorize(queries)
    d_inputs = dtok.tensorize(docs)
    return q_inputs, d_inputs

@torch.inference_mode()
def _rerank_single_query(colbert_module, config, query: str, texts: List[str], device, doc_bsize: int) -> List[Tuple[int, float]]:
    qtok, dtok = _get_tokenizers(config)

    # Encode query once
    q_ids, q_mask = qtok.tensorize([query])
    Q = colbert_module.query(q_ids.to(device), q_mask.to(device))  # [1, Lq, dim]

    N = len(texts)
    scores_all: List[float] = []
    idxs_all: List[int] = []

    # Encode docs (all at once if possible)
    if (not doc_bsize) or (doc_bsize >= N):
        d_ids, d_mask = dtok.tensorize(texts)
        D, D_mask = colbert_module.doc(d_ids.to(device), d_mask.to(device), keep_dims='return_mask')
        batch_scores = colbert_module.score(Q, D, D_mask)  # [N]
        scores_all = batch_scores.detach().float().cpu().tolist()
        idxs_all = list(range(N))
    else:
        for start in range(0, N, doc_bsize):
            end = min(start + doc_bsize, N)
            batch_texts = texts[start:end]
            d_ids, d_mask = dtok.tensorize(batch_texts)
            D, D_mask = colbert_module.doc(d_ids.to(device), d_mask.to(device), keep_dims='return_mask')
            batch_scores = colbert_module.score(Q, D, D_mask)  # [B]
            scores_all.extend(batch_scores.detach().float().cpu().tolist())
            idxs_all.extend(range(start, end))

    order = sorted(range(len(scores_all)), key=lambda i: scores_all[i], reverse=True)
    return [(idxs_all[i], float(scores_all[i])) for i in order]

@torch.inference_mode()
def _evaluate_benchmark(colbert_ddp, config, device, step: int, metrics_file: Path) -> Dict[str, float]:
    """
    Computes:
      - mean_nDCG@K per pool size K (same as before) under keys mean_ndcg_k{K}
      - mean_nDCG@15 for K==40 under key mean_ndcg_k40_15  (NEW; matches notebook's ndcg_at_k for k=15, K=40)
    """
    model = colbert_ddp.module if hasattr(colbert_ddp, "module") else colbert_ddp
    was_training = model.training
    model.eval()

    rows_all: List[Dict] = []
    ndcgs_allk: List[float] = []
    ndcgs_per_k: Dict[int, List[float]] = {K: [] for K in EVAL_K_LIST}

    # Collect nDCG@15 specifically for K==40
    ndcg_k40_15_list: List[float] = []

    for K in EVAL_K_LIST:
        in_path = EVAL_DATA_DIR / f"candidates_k{K}.jsonl"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing benchmark file: {in_path}")

        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                case  = row["case"]
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
                    int(any(_coverage_ratio(gt, txt, min_len=GT_MIN_LEN, strip_nums=STRIP_NUMS) >= COVERAGE_THR for gt in gts))
                    for txt in reranked_texts
                ]
                nd_fullK = _ndcg_fullK(rels)
                ndcgs_allk.append(nd_fullK)
                ndcgs_per_k[K].append(nd_fullK)
                rows_all.append({"case": case, "K": K, "nDCG_fullK": nd_fullK})
                if K == 40:
                    nd15 = ndcg_at_k(rels, k=15)
                    ndcg_k40_15_list.append(nd15)

    mean_ndcg_allk = float(sum(ndcgs_allk) / max(1, len(ndcgs_allk)))
    mean_ndcgs_per_k = {
        f"mean_ndcg_k{K}": float(sum(vals) / max(1, len(vals)))
        for K, vals in ndcgs_per_k.items()
    }

    # Aggregate nDCG@15 for K=40
    mean_ndcg_k40_15 = float(sum(ndcg_k40_15_list) / max(1, len(ndcg_k40_15_list)))

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "mean_ndcg_allk": mean_ndcg_allk,
        **mean_ndcgs_per_k,
        # NEW in metrics log: nDCG@15 for K=40
        "mean_ndcg_k40_15": mean_ndcg_k40_15,
        "counts": {
            "allk": len(ndcgs_allk),
            **{f"k{K}": len(vals) for K, vals in ndcgs_per_k.items()},
            "k40_for_ndcg15": len(ndcg_k40_15_list),
        }
    }
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(record) + "\n")

    if was_training:
        model.train()

    return {"mean_ndcg_allk": mean_ndcg_allk,
            **mean_ndcgs_per_k,
            "mean_ndcg_k40_15": mean_ndcg_k40_15}

def _overwrite_dir(path: Path):
    if path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------- TRAIN ------------------------------------
def train(config: ColBERTConfig, triples, queries=None, collection=None):
    # --------------------- Setup & Seeding ---------------------
    config.checkpoint = config.checkpoint or 'bert-base-uncased'
    if config.rank < 1:
        config.help()

    # Label smoothing for CE and for KD targets
    eps = float(getattr(config, 'label_smoothing', 0.1))

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks
    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    # --------------------- Reader -------------------------------------------
    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection,
                                   (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection,
                                 (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError("No collection provided to train()")

    # --------------------- Model & Optimizer --------------------------------
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

    # --------------------- Eval trackers & paths -----------------------------
    best_allk   = 0.56
    best_k40_15 = 0.62  # now refers to nDCG@15 (K=40)

    eval_logs_dir    = Path(config.root) / "eval_logs"
    metrics_file     = eval_logs_dir / "metrics.jsonl"
    best_allk_dir    = Path("/home/bshan/colbert/checkpoints") / "best_allk"
    best_k40_15_dir  = Path("/home/bshan/colbert/checkpoints") / "best_k40_15"

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps
        )

    warmup_bert = getattr(config, 'warmup_bert', None)
    if warmup_bert is not None:
        for p in colbert.bert.parameters():
            p.requires_grad = False

    amp    = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)  # positive index=0

    start_time       = time.time()
    train_loss       = None
    train_loss_mu    = 0.999
    start_batch_idx  = 0

    # --------------------- Main training loop --------------------------------
    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if warmup_bert is not None and warmup_bert <= batch_idx:
            for p in colbert.bert.parameters():
                p.requires_grad = True
            warmup_bert = None

        this_batch_loss = 0.0
        nway_this_seen  = None

        for batch in BatchSteps:
            # batch can be (queries, passages, target_scores, nway_this) OR variants
            if isinstance(batch, tuple) and len(batch) == 4:
                queries_t, passages_t, target_scores, nway_this = batch
                encoding = [queries_t, passages_t]
            elif isinstance(batch, tuple) and len(batch) == 3:
                enc, target_scores, nway_this = batch
                encoding = [enc.to(DEVICE)]
            else:
                queries_t, passages_t, target_scores = batch
                nway_this = getattr(config, 'nway', None)
                encoding = [queries_t, passages_t]

            nway_this_seen = nway_this

            with amp.context():
                scores = colbert(*encoding, nway_override=nway_this)  # [B*nway]
                if getattr(config, 'use_ib_negatives', False):
                    scores, ib_loss = scores

                scores = scores.view(-1, nway_this)  # [B, nway]

                if len(target_scores) and not getattr(config, 'ignore_scores', False):
                    # REMOVED KD!!!!
                    target_scores = torch.tensor(target_scores).view(-1, nway_this).to(DEVICE)

                    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)
                    pos_idx = target_scores.argmax(dim=1)
                    loss = lipo_lambda_loss(scores, pos_idx)
                else:
                    # CE fallback
                    ce = nn.CrossEntropyLoss(label_smoothing=eps)
                    ce_loss = ce(scores, labels[: scores.size(0)])
                    norm = max(1.0, math.log(float(nway_this)))
                    loss = ce_loss / norm

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
            mean_allk     = results["mean_ndcg_allk"]
            mean_k40_15   = results["mean_ndcg_k40_15"]  # nDCG@15 for K=40 (NEW)

            if config.rank in [0, -1]:
                print(f"[eval step {global_step}] mean_nDCG_allK={mean_allk:.4f} | nDCG@15@K=40={mean_k40_15:.4f}")

            if mean_allk > best_allk:
                best_allk = mean_allk
                manage_checkpoints(config, colbert, optimizer, global_step, savepath=str(best_allk_dir)+"_"+str(global_step))

            if mean_k40_15 > best_k40_15:
                best_k40_15 = mean_k40_15
                manage_checkpoints(config, colbert, optimizer, global_step, savepath=str(best_k40_15_dir)+"_"+str(global_step))

        if config.rank < 1:
            if nway_this_seen is not None:
                print_message(f"[step {batch_idx}] EMA loss={train_loss:.6f} (nway={nway_this_seen})")
            else:
                print_message(batch_idx, train_loss)
            # !NOTE REMOVED CHECKPOINTING HERE!
            # manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

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
