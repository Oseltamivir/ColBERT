import os
import ujson
import random
import numpy as np

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples


class LazyBatcher():
    """
    Dynamic-nway version:
      • Picks a single nway_this per batch uniformly at random in [config.nway_min, config.nway_max].
      • For each sample, keeps the first candidate (positive) at index 0.
      • Chooses (nway_this-1) negatives with probability proportional to their distillation scores.
      • Emits nway_this alongside each yielded micro-batch so the trainer can reshape correctly.
    """
    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps

        # Backwards-compat defaults (in case only config.nway is provided)
        self.nway_min = getattr(config, 'nway_min', getattr(config, 'nway', 40))
        self.nway_max = getattr(config, 'nway_max', getattr(config, 'nway', 80))
        assert 1 <= self.nway_min <= self.nway_max

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway_max).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)
        assert len(self.triples) > 0, "Received no triples on which to train."
        assert len(self.queries) > 0, "Received no queries on which to train."
        assert len(self.collection) > 0, "Received no collection on which to train."

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        # --- sample a per-batch nway_this in [nway_min, nway_max], inclusive
        nway_this = random.randint(self.nway_min, self.nway_max)

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            row = self.triples[position]     # [qid, [pid, score], [pid, score], ...]
            qid, *pairs = row
            query = self.queries[qid]

            # Split pids/scores. By your data contract, pairs[0] is the unique positive (score=1.0)
            try:
                pids_all, scores_all = zipstar(pairs)  # tuples
            except Exception:
                # If no scores were stored (shouldn't happen per your data), fall back to empty list
                pids_all, scores_all = zip(*pairs), []

            # Ensure lists
            pids_all = list(pids_all)
            scores_all = list(scores_all) if len(scores_all) else [1.0] + [0.0]*(len(pids_all)-1)

            # Positive always first
            pos_pid = pids_all[0]
            pos_score = scores_all[0]

            neg_pids = pids_all[1:]
            neg_scores = scores_all[1:]

            k_neg = nway_this - 1
            pop = len(neg_pids)

            if pop == 0:
                # No negatives at all for this row
                raise ValueError(f"No negatives for qid={qid}; need at least 1.")

            # If the row has fewer negatives than k_neg, allow replacement to keep the batch shape.
            replace = k_neg > pop

            weights = np.asarray(neg_scores, dtype=np.float64)
            weights = np.clip(weights, 0.0, None)

            if weights.sum() <= 0:
                # All-zero scores -> uniform sampling
                probs = None
            else:
                probs = weights / weights.sum()
                # If fewer non-zero entries than needed, fall back to uniform
                if (probs > 0).sum() < k_neg and not replace:
                    probs = None

            # Optional: tiny epsilon smoothing if you prefer to keep weighted sampling
            # eps = 1e-6
            # if probs is not None and (probs > 0).sum() < k_neg:
            #     probs = (probs + eps) / (probs + eps).sum()

            idxs = np.random.choice(pop, size=k_neg, replace=replace, p=probs)
            chosen_neg_pids = [neg_pids[i] for i in idxs]
            chosen_neg_scores = [neg_scores[i] for i in idxs]


            # Assemble per-sample lists with positive at index 0
            pids = [pos_pid] + chosen_neg_pids
            scores = [pos_score] + chosen_neg_scores

            passages = [self.collection[pid] for pid in pids]

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)

        assert len(all_queries) == self.bsize
        assert len(all_passages) == nway_this * self.bsize, (len(all_passages), nway_this, self.bsize)

        return self._collate_with_nway(all_queries, all_passages, all_scores, nway_this)

    def _collate_with_nway(self, queries, passages, scores, nway_this: int):
        """
        Wrap the underlying tensorize_triples generator and append nway_this
        to each yielded micro-batch so the trainer can reshape with it.
        """
        B = self.bsize // self.accumsteps
        gen = self.tensorize_triples(queries, passages, scores, B, nway_this)

        # The upstream yields either:
        #   (queries_tensor, passages_tensor, target_scores_list)    OR
        #   (encoding_tensor, target_scores_list)
        # We keep the same structure but append nway_this.
        for item in gen:
            if isinstance(item, tuple) and len(item) == 3:
                q_t, p_t, tgt_scores = item
                yield (q_t, p_t, tgt_scores, nway_this)
            elif isinstance(item, tuple) and len(item) == 2:
                enc_t, tgt_scores = item
                yield (enc_t, tgt_scores, nway_this)
            else:
                # Fallback: still attach nway_this
                yield (*item, nway_this)
