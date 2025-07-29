from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer
from pathlib import Path

ROOT = "/home/bshan/colbert/"
DATA = Path("/home/bshan/colbert/scores80")

def train():
    with Run().context(RunConfig(nranks=4, experiment="zw80_scores_distil_smooth")):
        cfg = ColBERTConfig(
            root=ROOT,
            bsize=4,
            lr=1e-5,
            warmup=100,
            doc_maxlen=495,
            attend_to_mask_tokens=False,
            # Keep a nominal nway for legacy defaults, but enable dynamic bounds:
            nway=80,
            nway_min=40,
            nway_max=60,
            accumsteps=1,
            similarity="cosine",
            use_ib_negatives=True,
            reranker=False,
            # distillation_alpha is already used in your script
        )

        trainer = Trainer(
            triples=str(DATA / "triples_scores.jsonl"),
            queries=str(DATA / "queries.tsv"),
            collection=str(DATA / "collection.tsv"),
            config=cfg,
        )
        trainer.train(checkpoint="colbert-ir/colbertv2.0")

if __name__ == '__main__':
    train()
