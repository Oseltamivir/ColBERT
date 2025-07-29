from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer
from pathlib import Path

ROOT = "/home/bshan/colbert/"   # where checkpoints go
DATA = Path("/home/bshan/colbert/scores80")    # output of the converter

def train():
    with Run().context(RunConfig(nranks=4, experiment="zw80_not_dyn_scores")):
        cfg = ColBERTConfig(
            root=ROOT,
            bsize=8,
            lr=2e-5,
            warmup=200,
            doc_maxlen=512,
            attend_to_mask_tokens=False,
            nway=40,                       # This should match your prep script
            accumsteps=1,
            similarity="cosine",
            use_ib_negatives=True,
            reranker=False
        )

        trainer = Trainer(
            triples=str(DATA / "triples_scores.jsonl"),
            queries=str(DATA / "queries.tsv"),
            collection=str(DATA / "collection.tsv"),
            config=cfg,

        )

        trainer.train(checkpoint="colbert-ir/colbertv2.0")
        #trainer.train(checkpoint=" /home/bshan/colbert/experiments/ncdcg/none/2025-07/18/12.56.11/checkpoints/colbert")

if __name__ == '__main__':
    train()