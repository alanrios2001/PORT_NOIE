import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
#from trainers.trainer import ModelTrainer
from flair.trainers import ModelTrainer
from madgrad import MADGRAD
import typer
import torch

app = typer.Typer()


@app.command()
def fine_tune(name: str):

    # inicializando sequence tagger
    try:
        oie = SequenceTagger.load("train_output/" + name + "/best-model.pt")
    except:
        oie = SequenceTagger.load("train_output/" + name + "/final-model.pt")


    corpus = ColumnCorpus(data_folder="datasets/validated_splits/normal",
                          column_format={0: 'text', 8: 'label'},  # 9: "pos", 10: "dep", 11: "ner"},
                          train_file="eval/pud_200.txt",
                          test_file="eval/100-gold.txt",
                          dev_file="eval/100-gold.txt"
                          )

    trainer = ModelTrainer(oie, corpus)

    trainer.fine_tune(f"train_output/{name}/fine_tune",
                      learning_rate=1e-3,
                      mini_batch_size=32,
                      max_epochs=20,
                      optimizer=MADGRAD,
                      use_final_model_for_eval=False
                      )


if __name__ == "__main__":
    app()