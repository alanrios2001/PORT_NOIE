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
def fine_tune():
    model_name = "TA_bertina3"

    # inicializando sequence tagger
    try:
        oie = SequenceTagger.load("train_output/" + model_name + "/best-model.pt")
        print("best model loaded")
    except:
        oie = SequenceTagger.load("train_output/" + model_name + "/final-model.pt")
        print("final model loaded")


    corpus = ColumnCorpus(data_folder="datasets/validated_splits/normal",
                          column_format={0: 'text', 8: 'label'},  # 9: "pos", 10: "dep", 11: "ner"},
                          train_file="eval/pud_200.txt",
                          test_file="eval/100-gold.txt",
                          dev_file="eval/100-gold.txt"
                          )

    trainer = ModelTrainer(oie, corpus)

    trainer.fine_tune(f"train_output/{model_name}/fine_tune",
                      learning_rate=1e-6,
                      mini_batch_size=4,
                      max_epochs=20,
                      optimizer=MADGRAD,
                      use_final_model_for_eval=False,
                      embeddings_storage_mode='cpu',
                      save_final_model=False
                      )


if __name__ == "__main__":
    app()