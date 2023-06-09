import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
#from trainers.trainer import ModelTrainer
from flair.trainers import ModelTrainer
from madgrad import MADGRAD
import typer

app = typer.Typer()


@app.command()
def fine_tune(name: str):
    # define the structure of the .datasets file
    corpus = ColumnCorpus(data_folder="datasets/validated_splits/normal",
                          column_format={0: 'text', 8: 'label'},# 9: "pos", 10: "dep", 11: "ner"},
                          train_file="fine_tune/fine_tune2.txt",
                          test_file="eval/100-gold.txt",
                          dev_file="eval/100-gold.txt"
                          )


    # inicializando sequence tagger
    try:
        oie = SequenceTagger.load("train_output/" + name + "/final-model.pt")
    except:
        oie = SequenceTagger.load("train_output/" + name + "/best-model.pt")

    pathlib.Path(f"train_output/{name}/fine_tune").mkdir(parents=True, exist_ok=True)

    # inicializando trainer
    trainer = ModelTrainer(oie, corpus)


    '8 epochs first round, second, 16'
    # fine tune

    trainer.fine_tune(f"train_output/{name}/fine_tune",
                      learning_rate=1e-4,
                      mini_batch_size=16,
                      max_epochs=20,
                      optimizer=MADGRAD,
                      use_final_model_for_eval=False
                      )


    corpus = ColumnCorpus(data_folder="datasets/validated_splits/normal",
                          column_format={0: 'text', 8: 'label'},  # 9: "pos", 10: "dep", 11: "ner"},
                          train_file="eval/pud_200.txt",
                          test_file="eval/100-gold.txt",
                          dev_file="eval/100-gold.txt"
                          )

    oie = SequenceTagger.load("train_output/" + name + "/fine_tune/best-model.pt")
    trainer = ModelTrainer(oie, corpus)


    trainer.fine_tune(f"train_output/{name}/fine_tune2",
                      learning_rate=1e-4,
                      mini_batch_size=16,
                      max_epochs=20,
                      optimizer=MADGRAD,
                      use_final_model_for_eval=False
                      )


if __name__ == "__main__":
    app()