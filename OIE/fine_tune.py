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
def train(epochs: int, name: str, folder: str, train: str, test: str, dev: str):
    # define the structure of the .datasets file
    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label', 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          test_file=test,
                          dev_file=dev
                          )

    label_type = "label"  # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
    print(label_dictionary)


    # inicializando sequence tagger
    try:
        oie = SequenceTagger.load("train_output/" + name + "/final-model.pt")
    except:
        oie = SequenceTagger.load("train_output/" + name + "/best-model.pt")

    pathlib.Path(f"train_output/{name}/fine_tune").mkdir(parents=True, exist_ok=True)

    # inicializando trainer
    trainer = ModelTrainer(oie, corpus)

    # fine tune
    trainer.fine_tune(f"train_output/{name}/fine_tune",
                      learning_rate=5e-6,
                      mini_batch_size=8,
                      max_epochs=epochs,
                      optimizer=MADGRAD,
                      )


if __name__ == "__main__":
    app()