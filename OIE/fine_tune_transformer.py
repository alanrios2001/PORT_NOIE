import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
#from flair.trainers import ModelTrainer
from trainers.trainer import ModelTrainer
from madgrad import MADGRAD
import typer
from flair.training_utils import (
    AnnealOnPlateau,
)

app = typer.Typer()


@app.command()
def train(epochs: int, name: str, folder: str, train: str, test: str, dev: str):
    # define the structure of the .datasets file
    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label'},  # , 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          test_file=test,
                          dev_file=dev
                          )

    label_type = "label"  # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
    print(label_dictionary)

    bert = TransformerWordEmbeddings('bert-base-multilingual-cased',
                                     layers="-1",
                                     subtoken_pooling="first",
                                     use_context=True,
                                     fine_tune=True,
                                     )

    roberta = TransformerWordEmbeddings('bert-base-multilingual-cased',
                                        layers="-1",
                                        subtoken_pooling="first",
                                        use_context=True,
                                        fine_tune=True,
                                        )

    trm = bert

    tagger = SequenceTagger(hidden_size=1024,
                            embeddings=trm,
                            tag_dictionary=label_dictionary,
                            tag_type='label',
                            use_crf=True,
                            use_rnn=True,
                            rnn_layers=2,
                            locked_dropout=0.0,
                            dropout=0.5,
                            reproject_embeddings=False,
                            )

    # inicializando trainer
    trainer = ModelTrainer(tagger, corpus)

    # fine tune
    trainer.fine_tune(f"train_output/{name}",
                      learning_rate=5e-6,
                      mini_batch_size=16,
                      max_epochs=epochs,
                      optimizer=MADGRAD,
                      decoder_lr_factor=2000,
                      scheduler=AnnealOnPlateau,
                      )


if __name__ == "__main__":
    app()
