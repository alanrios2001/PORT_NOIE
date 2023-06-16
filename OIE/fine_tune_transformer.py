import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
#from trainers.trainer import ModelTrainer
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
                          #dev_file=dev
                          )

    label_type = "label"  # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
    print(label_dictionary)

    bert = TransformerWordEmbeddings('neuralmind/bert-large-portuguese-cased',
                                     layers="-1",
                                     subtoken_pooling="first_last",
                                     use_context=True,
                                     fine_tune=True,
                                     )

    roberta = TransformerWordEmbeddings('xlm-roberta-large',
                                        layers="-1",
                                        subtoken_pooling="first",
                                        use_context=True,
                                        fine_tune=True,
                                        )

    trm = roberta

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=trm,
                            tag_dictionary=label_dictionary,
                            tag_type='label',
                            use_crf=False,
                            use_rnn=False,
                            rnn_layers=2,
                            #locked_dropout=0.0,
                            #dropout=0.5,
                            #word_dropout=0.0,
                            reproject_embeddings=False,
                            )

    # inicializando trainer
    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(f"train_output/{name}",
                      learning_rate=5e-6,
                      mini_batch_size=4,
                      #chunk_batch_size=1,
                      max_epochs=epochs,
                      optimizer=MADGRAD(tagger.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4),
                      #decoder_lr_factor=20,
                      #scheduler=AnnealOnPlateau,
                      use_final_model_for_eval=False
                      )

    """
    # fine tune
    trainer.fine_tune(f"train_output/{name}",
                      learning_rate=1e-6,
                      mini_batch_size=2,
                      chunk_batch_size=1,
                      max_epochs=epochs,
                      optimizer=MADGRAD,
                      decoder_lr_factor=20,
                      scheduler=AnnealOnPlateau,
                      use_final_model_for_eval=False
                      )
    """

if __name__ == "__main__":
    app()
