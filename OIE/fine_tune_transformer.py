import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from madgrad import MADGRAD
import typer

app = typer.Typer()


@app.command()
def train(epochs: int, name: str, folder: str, train: str, test: str, dev: str):
    # define the structure of the .datasets file
    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label'},#, 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          #test_file=test,
                          dev_file=dev
                          )

    label_type = "label"  # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
    print(label_dictionary)


    trm = TransformerWordEmbeddings('bert-base-multilingual-cased',
                                            fine_tune=True,
                                            layers="-1",
                                            subtoken_pooling="first",
                                            use_context=True,
                                            )

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=trm,
                            tag_dictionary=label_dictionary,
                            tag_type='label',
                            use_crf=True,
                            use_rnn=True,
                            reproject_embeddings=False,
                            )

    pathlib.Path(f"train_output/transformer/{name}").mkdir(parents=True, exist_ok=True)

    # inicializando trainer
    trainer = ModelTrainer(tagger, corpus.downsample(0.5))

    # fine tune
    trainer.fine_tune(f"train_output/transformer/{name}",
                      learning_rate=5e-5,
                      mini_batch_size=8,
                      max_epochs=epochs,
                      optimizer=MADGRAD
                      )


if __name__ == "__main__":
    app()