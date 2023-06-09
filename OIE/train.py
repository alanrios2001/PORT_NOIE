import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, WordEmbeddings, OneHotEmbeddings
from flair.models import SequenceTagger
from madgrad import MADGRAD
import typer
from flair.trainers import ModelTrainer

app = typer.Typer()

@app.command()
def train(epochs: int, name: str, folder: str, train:str, test:str, dev:str):
    # define the structure of the .datasets file
    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label'},# 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          test_file=test,
                          dev_file=dev
                          )

    label_type = "label"    # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
    print(label_dictionary)

    roberta = TransformerWordEmbeddings(
        "rdenadai/BR_BERTo"
    )

    roberta2 = TransformerWordEmbeddings(
        "josu/roberta-pt-br"
    )

    bert = TransformerWordEmbeddings(
        "neuralmind/bert-base-portuguese-cased",
    )

    transformer = bert

    embedding_types = [
        #transformer,
        WordEmbeddings("pt"),
        #OneHotEmbeddings.from_corpus(corpus=corpus, field='pos', min_freq=6, embedding_length=16),
        #OneHotEmbeddings.from_corpus(corpus=corpus, field='dep', min_freq=6, embedding_length=35),
        FlairEmbeddings('pt-forward'),
        FlairEmbeddings('pt-backward')
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # inicializando sequence tagger
    oie = SequenceTagger(hidden_size=1024,
                         embeddings=embeddings,
                         tag_dictionary=label_dictionary,
                         tag_type=label_type,
                         rnn_layers=5,
                         dropout=0.3,
                         locked_dropout=0.0,
                         word_dropout=0.1,
                         )

    pathlib.Path(f"train_output").mkdir(parents=True, exist_ok=True)

    # inicializando trainer
    trainer = ModelTrainer(oie, corpus)

    # iniciando treino
    trainer.train(f"train_output/{name}",
                  learning_rate=1e-3,
                  min_learning_rate=1e-5,
                  mini_batch_size=16,
                  #mini_batch_chunk_size=1,
                  max_epochs=epochs,
                  patience=4,
                  embeddings_storage_mode='cpu',
                  optimizer=MADGRAD,
                  save_final_model=False,
                  anneal_factor=0.5,
                  anneal_with_restarts=True,
                  #reduce_transformer_vocab=True,
                  #use_swa=True
                  #use_amp=True,
                  )

if __name__ == "__main__":
    app()
