import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, WordEmbeddings, OneHotEmbeddings
from flair.models import SequenceTagger
from madgrad import MADGRAD
import typer
from flair.trainers import ModelTrainer
import torch

app = typer.Typer()

@app.command()
def train(epochs: int, name: str, folder: str, train:str, test:str, dev:str):
    # define the structure of the .datasets file
    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label'},# 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          test_file=dev,
                          #dev_file=dev
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
        #OneHotEmbeddings.from_corpus(corpus=corpus, field='pos', min_freq=1, embedding_length=40),
        #OneHotEmbeddings.from_corpus(corpus=corpus, field='dep', min_freq=1, embedding_length=40),
        #OneHotEmbeddings.from_corpus(corpus=corpus, field='ner', min_freq=1, embedding_length=20),
        FlairEmbeddings('pt-forward'),
        FlairEmbeddings('pt-backward')
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # inicializando sequence tagger
    oie = SequenceTagger(#hidden_size=2560,
                         hidden_size=2048,
                         embeddings=embeddings,
                         tag_dictionary=label_dictionary,
                         reproject_embeddings=True,
                         tag_type=label_type,
                         rnn_layers=2,
                         dropout=0.5,
                         locked_dropout=0.0,
                         word_dropout=0.05,
                         )

    pathlib.Path(f"train_output").mkdir(parents=True, exist_ok=True)

    # inicializando trainer
    trainer = ModelTrainer(oie, corpus)
    optimizer = torch.optim.Adam(oie.parameters(), lr=1e-3, betas=(0.9, 0.999))


    # iniciando treino
    trainer.train(f"train_output/{name}",
                  learning_rate=1e-3,
                  min_learning_rate=1e-4,
                  mini_batch_size=16,
                  #mini_batch_chunk_size=1,
                  max_epochs=epochs,
                  patience=4,
                  embeddings_storage_mode='none',
                  #main_evaluation_metric=("micro avg", "precision"),
                  optimizer=MADGRAD(oie.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4),
                  #optimizer=optimizer,
                  save_final_model=False,
                  anneal_factor=0.5,
                  anneal_with_restarts=True,
                  reduce_transformer_vocab=True,
                  #use_swa=True
                  #use_amp=True,
                  )

    '''
    ### FINE TUNING ###
    corpus = ColumnCorpus(data_folder="datasets/validated_splits/normal",
                          column_format={0: 'text', 8: 'label'},  # 9: "pos", 10: "dep", 11: "ner"},
                          train_file="eval/pud_200.txt",
                          test_file="eval/100-gold.txt",
                          dev_file="fine_tune/fine_tune2.txt"
                          )
    trainer = ModelTrainer(oie, corpus)

    trainer.fine_tune(f"train_output/{name}/fine_tune2",
                      learning_rate=1e-3,
                      mini_batch_size=32,
                      max_epochs=20,
                      optimizer=MADGRAD(oie.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4),
                      use_final_model_for_eval=False
                      )
    '''

if __name__ == "__main__":
    app()
