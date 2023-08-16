import pathlib
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, WordEmbeddings, OneHotEmbeddings, PooledFlairEmbeddings
from flair.models import SequenceTagger
from madgrad import MADGRAD
import typer
from flair.trainers import ModelTrainer

app = typer.Typer()

def train():
    # define the structure of the .datasets file
    name = "TA3"
    epochs = 150
    folder = "datasets/validated_splits/normal"
    train = "TransAlign3/TA3_train.txt"
    test = "TransAlign3/s2_TA_valid.txt"
    dev = "TransAlign3/TA3_dev.txt"

    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label'},# 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          test_file=test,
                          dev_file=dev
                          )

    label_type = "label"    # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
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

    albertina = TransformerWordEmbeddings("PORTULAN/albertina-ptbr-base")

    transformer = albertina

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
    oie = SequenceTagger(hidden_size=1024,
                         #hidden_size=2048,
                         embeddings=embeddings,
                         tag_dictionary=label_dictionary,
                         reproject_embeddings=True,
                         tag_type=label_type,
                         rnn_layers=2,
                         dropout=0.3,
                         locked_dropout=0.0,
                         word_dropout=0.05,
                         )

    pathlib.Path(f"train_output").mkdir(parents=True, exist_ok=True)

    # inicializando trainer
    trainer = ModelTrainer(oie, corpus)

    # iniciando treino
    trainer.train(f"train_output/{name}",
                  learning_rate=1e-3,
                  min_learning_rate=1e-4,
                  mini_batch_size=32,
                  #mini_batch_chunk_size=1,
                  max_epochs=epochs,
                  patience=8,
                  embeddings_storage_mode='none',
                  optimizer=MADGRAD,
                  save_final_model=False,
                  anneal_factor=0.5,
                  anneal_with_restarts=True,
                  )


    ### FINE TUNING ###
    corpus = ColumnCorpus(data_folder="datasets/validated_splits/normal",
                          column_format={0: 'text', 8: 'label'},  # 9: "pos", 10: "dep", 11: "ner"},
                          train_file="eval/pud_200.txt",
                          test_file="eval/100-gold.txt",
                          dev_file="eval/100-gold.txt"
                          #dev_file="fine_tune/fine_tune2.txt"
                          )
    trainer = ModelTrainer(oie, corpus)

    trainer.fine_tune(f"train_output/{name}/fine_tune",
                      learning_rate=1e-3,
                      mini_batch_size=32,
                      max_epochs=20,
                      optimizer=MADGRAD,
                      use_final_model_for_eval=False
                      )


train()
