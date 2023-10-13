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



def train():
    # define the structure of the .datasets file
    name = "albertina_s2"
    epochs = 7
    folder = "datasets/validated_splits/normal/TA4"
    train = "s2_DStrain_corpus.txt"
    test = "100-gold.txt"
    dev = "s2_DSvalid_corpus.txt"

    corpus = ColumnCorpus(data_folder=folder,
                          column_format={0: 'text', 8: 'label'},  # , 9: "pos", 10: "dep", 11: "ner"},
                          train_file=train,
                          test_file=test,
                          dev_file=dev,
                          autofind_splits=False
                          )

    label_type = "label"  # criando dicionario de tags
    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
    print(label_dictionary)

    bert = TransformerWordEmbeddings('neuralmind/bert-base-portuguese-cased',
                                     layers="-1",
                                     subtoken_pooling="first_last",
                                     use_context=True,
                                     fine_tune=True,
                                     )

    roberta = TransformerWordEmbeddings('thegoodfellas/tgf-xlm-roberta-base-pt-br',
                                        layers="-1",
                                        subtoken_pooling="first_last",
                                        use_context=True,
                                        fine_tune=True,
                                        )

    albertina_base = TransformerWordEmbeddings('PORTULAN/albertina-ptbr-base',
                                          layers="-1,-2,-3",
                                          layer_mean=True,
                                          subtoken_pooling="mean",
                                          use_context=True,
                                          fine_tune=True,
                                          )

    albertina_nobrwac = TransformerWordEmbeddings('PORTULAN/albertina-ptbr-nobrwac',
                                          layers="-1",
                                          subtoken_pooling="first_last",
                                          use_context=True,
                                          fine_tune=True,
                                          )

    trm = albertina_base

    
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=trm,
                            tag_dictionary=label_dictionary,
                            tag_type='label',
                            use_crf=True,
                            use_rnn=False,
                            rnn_layers=2,
                            #locked_dropout=0.5,
                            #dropout=0.5,
                            #word_dropout=0.05,
                            reproject_embeddings=False,
                            )

    #tagger = SequenceTagger.load(f"train_output/{name}/best-model.pt")


    # inicializando trainer
    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(f"train_output/{name}",
                      learning_rate=1e-7,
                      mini_batch_size=2,
                      #chunk_batch_size=1,
                      #min_learning_rate=1e-6,
                      max_epochs=epochs,
                      optimizer=MADGRAD,
                      #warmup_fraction=1.0,
                      #decoder_lr_factor=20,
                      #scheduler=AnnealOnPlateau,
                      use_final_model_for_eval=False,
                      )

train()
