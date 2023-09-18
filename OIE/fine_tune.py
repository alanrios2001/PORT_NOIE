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
    model_name = "TA_bertina4/feedback"
    pre_fine_tune = ["eval/pud_200.txt", "eval/100-gold.txt", "eval/100-gold.txt"]
    feed_back = ["feedback/fb_dataset.txt", "eval/100-gold.txt", "eval/100-gold.txt"]

    # inicializando sequence tagger
    try:
        oie = SequenceTagger.load("train_output/" + model_name + "/best-model.pt")
        print("best model loaded")
    except:
        oie = SequenceTagger.load("train_output/" + model_name + "/final-model.pt")
        print("final model loaded")

    dataset = pre_fine_tune
    corpus = ColumnCorpus(data_folder="datasets/validated_splits/normal",
                          column_format={0: 'text', 8: 'label'},  # 9: "pos", 10: "dep", 11: "ner"},
                          train_file=dataset[0],
                          dev_file=dataset[1],
                          test_file=dataset[2]
                          )

    trainer = ModelTrainer(oie, corpus)

    trainer.fine_tune(f"train_output/{model_name}/fn",
                      learning_rate=1e-8,
                      mini_batch_size=4,
                      max_epochs=20,
                      optimizer=MADGRAD,
                      use_final_model_for_eval=False,
                      embeddings_storage_mode='cpu',
                      save_final_model=False
                      )


if __name__ == "__main__":
    app()