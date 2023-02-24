import pathlib
import random

class Merge:
    def __init__(self, datasets: list, output_name: str):
        self.datasets = datasets
        self.merged = ""

        for i in range(len(datasets)):
            with open(datasets[i], "r", encoding="utf-8") as file:
                data = file.read()
                data = data.split("\n\n")
                random.shuffle(data)
                data = "\n\n".join(data)
                self.merged += data

        path = pathlib.Path("merges")
        path.mkdir(parents=True, exist_ok=True)

        with open(f"merges/{output_name}.txt", "a", encoding="utf-8") as file:
            file.write(self.merged)

        print(f"len {output_name}: ", len(self.merged.split("\n\n")))

corpus = ["saida_match1/PTOIE_corpus.txt","saida_match1/PTOIE2_corpus.txt"]

datasets_train = ["saida_match1/PTOIE_train.txt",
                  "conll2bioes_output/gamalho.txt",
                  "conll2bioes_output/pragmatic_ceten.txt",
                  "conll2bioes_output/pragmatic_wiki.txt",
                  "conll2bioes_output/pud_100.txt",
                  "conll2bioes_output/pud_200.txt"]

datasets_test = ["saida_match1/PTOIE_test.txt"]

datasets_dev = ["saida_match1/PTOIE_dev.txt"]

names = ["PTOIE_plus_train", "PTOIE_plus_test", "PTOIE_plus_dev", "merge_train", "merge_test", "merge_dev","PTOIE_plus_corpus"]

control = False
if control:
    Merge(corpus, names[-1])

elif len(datasets_dev) == 6:
    Merge(datasets_train, names[0])
    Merge(datasets_test, names[1])
    Merge(datasets_dev, names[2])
else:
    Merge(datasets_train, names[3])
    Merge(datasets_test, names[4])
    Merge(datasets_dev, names[5])