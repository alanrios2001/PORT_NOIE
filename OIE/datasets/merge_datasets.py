import pathlib
import random
import train_test_dev as split

class Merge:
    def __init__(self, datasets: list, output_name: str):
        self.datasets = datasets
        self.merged = ""

        for i in range(len(datasets)):
            with open(datasets[i], "r", encoding="utf-8") as file:
                data = file.read()
                data = data.split("\n\n")
                random.shuffle(data)
                random.shuffle(data)
                random.shuffle(data)
                data = "\n\n".join(data)
                self.merged += data

        path = pathlib.Path("merges")
        path.mkdir(parents=True, exist_ok=True)

        with open(f"merges/{output_name}_corpus.txt", "a", encoding="utf-8") as file:
            file.write(self.merged)

        print(f"len {output_name}: ", len(self.merged.split("\n\n")))

datasets = ["saida_pos_tag/pud_200.txt", "translated/saida_pos_tag/LSOIE_corpus.txt"]


TEST_SIZE = 0.1
DEV_SIZE = 0.0
OUTPUT_NAME = "PtOIE_LsOIE_plus"
IN_PATH = "merges"
OUT_PATH = "saida_pos_tag"
Merge(datasets, OUTPUT_NAME)
split.train_dev_test(TEST_SIZE, DEV_SIZE, OUTPUT_NAME, IN_PATH, OUT_PATH)
