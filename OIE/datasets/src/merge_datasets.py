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
                if data[-1] == "":
                    data = data[:-1]
                random.shuffle(data)
                data = "\n\n".join(data)
                self.merged += data

        path = pathlib.Path("outputs/merges")
        path.mkdir(parents=True, exist_ok=True)

        with open(f"outputs/merges/{output_name}_corpus.txt", "a", encoding="utf-8") as file:
            file.write(self.merged)

        print(f"len {output_name}: ", len(self.merged.split("\n\n")))
