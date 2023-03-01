import pathlib
import random

class Conversor:
    def __init__(self, path:str, conll_name: str):
        self.conll_name = conll_name
        self.path = path
        self.conll = self.path + self.conll_name
        self.last_tag = ""
        self.total_len = 0
        with open(self.conll, "r", encoding="utf-8") as file:
            self.lines = file.read()
            self.lines = self.lines.replace("(ARG0*)", "S-ARG0")
            self.lines = self.lines.replace("(ARG0*", "B-ARG0")
            self.lines = self.lines.replace("(ARG1*)", "S-ARG1")
            self.lines = self.lines.replace("(ARG1*", "B-ARG1")
            self.lines = self.lines.replace("(V*)", "S-V")
            self.lines = self.lines.replace("(V*", "B-V")
            self.last_tag = ""
            self.lines = self.lines.split("\n")
            for idx in range(len(self.lines)):
                line = self.lines[idx]
                if "B-ARG0" in line:
                    self.last_tag = "B-ARG0"
                elif "B-ARG1" in line:
                    self.last_tag = "B-ARG1"
                elif "B-V" in line:
                    self.last_tag = "B-V"
                elif "*)" in line:
                    new_line = line.replace("*)", "E-" + self.last_tag[2:])
                    self.lines[idx] = new_line
                if len(line) == 0:
                    self.last_tag = ""

            for idx in range(len(self.lines)):
                line = self.lines[idx].split("\t")
                if "B-ARG0" in line:
                    self.last_tag = "B-ARG0"
                elif "B-ARG1" in line:
                    self.last_tag = "B-ARG1"
                elif "B-V" in line:
                    self.last_tag = "B-V"
                if "E-ARG0" in line:
                    self.last_tag = ""
                elif "E-ARG1" in line:
                    self.last_tag = ""
                elif "E-V" in line:
                    self.last_tag = ""
                elif len(line) == 1:
                    self.last_tag = ""

                if len(line) > 1:
                    if line[11] == "*":
                        if self.last_tag != "":
                            if self.last_tag == "B-ARG0":
                                line[11] = "I-ARG0"
                            elif self.last_tag == "B-ARG1":
                                line[11] = "I-ARG1"
                            elif self.last_tag == "B-V":
                                line[11] = "I-V"
                        if self.last_tag == "":
                            line[11] = "O"
                line = line[3:]
                line = "".join(str(x) + "\t" for x in line)
                line = line[:-1]
                self.lines[idx] = line

        path = pathlib.Path("conll2bioes_output")
        path.mkdir(parents=True, exist_ok=True)

        with open(f"conll2bioes_output/{conll_name.replace('.conll', '')}.txt", "a", encoding="utf-8") as file:
            for line in self.lines:
                file.writelines(line+'\n')


    def train_dev_test(self, test_slice: float, dev_slice: float):
        with open(f"conll2bioes_output/{self.conll_name.replace('.conll', '')}.txt", "r", encoding="utf-8") as file:
            lines = file.read()
            lines = lines.split("\n\n")
            random.shuffle(lines)
            test_size = int(len(lines) * test_slice)
            dev_size = int(len(lines) * dev_slice)
            train_size = len(lines) - test_size - dev_size
            train = lines[:train_size]
            dev = lines[train_size:train_size + dev_size]
            test = lines[train_size + dev_size:]
            self.total_len += len(lines)
            file.close()

        if len(dev) and len(test) != 0:
            with open(f"conll2bioes_output/{self.conll_name.replace('.conll', '')}_train.txt", "a", encoding="utf-8") as file:
                file.writelines("\n\n".join(train))
                file.close()
        if len(dev) > 0:
            with open(f"conll2bioes_output/{self.conll_name.replace('.conll', '')}_dev.txt", "a", encoding="utf-8") as file:
                file.writelines("\n\n".join(dev))
                file.close()
        if len(test) > 0:
            with open(f"conll2bioes_output/{self.conll_name.replace('.conll', '')}_test.txt", "a", encoding="utf-8") as file:
                file.writelines("\n\n".join(test))
                file.close()

        print("train: ", len(train), "|| dev: ", len(dev), "|| test: ", len(test))


if __name__ == "__main__":
    from pos_tag import PosTag
    datasets = ["pud_200.conll"]

    #path = pathlib.Path("other_corpus")
    #path.mkdir(parents=True, exist_ok=True)

    path = pathlib.Path("other_corpus/mod")
    path.mkdir(parents=True, exist_ok=True)

    #seleciona apenas extrações corretas(com arg0, v e arg1)
    total = 0
    for dataset in datasets:
        with open("other_corpus/" + dataset, "r", encoding="utf-8") as file:
            with open("other_corpus/mod/" + dataset, "a", encoding="utf-8") as file2:
                lines = file.read()
                lines = lines.split("\n\n")
                for line in lines:
                    if "ARG0" in line and "ARG1" in line and "V" in line:
                        file2.write(line+"\n\n")

        #apos selecionar extrações corretas, converte para BIOES
        conv = Conversor("other_corpus/mod/", dataset)
        PosTag(f"conll2bioes_output/{dataset.replace('.conll', '')}.txt").run(f"{dataset.replace('.conll', '')}.txt")
        conv.train_dev_test(0.0, 0.0)
        total += conv.total_len
    print("total: ", total)