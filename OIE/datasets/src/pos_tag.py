import spacy
import pathlib
import os
from tqdm import tqdm

class PosTag:
    def __init__(self, corpus_dir: str, path: str):
        self.path = path
        self.output_dir = pathlib.Path(self.path+"/saida_pos_tag").mkdir(parents=True, exist_ok=True)
        self.corpus_dir = corpus_dir
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            print("Baixando pt_core_news_lg")
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

        with open(self.corpus_dir, "r", encoding="utf-8") as f:
            self.data = f.read()

        self.dict = {}
        self.counter = 0
        self.extraction = self.data.split("\n\n")
        self.extraction = [x.split("\n") for x in self.extraction]
        aux = []
        for ext in self.extraction:
            lines = []
            for line in ext:
                lines.append(line.split("\t"))
            aux.append(lines)
        self.extraction = aux


        for ext in self.extraction:
            if ext != [['']]:
                label = [line[8] for line in ext if line != ['']]
                sentence = [line[0] for line in ext if line != ['']]
                sentence = [x for x in sentence if x != " "]

                self.dict[self.counter] = {"sent": sentence, "label": label}
                self.counter += 1


    def pos_tag(self):
        for key in tqdm(range(len(self.dict)), desc="Tagging"):
            sentence = self.nlp(" ".join(self.dict[key]["sent"]))
            pos_tag = [token.pos_ for token in sentence]
            dep_tag = [token.dep_ for token in sentence]

            f = lambda x: x if x != "" else "O"
            ner_tag = [f(token.ent_type_) for token in sentence]

            self.dict[key]["ner_tag"] = ner_tag
            self.dict[key]["dep_tag"] = dep_tag
            self.dict[key]["pos_tag"] = pos_tag


    def save(self, name):
        raw_file = ""
        with open(f"{self.path}\saida_pos_tag\{name}", "a", encoding="utf-8") as f:
            for key in tqdm(range(len(self.dict)), desc="Salvando"):
                sentence = self.dict[key]["sent"]
                sentence = [x for x in sentence if x != ""]
                label = self.dict[key]["label"]
                pos_tag = self.dict[key]["pos_tag"]
                dep_tag = self.dict[key]["dep_tag"]
                ner_tag = self.dict[key]["ner_tag"]

                for i in range(len(sentence)):
                    raw_file += f"{sentence[i]}\tXX\t-\t-\t-\t-\t-\t*\t{label[i]}\t{pos_tag[i]}\t{dep_tag[i]}\t{ner_tag[i]}\t-\n"
                raw_file += "\n"
            raw_file = raw_file[:-2]
            f.write(raw_file)

    def run(self, name):
        self.pos_tag()
        self.save(name)

if __name__ == "__main__":
    eng = PosTag("E:\Dev/3-facul\PIBIC\PLN\Flair-oie\OIE\datasets/validated_splits/normal/trad_v3/trad.txt", "E:\Dev/3-facul\PIBIC\PLN\Flair-oie\OIE\datasets\outputs")
    eng.run("trad_v3")