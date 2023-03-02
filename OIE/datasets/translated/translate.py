import pathlib
from OIE.datasets.conll2bioes import Conversor
import os
import spacy
from tqdm import tqdm
from OIE.datasets.main import criar_conll
import typer
from googletrans import Translator
from transformers import MarianMTModel, MarianTokenizer, pipeline
import json

app = typer.Typer()


class LoadDataset:
    def __init__(self, dataset_path: str, dataset_name: str, out_path: str):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        with open(self.dataset_path +"/"+ self.dataset_name, "r", encoding="utf-8") as f:
            data = f.read()

        # selecionando apenas exts com arg0 rel e arg1
        data = data.split("\n\t")
        data_norm = []
        for ext in data:
            if "ARG5" not in ext:
                if "ARG4" not in ext:
                    if "ARG3" not in ext:
                        if "ARG2" not in ext:
                            if "ARG1" in ext:
                                if "V" in ext:
                                    if "ARG0" in ext:
                                        data_norm.append(ext)
        path = out_path+"/mod"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        lenght = len(data_norm)
        with open(path + "/" + dataset_name, "a", encoding="utf-8") as f:
            raw = data_norm[:10000]
            raw = "\n\t".join(raw)
            f.write(raw)
        Conversor(path+"/", dataset_name, out_path)

class ArgsRel:
    def __init__(self):
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    #Separa arg1, rel e arg2 da extração a partir da analise sintatica de dependencia da extração
    def get_args_rel(self, ext):
        doc = self.nlp(ext)
        arg1 = ""
        rel = ""
        arg2 = ""
        root_idx = (0,0)
        for token in doc:
            if (token.pos_ == "VERB" and token.dep_ == "ROOT"):
                rel += token.text + " "
                root_idx = (token.idx, token.idx + len(token.text))
        for token in doc:
            if token.idx < root_idx[0]:
                arg1 += token.text + " "
            if token.idx > root_idx[1]:
                arg2 += token.text + " "
        return arg1, rel, arg2


class TranslateDataset:
    def __init__(self, dataset_dir: str, dataset_name: str, out_path: str):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.out_path = out_path

    def translator(self, sent):
        SRC = "en"
        DEST = "pt"
        translator = Translator()
        result = translator.translate(sent, src=SRC, dest=DEST)
        return result.text

    def translator2(sample: list):
        sent = sample[0]
        ext = sample[1]
        model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        pipe = pipeline("translation", model=model_name, device=0)
        trad_sent = pipe(sent, max_length=1000)[0]["translation_text"]
        trad_ext = pipe(ext, max_length=1000)[0]["translation_text"]
        return trad_sent, trad_ext

    def save_dict(self, data_dict):
        with open(self.out_path+"/saida_match/json_dump.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(data_dict))

    def translate(self):
        # estrutura o dataset em um dicionario
        with open(f"{self.out_path}/conll2bioes_output/{self.dataset_name.replace('.conll', '.txt')}",
                  "r", encoding="utf-8") as f:
            data = f.read()
        data = data.split("\n\t")
        data = [ext.split("\n") for ext in data]
        #data = data[:2]
        for ext in data:
            for i in range(len(ext)):
                ext[i] = ext[i].split("\t")
        data_dict = {}
        counter = 0
        for ext in tqdm(data, desc="Traduzindo dataset"):
            sentence = ""
            arg0 = ""
            rel = ""
            arg1 = ""
            for e in ext:
                if e != [""]:
                    sentence += e[0] + " "
                    if "ARG0" in e[8]:
                        arg0 += e[0] + " "
                    if "ARG1" in e[8]:
                        arg1 += e[0] + " "
                    if "V" in e[8]:
                        rel += e[0] + " "

            # traduz sentença, arg0, rel e arg1
            sentence_trad = self.translator(sentence)
            ext_trad = self.translator(arg0 + rel + arg1)
            arg0_trad, rel_trad, arg1_trad = ArgsRel().get_args_rel(ext_trad)
            data_dict[str(counter)] = {"ID": counter, "sent": sentence_trad,
                                       "ext": [{"arg1": arg0_trad, "rel": rel_trad, "arg2": arg1_trad}]}
            counter += 1
        self.save_dict(data_dict)

@app.command()
def run(dataset_dir: str, dataset_name: str, test_size: float, dev_size: float):
    converted = True
    OUT_NAME = dataset_name.replace(".conll", "")
    INPUT_PATH = ""

    path = "outputs"+"/"+OUT_NAME
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    json_dir = path+"/saida_match"
    pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)

    LoadDataset(dataset_dir, dataset_name, path)
    TranslateDataset(dataset_dir, dataset_name, path).translate()
    criar_conll(OUT_NAME, INPUT_PATH, test_size, dev_size, converted)

if __name__ == "__main__":
    app()