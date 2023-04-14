from src.conll2bioes import Conversor
import os
import spacy
from tqdm.auto import tqdm
from main import criar_conll
import typer
from deep_translator import GoogleTranslator
from transformers import pipeline
import json
import pathlib
from diskcache import Cache
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions, clean_extraction

app = typer.Typer()


class LoadDataset:
    def __init__(self,
                 dataset_path: str,
                 dataset_name: str,
                 out_path: str
                 ):
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


        path = out_path + "/mod"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + "/" + dataset_name, "a", encoding="utf-8") as f:
            raw = data_norm
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
        ext = transform_portuguese_contractions(ext)
        doc = self.nlp(ext)
        doc_dict = {}
        i = 0
        for token in doc:
            doc_dict[i] = {"text": token.text, "pos": token.pos_, "dep": token.dep_}
            i += 1
        arg1 = ""
        rel = ""
        arg2 = ""
        root_idx = (0, 0)
        #encontra o root da extração
        for idx in doc_dict:
            token = doc_dict[idx]["text"]
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "VERB" and dep == "ROOT"):
                rel += token + " "
                root_idx = (idx, idx)
                break
            if (pos == "AUX" and dep == "ROOT"):
                rel += token + " "
                root_idx = (idx, idx)
                break

        #aqui encontramos tudo que está relacionado ao root caso ele seja um verbo
        #(auxiliares, advmod, etc)
        #estes, por serem modificadores ou auxiliares do verbo, são adicionados a rel
        i = 0
        while i < len(doc):
            token = doc_dict[i]["text"]
            pos = doc_dict[i]["pos"]
            dep = doc_dict[i]["dep"]

            if (dep == "xcomp" or pos == "VERB"):
                if i == root_idx[0] - 1:
                    rel = token + " " + rel
                    root_idx = (i, root_idx[1])
                    i = 0
                if root_idx[1] + 1 == i:
                    rel += token + " "
                    root_idx = (root_idx[0], i)
                    i = root_idx[1]

            if (dep == "aux" or pos == "AUX"):
                if i == root_idx[0] - 1:
                    rel = token + " " + rel
                    root_idx = (i, root_idx[1])
                    i = 0
                if root_idx[1] + 1 == i:
                    rel += token + " "
                    root_idx = (root_idx[0], i)
                    i = root_idx[1]
            if (dep == "advmod" or pos == "ADV"):
                if i == root_idx[0] - 1:
                    rel = token + " " + rel
                    root_idx = (i, root_idx[1])
                    i = 0
                if root_idx[1] + 1 == i:
                    rel += token + " "
                    root_idx = (root_idx[0], i)
                    i = root_idx[1]
            if (dep == "case" or pos == "ADP"):
                if i == root_idx[0] - 1:
                    rel = token + " " + rel
                    root_idx = (i, root_idx[1])
                    i = 0
                if root_idx[1] + 1 == i:
                    rel += token + " "
                    root_idx = (root_idx[0], i)
                    i = root_idx[1]
            if (dep == "aux:pass"):
                if i == root_idx[0] - 1:
                    rel = token + " " + rel
                    root_idx = (i, root_idx[1])
                    i = 0
                if root_idx[1] + 1 == i:
                    rel += token + " "
                    root_idx = (root_idx[0], i)
                    i = root_idx[1]
            if (dep == "expl"):
                if i == root_idx[0] - 1:
                    rel = token + " " + rel
                    root_idx = (i, root_idx[1])
                    i = 0
                if root_idx[1] + 1 == i:
                    rel += token + " "
                    root_idx = (root_idx[0], i)
                    i = root_idx[1]

            i += 1

        #aqui separamos arg1 e arg2 a partir do root
        for idx in doc_dict:
            token = doc_dict[idx]["text"]
            if idx < root_idx[0]:
                arg1 += token + " "
            if idx > root_idx[1]:
                arg2 += token + " "
        return arg1, rel, arg2


class Translators:
    def __init__(self, google: bool):
        model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        if not google:
            self.pipe = pipeline("translation", model=model_name, device=0)
        else:
            self.google_translator = GoogleTranslator(source="en", target="pt")

    def batch_google(self, txt):
        txt = self.google_translator.translate(txt)
        return txt

    def mt(self, text):
        trad_text = self.pipe(text, max_length=1000)[0]["translation_text"]
        return trad_text

    def batch_mt(self, dataset):
        if len(dataset[0]) == 1 and len(dataset[1]) == 1:
            trad = self.mt(dataset[0][0])
            ext = self.mt(dataset[1][0])
            trad = trad
            ext = ext
        else:
            trad = self.pipe(dataset[0])
            ext = self.pipe(dataset[1])
            trad = [t["translation_text"] for t in trad]
            ext = [t["translation_text"] for t in ext]
        return trad, ext


class TranslateDataset:
    def __init__(self, dataset_dir: str,
                 dataset_name: str,
                 out_path: str,
                 batch_size: int,
                 google: bool,
                 debug: bool = False
                 ):
        self.batch_size = batch_size
        self.google = google
        self.debug = debug
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.out_path = out_path
        self.translators = Translators(google)

    def debugging(self, sentence,  ext, raw_sent, raw_ext):
        arg0_trad, rel_trad, arg1_trad = ArgsRel().get_args_rel(ext)
        print("\nDebugging")
        print(f"sent: {sentence}")
        print(f"raw_sent: {raw_sent}")
        print(f"ext: {ext}")
        print(f"raw_ext: {raw_ext}")
        print(f"arg0: {arg0_trad}")
        print(f"rel: {rel_trad}")
        print(f"arg1: {arg1_trad}\n")

    def save_dict(self, data_dict):
        path = self.out_path+"/saida_match"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(self.out_path+"/saida_match/json_dump.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(data_dict))

    def save_translate(self, data):
        path = self.out_path+"/translate"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(self.out_path+"/translate/translate.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(data))

    def load_dataset(self):
        # estrutura o dataset em um dicionario
        with open(f"{self.out_path}/conll2bioes_output/{self.dataset_name.replace('.conll', '.txt')}",
                  "r", encoding="utf-8") as f:
            data = f.read()
        data = data.split("\n\t")
        data = [ext.split("\n") for ext in data]
        if self.debug:
            data = data[:32]
        for ext in data:
            for i in range(len(ext)):
                ext[i] = ext[i].split("\t")

        dataset = []
        sents = []
        exts = []
        for ext in tqdm(data, desc="Carregando dataset"):
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
            ext = arg0 + rel + arg1
            sents.append(sentence)
            exts.append(ext)
        dataset.append(sents)
        dataset.append(exts)
        return dataset

    def translate_google(self, cache_dir: str, dataset: list = None):
        cache = Cache(cache_dir)
        if dataset is None:
            dataset = self.load_carb()

        #traduz dataset
        all_sent = []
        all_ext = []
        raw_sent = []
        raw_ext = []
        for i in tqdm(range(len(dataset[0])), desc=f"Traduzindo dataset"):
            if dataset[0][i] in cache:
                sent = cache[dataset[0][i]]
            else:
                sent = self.translators.batch_google(dataset[0][i])
                cache[dataset[0][i]] = sent
            if dataset[1][i] in cache:
                ext = cache[dataset[1][i]]
            else:
                ext = self.translators.batch_google(dataset[1][i])
                cache[dataset[1][i]] = ext

            all_sent.append(sent)
            all_ext.append(ext)
            raw_sent.append(dataset[0][i])
            raw_ext.append(dataset[1][i])

        cache.clear()
        cache.close()
        trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
        self.save_translate(trans_dict)

    def translate_mt(self):
        batch_size = self.batch_size
        dataset = self.load_dataset()
        # batching
        dataloader = []
        for i in tqdm(range(0, len(dataset[0]), batch_size), desc="dataloader"):
            batch = [dataset[0][i:i + batch_size], dataset[1][i:i + batch_size]]
            dataloader.append(batch)

        # traduz dataset
        all_sent = []
        all_ext = []
        raw_sent = []
        raw_ext = []

        for batch in tqdm(dataloader, desc=f"Traduzindo dataset com batching de {batch_size}"):
            sent, ext = self.translators.batch_mt(batch)
            all_sent += [sent]
            all_ext += [ext]
            raw_sent += batch[0]
            raw_ext += batch[1]

        trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
        self.save_translate(trans_dict)

    def create_dict(self):
        argsRel_eng = ArgsRel()
        with open(self.out_path + "/translate/translate.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        all_sent = data["sent"]
        all_ext = data["ext"]
        raw_sent = data["raw_sent"]
        raw_ext = data["raw_ext"]
        if self.debug:
            for sent, ext, rs, re in zip(all_sent, all_ext, raw_sent, raw_ext):
                if not self.google:
                    self.debugging(sent, ext, rs, re)
                else:
                    self.debugging(sent, ext, rs, re)
        data_dict = {}
        #identifica elementos da tripla traduzida e armazena em um dicionario
        counter = 0
        if not self.google:
            for sample in tqdm(zip(all_sent, all_ext), desc="Alinhando extrações", total=len(all_sent)):
                for sent, ext in zip(sample[0], sample[1]):
                    arg0_trad, rel_trad, arg1_trad = argsRel_eng.get_args_rel(ext)
                    data_dict[str(counter)] = {"ID": counter,
                                               "sent": sent,
                                               "ext": [{"arg1": arg0_trad,
                                                        "rel": rel_trad,
                                                        "arg2": arg1_trad}]}
                    counter += 1
        if self.google:
            for sample in tqdm(zip(all_sent, all_ext), desc="Alinhando extrações", total=len(all_sent)):
                arg0_trad, rel_trad, arg1_trad = argsRel_eng.get_args_rel(sample[1])
                data_dict[str(counter)] = {"ID": counter,
                                           "sent": transform_portuguese_contractions(sample[0]),
                                           "ext": [{"arg1": transform_portuguese_contractions(arg0_trad),
                                                    "rel": transform_portuguese_contractions(rel_trad),
                                                    "arg2": transform_portuguese_contractions(arg1_trad)}]}
                counter += 1

        #salva dicionario
        self.save_dict(data_dict)



def run(batch_size: int,
        dataset_dir: str,
        dataset_name: str,
        test_size: float,
        dev_size: float,
        translated: bool,
        debug: bool = False,
        use_google: bool = True,
        sequential: bool = True,
        cache_dir: str = "cache"
        ):
    converted = True
    OUT_NAME = dataset_name.replace(".conll", "")
    INPUT_PATH = ""

    path = "outputs"+"/"+OUT_NAME
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    json_dir = path+"/saida_match"
    pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)

    if use_google or debug:
        batch_size = 1
    trans_eng = TranslateDataset(dataset_dir, dataset_name, path, debug=debug, batch_size=batch_size, google=use_google)
    if translated:
        pass
    else:
        if use_google:
            LoadDataset(dataset_dir, dataset_name, path)
            print("Traduzindo com Google")
            trans_eng.translate_google(cache_dir=cache_dir)
        else:
            LoadDataset(dataset_dir, dataset_name, path)
            print("Traduzindo com MarianMTModel")
            trans_eng.translate_mt()
    trans_eng.create_dict()
    criar_conll(OUT_NAME, INPUT_PATH, test_size, dev_size, converted=converted, sequential=sequential)
