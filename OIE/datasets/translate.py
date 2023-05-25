from src.conll2bioes import Conversor
import os
import spacy
from tqdm.auto import tqdm
from main import criar_conll
import typer
from deep_translator import GoogleTranslator
import json
import pathlib
from diskcache import Cache
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions, clean_extraction
from OIE.final.matcher import OIE_Match
import openai

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
        self.provavel_rel = []
        self.alinhamentos = []
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    def root_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "VERB" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("VERB")
                return (idx, idx)
        root_idx = self.verb_parse(doc_dict, root_idx)
        return root_idx

    def verb_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "VERB" and (dep == "xcomp" or dep == "acl" or dep == "acl:relacl")) and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("VERB")
                return (idx, idx)

        root_idx = self.aux_parse(doc_dict, root_idx)
        return root_idx



    def aux_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "AUX" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("AUX")
                return (idx, idx)

        root_idx = self.aux_parse2(doc_dict, root_idx)
        return root_idx


    def aux_parse2(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "AUX" and dep == "cop") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("AUX")
                return (idx, idx)
        root_idx = self.noun_parse(doc_dict, root_idx)
        return root_idx

    def noun_parse(self, doc_dict, root_idx):
        #encontra centro da extração pelo root
        for idx in doc_dict:
            pos = doc_dict[idx]["pos"]
            dep = doc_dict[idx]["dep"]
            if (pos == "NOUN" and dep == "ROOT") and (idx != 0 and idx != len(doc_dict) - 1):#restringe primeiro e último caracter da frase inteira
                self.provavel_rel.append("NOUN")
                return (idx, idx)
        return root_idx


    def get_args_rel(self, ext):
        self.alinhamentos = []
        doc = self.nlp(ext)
        doc_dict = {}
        i = 0
        for token in doc:
            doc_dict[i] = {"text": token.text, "pos": token.pos_, "dep": token.dep_}
            i += 1
        root_idx = (None, None)
        self.provavel_rel = []
        root_idx = self.root_parse(doc_dict, root_idx)

        if len(self.provavel_rel)>0 and self.provavel_rel[0] == "VERB":
            if root_idx[0]-1 != 0:
                if doc_dict[root_idx[0]-1]["pos"] == "AUX":
                    root_idx = (root_idx[0]-1, root_idx[1])
        #verificando elementos que compoem a rel depois do centro
        if root_idx != (None, None):
            for j in range(root_idx[1]+1, len(doc_dict)):
                pos = doc_dict[j]["pos"]
                self.provavel_rel.append(pos)

        adp_idxs = [i for i in range(len(self.provavel_rel[0:-1])) if self.provavel_rel[i] == "ADP"]
        adp_idxs.append(0)

        for idx in adp_idxs:
            arg1 = ""
            rel = ""
            arg2 = ""
            if root_idx != (None, None):
                new_root_idx = (root_idx[0],root_idx[1]+idx)
                j = new_root_idx[0]
                while j <= new_root_idx[1]:
                    rel += doc_dict[j]["text"] + " "
                    j += 1

                for idx in doc_dict:
                    token = doc_dict[idx]["text"]
                    if idx < new_root_idx[0]:
                        arg1 += token + " "
                    if idx > new_root_idx[1]:
                        arg2 += token + " "

            self.alinhamentos.append((arg1,rel,arg2))


        return self.alinhamentos



class Translators:
    def __init__(self, google: bool):
        if not google:
            openai.api_key = 'YOUR API KEY HERE'
            self.prompt_tradução = "Você é um tradutor muito preciso que faz traduções de textos da lingua inglêsa para a lingua pt-br. " \
                  "Você irá receber dois textos, uma setença e um fato relacionado a essa sentença, siga as regras:" \
                  "1.Em hipótese alguma retorne qualquer mensagem de erro ou aviso, retorne somente o formato de saída que foi pedido" \
                  "2.Você deve traduzir ambas de forma que todas os tokens que fazem parte do fato, estejam presentes na sentença, e em ordem de ocorrencia na sentença." \
                  "3.Caso a sentença e o fato sejam iguais, traduza da mesma maneira ambos, sendo que a tradução de tanto a sentença quanto o fato são iguais" \
                  "4.Caso ocorra qualquer erro, crie um fato seguindo as regras 1" \
                  "5.A entrada ocorrerá da seguinte maneira:" \
                  "SENTENÇA: The dog is walking through the park, he is very happy." \
                  "FATO: The dog is very happy." \
                  "6.A saída deve ser só e somente só, a seguinte:" \
                  "SENTENÇA: O cachorro está andando pelo parque, ele está muito feliz." \
                  "FATO: O cachorro está muito feliz." \


        else:
            self.google_translator = GoogleTranslator(source="en", target="pt")

    def batch_google(self, txt):
        txt = self.google_translator.translate(txt)
        return txt

    def gpt(self, sent, ext):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": self.prompt_tradução},
                {"role": "user", "content": f"SENTENÇA: {sent}"},
                {"role": "user", "content": f"FATO: {ext}"}
            ]
        )
        sentence = response['choices'][0]['message']['content'].split("\n")[0].split(": ")[-1]
        extraction = response['choices'][0]['message']['content'].split("\n")[-1].split(": ")[-1]
        #print("sentence: ", sentence)
        #print("extraction: ", extraction)
        return sentence, extraction


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
        self.matcher = OIE_Match(sequential=True)
        self.argreleng = ArgsRel()

    def debugging(self, sentence,  ext, raw_sent, raw_ext):
        alignments = self.argreleng.get_args_rel(ext)
        for alignment in alignments:
            arg0_trad = alignment[0]
            rel_trad = alignment[1]
            arg1_trad = alignment[2]
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
            open(self.out_path + "/translate/translate.json", "w", encoding="utf-8").close()
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

    def half_translated(self):
        try:
            open(f"{self.out_path}/translate/translate.json", "r", encoding="utf-8")
            return True
        except:
            return False

    def translate_google(self, cache_dir: str):
        cache = Cache(cache_dir)
        dataset = self.load_dataset()


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

    def translate_gpt(self):
        dataset = self.load_dataset()

        # traduz dataset
        all_sent = []
        all_ext = []
        raw_sent = []
        raw_ext = []

        if self.half_translated():
            with open(f"{self.out_path}/translate/translate.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            all_sent = data["sent"]
            all_ext = data["ext"]
            raw_sent = data["raw_sent"]
            raw_ext = data["raw_ext"]
            i = len(all_sent)
        else:
            i = 0

        while i < len(dataset[0]):
            try:
                sent, ext = self.translators.gpt(dataset[0][i], dataset[1][i])

                all_sent.append(sent)
                all_ext.append(ext)
                raw_sent.append(dataset[0][i])
                raw_ext.append(dataset[1][i])
                os.system("cls")
                print(f"{i/len(dataset[0])*100:.2f}% concluído ||| {i}/{len(dataset[0])}")
                trans_dict = {"sent": all_sent, "ext": all_ext, "raw_sent": raw_sent, "raw_ext": raw_ext}
                self.save_translate(trans_dict)
                i+=1
            except:
                print("provavelmente o modelo está sobrecarregado, tentando novamente")


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
        for sample in tqdm(zip(all_sent, all_ext), desc="Alinhando extrações", total=len(all_sent)):
            alignments = argsRel_eng.get_args_rel(transform_portuguese_contractions(sample[1]))
            for ali in alignments:
                arg0_trad, rel_trad, arg1_trad = ali

                if len(alignments) > 1:
                    match = self.matcher.match(transform_portuguese_contractions(sample[0]),
                                               transform_portuguese_contractions(arg0_trad),
                                               transform_portuguese_contractions(rel_trad),
                                               transform_portuguese_contractions(arg1_trad)
                                               )

                    if match[3] == True:
                        data_dict[str(counter)] = {"ID": counter,
                                                   "sent": transform_portuguese_contractions(sample[0]),
                                                   "ext": [{"arg1": transform_portuguese_contractions(arg0_trad),
                                                            "rel": transform_portuguese_contractions(rel_trad),
                                                            "arg2": transform_portuguese_contractions(arg1_trad)}]}
                        counter += 1
                        break



                else:
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
            print("Traduzindo com ChatGPT")
            trans_eng.translate_gpt()
    trans_eng.create_dict()
    criar_conll(OUT_NAME, INPUT_PATH, test_size, dev_size, converted=converted, sequential=sequential)
