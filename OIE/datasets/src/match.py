import os
import json
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.symbols import ORTH
from tqdm import tqdm
import re


class OIE_Match:
    def __init__(self, output_name: str, path_dir):
        self.path_dir = path_dir
        self.output_name = output_name
        self.valid = {}
        self.invalid = {}
        self.valid_data = {}
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            print("Baixando pt_core_news_lg")
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    def remove_special_case(self, tokenizer, special_case):
        # Remova a entrada do dicionário special_cases se ela existir
        if special_case in tokenizer.special_cases:
            del tokenizer.special_cases[special_case]
    def validate_ext(self, sequential):

        json_dir = self.path_dir + "/json_dump.json"
        with open(json_dir, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in tqdm(range(len(data)), desc="Carregando dados"):
            cases = []
            key = str(key)
            raw_sent = data[key]["sent"].split(" ")
            raw_sent = [item for item in raw_sent if ((item != "''" and item != "``")and(item != "'" and item != "`"))]
            raw_sent = " ".join(raw_sent)
            raw_sent = re.sub(r'\u200b', '', raw_sent)

            sent1 = self.nlp(raw_sent)
            raw_sent2 = [token.text for token in sent1]
            if raw_sent2[-1] != ".":
                raw_sent2.append(".")
            raw_sent2 = " ".join(raw_sent2)
            raw_sent2 = raw_sent2.replace(" ,", ",")
            raw_sent2 = raw_sent2.replace(" .", ".")
            sent2 = self.nlp(raw_sent2)
            for token in sent2:
                if token.pos_ == "PROPN" and "." in token.text:
                    abreviation = [{"ORTH": str(token.text).lower()}]
                    self.nlp.tokenizer.add_special_case(str(token.text).lower(), abreviation)
                    cases.append(str(token.text).lower())

            sent = raw_sent2.lower()
            sentence = self.nlp(sent)

            ext = data[key]["ext"][0]

            arg1 = ext["arg1"].lower().split(" ")
            rel = ext["rel"].lower().split(" ")
            arg2 = ext["arg2"].lower().split(" ")

            arg1 = " ".join([item for item in arg1 if ((item != "''" and item != "`")and(item != "'" and item != "``"))])
            arg2 = " ".join([item for item in arg2 if ((item != "''" and item != "`")and(item != "'" and item != "``"))])
            rel = " ".join([item for item in rel if ((item != "''" and item != "`")and(item != "'" and item != "``"))])
            arg1 = re.sub(r'\u200b', '', arg1)
            arg2 = re.sub(r'\u200b', '', arg2)
            rel = re.sub(r'\u200b', '', rel)
            if len(arg2) > 0 and arg2[-1] == ".":
                arg2 = arg2[:-1]

            arg1 = self.nlp(arg1.lower())
            rel = self.nlp(rel.lower())
            arg2 = self.nlp(arg2.lower())

            # encontrar arg1
            arg1_matcher = PhraseMatcher(self.nlp.vocab)
            arg1_matcher.add("arg1", [arg1])
            arg1_match = arg1_matcher(sentence)
            #print("arg1_match")
            #print(arg1)
            #print(arg1_match)

            # encontrar arg2
            arg2_matcher = PhraseMatcher(self.nlp.vocab)
            arg2_matcher.add("arg2", [arg2])
            arg2_match = arg2_matcher(sentence)
            #print("arg2_match")
            #print(arg2)
            #print(arg2_match)

            # encontrar relações
            rel_matcher = PhraseMatcher(self.nlp.vocab)
            rel_matcher.add("rel", [rel])
            rel_match = rel_matcher(sentence)
            #print("rel_match")
            #print(rel)
            #print(rel_match)

            # select valid extractions
            if len(arg1_match) > 0 and len(rel_match) > 0 and len(arg2_match) > 0:
                if sequential:
                    if arg1_match[0][2] < rel_match[0][2] < arg2_match[0][2]:
                        self.valid[raw_sent2] = {
                            "arg1": arg1_match,
                            "arg2": arg2_match,
                            "rel": rel_match,
                        }
                        sent = self.nlp(raw_sent2)
                        tk = [token.text for token in sent]
                        self.valid_data[raw_sent2] = {
                            "arg1": (tk[arg1_match[0][1]:arg1_match[0][2]]),
                            "rel": (tk[rel_match[0][1]:rel_match[0][2]]),
                            "arg2": (tk[arg2_match[0][1]:arg2_match[0][2]]),
                        }
                elif (arg1_match[0][2] < rel_match[0][2] < arg2_match[0][2]) == False:
                    self.valid[raw_sent2] = {
                        "arg1": arg1_match,
                        "arg2": arg2_match,
                        "rel": rel_match,
                    }

            else:
                #collect invalid extractions
                try:
                    arg1_tuple = (arg1_match[0][1], arg1_match[0][2])
                except:
                    arg1_tuple = (0, 0)
                try:
                    arg2_tuple = (arg2_match[0][1], arg2_match[0][2])
                except:
                    arg2_tuple = (0, 0)
                try:
                    rel_tuple = (rel_match[0][1], rel_match[0][2])
                except:
                    rel_tuple = (0, 0)
                sent = self.nlp(raw_sent2)
                tk = [token.text for token in sent]

                try:
                    self.invalid[raw_sent2] = {
                        "ID": data[key]["ID"],
                        "expected": ext,
                        "arg1": (arg1_tuple[0], arg1_tuple[1], tk[arg1_tuple[0]:arg1_tuple[1]]),
                        "rel": (rel_tuple[0], rel_tuple[1], tk[rel_tuple[0]:rel_tuple[1]]),
                        "arg2": (arg2_tuple[0], arg2_tuple[1], tk[arg2_tuple[0]:arg2_tuple[1]]),
                    }
                except:
                    self.invalid[raw_sent2] = {
                        "ID": key,
                        "expected": ext,
                        "arg1": (arg1_tuple[0], arg1_tuple[1], tk[arg1_tuple[0]:arg1_tuple[1]]),
                        "rel": (rel_tuple[0], rel_tuple[1], tk[rel_tuple[0]:rel_tuple[1]]),
                        "arg2": (arg2_tuple[0], arg2_tuple[1], tk[arg2_tuple[0]:arg2_tuple[1]]),
                    }
        for case in cases:
            self.nlp.tokenizer.remove_special_case(self.nlp, case)

        with open(self.path_dir+"/invalid.json", "a", encoding="utf-8") as f:
            json.dump(self.invalid, f, ensure_ascii=False, indent=4)
        with open(self.path_dir+"/gold_valid.json", "a", encoding="utf-8") as f:
            json.dump(self.valid_data, f)

        print("initial samples: ", len(data), "|| valid samples: ", len(self.valid))

    def create_corpus(self):
        with open(f"{self.path_dir}/{self.output_name}_corpus.txt", "a", encoding="utf-8") as file:
            for sent in tqdm(self.valid, desc="Criando conll"):
                sentence = self.nlp(sent)
                sent_tokens = [token.text for token in sentence]
                arg1_spans = []
                arg2_spans = []
                rel_spans = []
                for match in self.valid[sent]["arg1"]:
                    arg1_spans.append((match[1], match[2]))
                for match in self.valid[sent]["arg2"]:
                    arg2_spans.append((match[1], match[2]))
                for match in self.valid[sent]["rel"]:
                    rel_spans.append((match[1], match[2]))

                label_lines = ""
                for i in range(len(sent_tokens)):
                    if i >= arg1_spans[0][0] and i < arg1_spans[0][1]:
                        if i == arg1_spans[0][0] and arg1_spans[0][1] - arg1_spans[0][0] == 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tS-ARG0\t-"
                            label_lines += line + "\n"
                        elif i == arg1_spans[0][0] and arg1_spans[0][1] - arg1_spans[0][0] > 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tB-ARG0\t-"
                            label_lines += line + "\n"
                        elif i > arg1_spans[0][0] and i < arg1_spans[0][1] - 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tI-ARG0\t-"
                            label_lines += line + "\n"
                        elif i == arg1_spans[0][1] - 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tE-ARG0\t-"
                            label_lines += line + "\n"

                    elif i >= arg2_spans[0][0] and i < arg2_spans[0][1]:
                        if i == arg2_spans[0][0] and arg2_spans[0][1] - arg2_spans[0][0] == 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tS-ARG1\t-"
                            label_lines += line + "\n"
                        elif i == arg2_spans[0][0] and arg2_spans[0][1] - arg2_spans[0][0] > 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tB-ARG1\t-"
                            label_lines += line + "\n"
                        elif i > arg2_spans[0][0] and i < arg2_spans[0][1] - 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tI-ARG1\t-"
                            label_lines += line + "\n"
                        elif i == arg2_spans[0][1] - 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tE-ARG1\t-"
                            label_lines += line + "\n"

                    elif i >= rel_spans[0][0] and i < rel_spans[0][1]:
                        if i == rel_spans[0][0] and rel_spans[0][1] - rel_spans[0][0] == 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tS-V\t-"
                            label_lines += line + "\n"
                        elif i == rel_spans[0][0] and rel_spans[0][1] - rel_spans[0][0] > 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tB-V\t-"
                            label_lines += line + "\n"
                        elif i > rel_spans[0][0] and i < rel_spans[0][1] - 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tI-V\t-"
                            label_lines += line + "\n"
                        elif i == rel_spans[0][1] - 1:
                            line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tE-V\t-"
                            label_lines += line + "\n"

                    elif sent_tokens[i] != " ":
                        line = f"{sent_tokens[i]}\tXX\t-\t-\t-\t-\t-\t*\tO\t-"
                        label_lines += line + "\n"

                    if i == len(sent_tokens) - 1:
                        label_lines += "\n"

                file.writelines(label_lines)

    def run(self, sequential: bool = True):
        self.validate_ext(sequential=sequential)
        self.create_corpus()
