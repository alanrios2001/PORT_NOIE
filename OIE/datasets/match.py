import os
import json
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm


class OIE_Match:
    def __init__(self, output_name: str, path_dir):
        self.path_dir = path_dir
        self.output_name = output_name
        self.valid = {}
        self.invalid = {}
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            print("Baixando pt_core_news_lg")
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    def validate_ext(self, sequential: bool = True):

        json_dir = self.path_dir + "/json_dump.json"
        with open(json_dir, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in tqdm(range(len(data)), desc="Carregando dados"):

            key = str(key)
            sentence = self.nlp(data[key]["sent"].lower())
            ext = data[key]["ext"][0]

            arg1 = ext["arg1"].lower().split(" ")
            rel = ext["rel"].lower().split(" ")
            arg2 = ext["arg2"].lower().split(" ")

            arg1_str = list(filter(None, arg1))
            rel_str = list(filter(None, rel))
            arg2_str = list(filter(None, arg2))

            arg1 = [self.nlp.make_doc(text) for text in arg1_str]
            rel = [self.nlp.make_doc(text) for text in rel_str]
            arg2 = [self.nlp.make_doc(text) for text in arg2_str]

            # encontrar arg1
            arg1_matcher = Matcher(self.nlp.vocab)

            pattern = []
            for token in arg1:
                pattern.append({"LOWER": token.text})
            if len(pattern) > 0:
                arg1_matcher.add("arg1", [pattern])
            arg1_match = arg1_matcher(sentence)

            # encontrar arg2
            arg2_matcher = Matcher(self.nlp.vocab)

            pattern = []
            for token in arg2:
                pattern.append({"LOWER": token.text})
            if len(pattern) > 0:
                arg2_matcher.add("arg2", [pattern])
            arg2_match = arg2_matcher(sentence)

            # encontrar relações
            rel_matcher = Matcher(self.nlp.vocab)

            pattern = []
            for token in rel:
                pattern.append({"LOWER": token.text})

            if len(pattern) > 0:
                rel_matcher.add("rel", [pattern])
            rel_match = rel_matcher(sentence)

            # select valid extractions
            if len(arg1_match) > 0 and len(rel_match) > 0 and len(arg2_match) > 0:
                if sequential:
                    if arg1_match[0][2] < rel_match[0][2] < arg2_match[0][2]:
                        self.valid[data[key]["sent"]] = {
                            "arg1": arg1_match,
                            "arg2": arg2_match,
                            "rel": rel_match,
                        }
                else:
                    self.valid[data[key]["sent"]] = {
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
                sent = self.nlp(data[key]["sent"])
                tk = [token.text for token in sent]

                self.invalid[data[key]["sent"]] = {
                    "expected": ext,
                    "arg1": (arg1_tuple[0], arg1_tuple[1], tk[arg1_tuple[0]:arg1_tuple[1]]),
                    "rel": (rel_tuple[0], rel_tuple[1], tk[rel_tuple[0]:rel_tuple[1]]),
                    "arg2": (arg2_tuple[0], arg2_tuple[1], tk[arg2_tuple[0]:arg2_tuple[1]]),
                }

        with open(self.path_dir+"/invalid.json", "a", encoding="utf-8") as f:
            json.dump(self.invalid, f, ensure_ascii=False, indent=4)

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
