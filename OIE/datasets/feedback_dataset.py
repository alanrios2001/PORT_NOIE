import pathlib
import re
import os
from final import matcher
import spacy

class FeedBackDataset:
    def __init__(self):
        self.dir = "feedback"
        self.data_dict = {}
        self.matcher = matcher.OIE_Match()
        self.nlp = spacy.load("pt_core_news_lg")
        path = pathlib.Path(self.dir)
        path.mkdir(parents=True, exist_ok=True)


    def main(self, sentence, arg0, rel, arg1):
        print('sentence:', sentence)
        print('arg0:', arg0)
        print('rel:', rel)
        print('arg1:', arg1)

        dataset = self.open_dataset()
        self.extruct_dataset(dataset)
        new_data = self.extruct_new_data(sentence, arg0, rel, arg1)
        compare = self.compare_data(new_data)
        if not compare:
            self.save_data()

    def save_data(self):
        file_text = ""
        before_tag = '\tXX\t-\t-\t-\t-\t-\t*\t'
        dict = self.data_dict.keys()
        dict = list(dict)[-1]
        for key in [dict]:
            match = self.matcher.match(key, self.data_dict[key][0], self.data_dict[key][1], self.data_dict[key][2])
            if match[-1]:
                for i,token in enumerate([token.text for token in self.nlp(key)]):
                    if i in range(match[0][0], match[0][1]+1):
                        if len(range(match[0][0], match[0][1]+1)) == 1:
                            file_text += token + before_tag + "S-ARG0" + "\n"
                        elif i == match[0][0]:
                            file_text += token + before_tag + "B-ARG0" + "\n"
                        elif i > match[0][0] and i<match[0][1]:
                            file_text += token + before_tag + "I-ARG0" + "\n"
                        elif i == match[0][1]:
                            file_text += token + before_tag + "E-ARG0" + "\n"

                    elif i in range(match[1][0], match[1][1]+1):
                        if len(range(match[1][0], match[1][1] + 1)) == 1:
                            file_text += token + before_tag + "S-V" + "\n"
                        elif i == match[1][0]:
                            file_text += token + before_tag + "B-V" + "\n"
                        elif i > match[1][0] and i < match[1][1]:
                            file_text += token + before_tag + "I-V" + "\n"
                        elif i == match[1][1]:
                            file_text += token + before_tag + "E-V" + "\n"

                    elif i in range(match[2][0], match[2][1]+1):
                        if len(range(match[2][0], match[2][1] + 1)) == 1:
                            file_text += token + before_tag + "S-ARG1" + "\n"
                        elif i == match[2][0]:
                            file_text += token + before_tag + "B-ARG1" + "\n"
                        elif i > match[2][0] and i < match[2][1]:
                            file_text += token + before_tag + "I-ARG1" + "\n"
                        elif i == match[2][1]:
                            file_text += token + before_tag + "E-ARG1" + "\n"
                    else:
                        file_text += token + before_tag + "O" + "\n"
                file_text += "\n"
        with open(self.dir + "/fb_dataset.txt", "a", encoding="utf-8") as f:
            try:
                txt_f = f.read()
            except:
                txt_f = ""
            txt_f += file_text
            f.write(txt_f)



    def compare_data(self, new_data):
        key = list(new_data.keys())[0]
        try:
            if new_data[key] == self.data_dict[key]:
                print("dado já existente")
                return True
            else:
                print("dado já existente, mas diferente")
                self.data_dict[key].append(new_data[key])
                return False
        except:
            print("adicionando novo dado")
            self.data_dict.update(new_data)
            return False


    def extruct_new_data(self, sentence, arg0, rel, arg1):
        new_data = {}
        new_data[sentence] = [arg0, rel, arg1]
        return new_data

    def open_dataset(self):
        try:
            with open(self.dir + "/fb_dataset.txt", "r", encoding="utf-8") as f:
                return f.read()
        except:
            with open(self.dir + "/fb_dataset.txt", "a", encoding="utf-8") as f:
                return ''

    def extruct_dataset(self, dataset):
        dataset = dataset.split("\n\n")
        for line in dataset:
            arg0 = ""
            rel = ""
            arg1 = ""
            sent = ""
            line = line.split("\n")
            for element in line:
                sent += element.split("\t")[0] + " "
                if "ARG0" in element:
                    arg0 += element.split("\t")[0] + " "
                elif "V" in element:
                    rel += element.split("\t")[0] + " "
                elif "ARG1" in element:
                    arg1 += element.split("\t")[0] + " "
            sent = sent[0:-1]
            arg0 = arg0[0:-1]
            rel = rel[0:-1]
            arg1 = arg1[0:-1]
            self.data_dict[sent] = [arg0, rel, arg1]


#fb = FeedBackDataset()
#fb.main("O Brasil é um país tropical", "O Brasil", "é", "um país tropical")
