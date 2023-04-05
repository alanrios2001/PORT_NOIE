import json
import pathlib


class Convert:
    def __init__(self, txt_path, path_dir, name="_"):
        self.name = name
        self.txt_path = txt_path
        self.dictio = {}
        with open(txt_path, "r", encoding='utf-8') as file:
            read_file = file.read()
        # criando lista dividida por \n\n no txt
        self.splited_pre = read_file.strip().split('\n\n')
        self.splited = []
        # separando cada sentença por listas
        for obj in self.splited_pre:
            self.splited.append(obj.split("\n"))

        def transform_in_dict():
            i = 0
            lista = []
            dic = {}

            for obj in self.splited:
                # tratando strings vazias nas listas
                obj = list(filter(lambda x: x != "", obj))

                # tratando sentenças
                sent = obj[2].split(":")

                if len(sent) == 2:
                    sent = sent[1]
                elif len(sent) > 2:
                    sentença = ""
                    for j in range(len(sent)):
                        if j != 0:
                            sentença += sent[j]
                    sent = sentença

                dic["Id"] = i
                dic["sent"] = sent
                lista.append(sent)

                # tratando extrações
                splited = obj[3].split("|||")
                try:
                    dic["ext"] = [{"arg1": splited[0], "rel": splited[1], "arg2": splited[2]}]
                except:
                    dic["ext"] = [{"arg1": "", "rel": "", "arg2": ""}]
                self.dictio[i] = dic
                i = i + 1
                dic = {}

            json_str = json.dumps(self.dictio)
            with open(path_dir+"/saida_match/json_dump.json", "a", encoding ="utf-8") as file:
                file.write(json_str)

            with open(path_dir+"/saida_match/gold_valid.json", "a", encoding ="utf-8") as file:
                file.write(json.dumps(self.dictio, indent=4, ensure_ascii=False))

        transform_in_dict()
