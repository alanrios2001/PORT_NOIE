import translate
from main import criar_conll
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions, clean_extraction

def load_carb():
    sents = []
    exts = []
    dataset = []
    with open("translated/carb/dev.tsv", "r", encoding="utf-8") as f:
        dev = f.read()
        for line in dev.split("\n"):
            if len(line.split("\t")) == 4:
                sent = line.split("\t")[0]
                rel = line.split("\t")[1]
                arg0 = line.split("\t")[2]
                arg1 = line.split("\t")[3]
                sents.append(sent)
                exts.append(arg0 + " " + rel + " " + arg1)
        f.close()
    with open("translated/carb/test.tsv", "r", encoding="utf-8") as f:
        test = f.read()
        for line in test.split("\n"):
            if len(line.split("\t")) == 4:
                sent = line.split("\t")[0]
                rel = line.split("\t")[1]
                arg0 = line.split("\t")[2]
                arg1 = line.split("\t")[3]
                sents.append(sent)
                exts.append(arg0 + " " + rel + " " + arg1)
        f.close()

    dataset.append(sents)
    dataset.append(exts)
    return dataset

def load_s2_valid():
    sents = []
    exts = []
    dataset = []
    with open("translated/s2/valid.tsv", "r", encoding="utf-8") as f:
        file = f.read().split("<e>")
        for line in file:
            sent = ""
            l = ""
            arg0 = ""
            rel = ""
            arg1 = ""
            line = line.replace("\n", "").split(".\t")
            if line != ['']:
                try:
                    sent = line[0].split("<r>")[1] + "."
                except:
                    pass
                try:
                    arg0 = line[1].split("<a1>")[1].split("</a1>")[0]
                except:
                    pass
                try:
                    rel = line[1].split("<r>")[1].split("</r>")[0]
                except:
                    pass
                try:
                    arg1 = line[1].split("<a2>")[1].split("</a2>")[0]
                except:
                    pass
                try:
                    l = line[1].split("<l>")[1].split("</l>")[0]
                except:
                    pass
            if arg0 and rel and arg1 != "":
                sents.append(sent)
                exts.append(arg0 + " " + rel + " " + arg1)
    dataset.append(sents)
    dataset.append(exts)
    return dataset


def load_s2_train():
    sents = []
    exts = []
    dataset = []
    with open("translated/s2/train.tsv", "r", encoding="utf-8") as f:
        file = f.read().split("<e>")
        for line in file:
            sent = ""
            l = ""
            arg0 = ""
            rel = ""
            arg1 = ""
            line = line.replace("\n", "").split(".\t")
            if line != ['']:
                try:
                    sent = line[0].split("<r>")[1] + "."
                except:
                    pass
                try:
                    arg0 = line[1].split("<a1>")[1].split("</a1>")[0]
                except:
                    pass
                try:
                    rel = line[1].split("<r>")[1].split("</r>")[0]
                except:
                    pass
                try:
                    arg1 = line[1].split("<a2>")[1].split("</a2>")[0]
                except:
                    pass
                try:
                    l = line[1].split("<l>")[1].split("</l>")[0]
                except:
                    pass
            if arg0 and rel and arg1 != "":
                sents.append(sent)
                exts.append(arg0 + " " + rel + " " + arg1)
    dataset.append(sents)
    dataset.append(exts)
    return dataset


def run():
    datasets_to_translate = [
        #{"dir":"","name": "carb", "load": load_carb(), "out_path": "outputs/carb/", "batch_size": 1, "google": False},
        {"dir": "", "name": "s2_alan_train", "load": load_s2_train(), "out_path": "outputs/s2_alan_train/", "batch_size":1, "google": False},
        #{"dir": "", "name": "s2_alan_valid", "load": load_s2_valid(), "out_path": "outputs/s2_alan_valid/", "batch_size":1, "google": False},
    ]
    for dataset in datasets_to_translate:
        eng = translate.TranslateDataset(dataset["dir"], dataset["name"], dataset["out_path"], dataset["batch_size"], dataset["google"])
        eng.translate_gpt(dataset=dataset["load"])
        eng.create_dict()
        criar_conll(dataset["name"], "", 0.0, 0.0, converted=True, sequential=True)

run()