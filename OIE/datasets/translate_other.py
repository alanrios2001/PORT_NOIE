import translate
from main import criar_conll

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

def load_largeie():
    sents_list = []
    exts_list = []
    dataset = []
    with open("sents.txt", "r", encoding="utf-8") as f:
        with open("exts.txt", "r", encoding="utf-8") as f2:
            sents = f.read().split("\n")
            exts = f2.read().split("\n")
            for i in range(len(sents)):
                arg0 = ""
                rel = ""
                arg1 = ""
                sent = sents[i]
                ext = exts[i]
                ext = ext.split("<arg1>")
                ext = ext[1].split("</arg1>")
                arg0 = ext[0]
                ext = ext[1].split("<rel>")
                ext = ext[1].split("</rel>")
                rel = ext[0]
                ext = ext[1].split("<arg2>")
                ext = ext[1].split("</arg2>")
                arg1 = ext[0]
                sents_list.append(sent)
                exts_list.append(arg0 + " " + rel + " " + arg1)
            dataset.append(sents_list)
            dataset.append(exts_list)
    return dataset

def run():
    datasets_to_translate = [
        {"dir":"","name": "carb", "load": load_carb(), "out_path": "outputs/carb/", "batch_size": 1, "google": True},
        #{"dir": "", "name": "largeIE", "load": load_largeie(), "out_path": "outputs/largeie/", "batch_size": 1, "google": True},
    ]
    for dataset in datasets_to_translate:
        eng = translate.TranslateDataset(dataset["dir"], dataset["name"], dataset["out_path"], dataset["batch_size"], dataset["google"])
        #eng.translate_google(cache_dir="translated/cache",dataset=dataset["load"])
        eng.create_dict()
        criar_conll(dataset["name"], "", 0.0, 0.0, converted=True, sequential=True)

run()