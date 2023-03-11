from OIE.datasets.main import criar_conll
import json
import pathlib


def remove_contractions(txt):
    txt = txt.replace("de o", "do")
    txt = txt.replace("de a", "da")
    txt = txt.replace("de os", "dos")
    txt = txt.replace("de as", "das")
    txt = txt.replace("em o", "no")
    txt = txt.replace("em a", "na")
    txt = txt.replace("em os", "nos")
    txt = txt.replace("em as", "nas")

    return txt

#@app.command()
def main(dir: str , dataset: str):

    path = pathlib.Path(f"{dir}/outputs/{dataset.replace('.txt', '')}/saida_match")
    path.mkdir(parents=True, exist_ok=True)

    total = 0
    data_dict = {}
    final_dict = {}
    with open(f"{dir}/{dataset}", "r", encoding="utf-8") as file:
        with open(f"{dir}/outputs/{dataset.replace('.txt', '')}/saida_match/json_dump.json", "a", encoding="utf-8") as file2:
            lines = file.read()
            lines = lines.split("\n")
            sent = ""
            exts = []
            for line in lines:
                if line.count("\t") == 1:
                    if sent == "":
                        sent = remove_contractions(line.split("\t")[1])
                        id = line.split("\t")[0]
                    else:
                        if exts != []:
                            data_dict[sent] = exts
                            sent = line.split("\t")[1]
                            exts = []
                elif line.count("\t") == 5:
                    ext = line.split("\t")
                    if ext[4] == "1":
                        exts.append(ext[0:3])
                        exts.append(id)

            for key in data_dict:
                for ext in data_dict[key]:
                    if type(ext) == list:
                        final_dict[total] = {"ID": data_dict[key][-1], "sent": key, "ext": [{"arg1": ext[0], "rel": ext[1], "arg2": ext[2]}]}
                        total += 1
            file2.write(json.dumps(final_dict))

    criar_conll(dataset.replace(".txt", ""), "other_corpus/", 0.0, 0.0, converted=True, sequential=False)


if __name__ == "__main__":
    dir = "other_corpus"
    datasets = ["200-silver.txt", "100-dev.txt"]
    for dataset in datasets:
        main(dir ,dataset)
