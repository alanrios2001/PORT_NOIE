from src.conll2bioes import Conversor
from src.pos_tag import PosTag
import pathlib
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions, clean_extraction
import json
from src.match import OIE_Match

datasets = ["pud_100.conll","pud_200.conll","gamalho.conll", "pragmatic_ceten.conll", "pragmatic_wiki.conll"]

path = pathlib.Path("other_corpus/outputs/mod")
path.mkdir(parents=True, exist_ok=True)

#seleciona apenas extrações corretas(com arg0, v e arg1)
total = 0
for dataset in datasets:
    with open("other_corpus/" + dataset, "r", encoding="utf-8") as file:
        with open("other_corpus/outputs/mod/" + dataset, "a", encoding="utf-8") as file2:
            lines = file.read()
            lines = lines.split("\n\n")
            for line in lines:
                if "ARG0" in line and "ARG1" in line and "V" in line:
                    file2.write(line+"\n\n")



    #apos selecionar extrações corretas, converte para BIOES
    conv = Conversor("other_corpus/outputs/mod/", dataset, out_dir="other_corpus/outputs/")

    with open("other_corpus/outputs/conll2bioes_output/" + dataset.replace("conll", "txt"), "r",
              encoding="utf-8") as file:
        lines = file.read()
        lines = lines.split("\n\n")
        data_dict = {}
        counter = 0
        for ext in lines:
            if ext != "":
                sent = ""
                arg0 = ""
                arg1 = ""
                rel = ""
                unit = ext.split("\n")
                for i in unit:
                    sent += i.split("\t")[0] + " "
                    if "ARG0" in i:
                        arg0 += i.split("\t")[0] + " "
                    elif "ARG1" in i:
                        arg1 += i.split("\t")[0] + " "
                    elif "V" in i:
                        rel += i.split("\t")[0] + " "
                sent = clean_extraction(transform_portuguese_contractions(sent))
                arg0 = clean_extraction(transform_portuguese_contractions(arg0))
                arg1 = clean_extraction(transform_portuguese_contractions(arg1))
                rel = clean_extraction(transform_portuguese_contractions(rel))
                data_dict[counter] = {"ID": counter,
                                      "sent": sent,
                                      "ext": [{"arg1": arg0, "rel": rel, "arg2": arg1}]}
                counter += 1
    path = pathlib.Path(f"other_corpus/outputs/saida_match/{dataset.replace('.conll', '')}")
    path.mkdir(parents=True, exist_ok=True)
    with open(f"other_corpus/outputs/saida_match/{dataset.replace('.conll', '')}/" + "json_dump.json", "a", encoding="utf-8") as file2:
        file2.write(json.dumps(data_dict))
    matcher = OIE_Match(output_name=dataset.replace('.conll', ''), path_dir=f"other_corpus/outputs/saida_match/{dataset.replace('.conll', '')}/")
    matcher.run(sequential=True)
    PosTag(f"other_corpus/outputs/saida_match/{dataset.replace('.conll', '')}/{dataset.replace('.conll', '')}_corpus.txt", path="other_corpus/outputs/").run(f"{dataset.replace('.conll', '')}")
    #.train_dev_test(0.0, 0.0)
    total += conv.total_len
print("total: ", total)