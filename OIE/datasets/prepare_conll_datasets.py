from src.conll2bioes import Conversor
from src.pos_tag import PosTag
import pathlib

datasets = ["gamalho.conll", "pragmatic_ceten.conll", "pragmatic_wiki.conll"]

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
    PosTag(f"other_corpus/outputs/conll2bioes_output/{dataset.replace('.conll', '')}.txt", path="other_corpus/outputs/").run(f"{dataset.replace('.conll', '')}.txt")
    conv.train_dev_test(0.0, 0.0)
    total += conv.total_len
print("total: ", total)