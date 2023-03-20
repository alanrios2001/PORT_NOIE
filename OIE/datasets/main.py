from src.create_txt_csv import Convert
from src.match import OIE_Match
import typer
import pathlib
from src.pos_tag import PosTag
from src.train_test_dev import train_dev_test
from src.merge_datasets import Merge
import json

app = typer.Typer()


def save_dict(data_dict, out_path):
    path = f"{out_path}/saida_match"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with open(out_path + "/saida_match/json_dump.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(data_dict))


@app.command()
def criar_conll(out_name: str,
                txt_path: str,
                test_size: float,
                dev_size: float,
                input_path: str = None,
                converted: bool = False,
                sequential: bool = True,
                ):
    if input_path is None:
        input_path = ""
        path = f"outputs/{out_name}"
        path_saida_match = f"outputs/{out_name}" + "/saida_match"
    else:
        path = f"{input_path}/outputs/{out_name}"
        path_saida_match = f"{input_path}/outputs/{out_name}" + "/saida_match"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_saida_match).mkdir(parents=True, exist_ok=True)

    if not converted:
        Convert(txt_path, path, out_name)

    # selecionar e anotar sentenças validas
    oie_match = OIE_Match(out_name, path_saida_match)
    oie_match.run(sequential=sequential)

    print(path_saida_match)
    # POS tagging

    PosTag(f"{path_saida_match}\{out_name}_corpus.txt", path).run(f"{out_name}_corpus.txt")


    # train, dev, test
    if test_size > 0 and dev_size > 0:
        train_dev_test(test_size, dev_size, out_name, in_path=path+"/saida_pos_tag", out_path=path+"/saida_pos_tag")

@app.command()
def merge():
    fine_tune = ["other_corpus/outputs/saida_pos_tag/gamalho.txt",
             #"other_corpus/outputs/saida_pos_tag/pragmatic_ceten.txt",
             #"other_corpus/outputs/saida_pos_tag/pragmatic_wiki.txt",
             "other_corpus/outputs/saida_pos_tag/pud_200.txt"]

    ls_train = ["outputs/splits/ls_dev_test.txt",
          "outputs/splits/ls_dev_train.txt",
          "outputs/ls_train/saida_pos_tag/ls_train_corpus.txt"]

    ls_test = ["outputs/splits/ls_dev_dev.txt",
          "outputs/ls_test/saida_pos_tag/ls_test_corpus.txt"]

    ptoie = ["outputs/PTOIE/saida_pos_tag/PTOIE_dev_test.txt", "outputs/PTOIE/saida_pos_tag/PTOIE_test.txt"]

    OUTPUT_NAME = "ptoie_test"
    Merge(ptoie, OUTPUT_NAME)

@app.command()
def split():
    TEST_SIZE = 0.5
    DEV_SIZE = 0.0
    OUTPUT_NAME = "PTOIE_dev"
    IN_PATH = "outputs/PTOIE/saida_pos_tag"
    OUT_PATH = "outputs/PTOIE/saida_pos_tag"
    train_dev_test(TEST_SIZE, DEV_SIZE, OUTPUT_NAME, IN_PATH, OUT_PATH)


def load_s2(dataset_path: str):
    data_path = dataset_path

    out_path = f"other_corpus/outputs/s2_DS{data_path.split('/')[2].replace('.tsv', '')}"
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        file = f.read().split("<e>")
        data_dict = {}
        counter = 0
        for line in file:
            sent = ""
            l = ""
            arg0 = ""
            rel = ""
            arg1 = ""
            line = line.replace("\n", "").split(".\t")
            #print(line)
            if line != ['']:
                try:
                    sent = line[0].split("<r>")[1]+"."
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
                data_dict[str(counter)] = {
                    "ID": counter,
                    "sent": sent,
                    "ext": [{"arg1": arg0,
                             "rel": rel,
                             "arg2": arg1}]
                }
                counter += 1
        save_dict(data_dict, out_path)

        OUT_NAME = f"s2_DS{data_path.split('/')[2].replace('.tsv', '')}"
        TXT_PATH = ""
        TEST_SIZE = 0
        DEV_SIZE = 0
        CONVERTED = True
        SEQUENTIAL = True
        criar_conll(OUT_NAME,
                    TXT_PATH,
                    TEST_SIZE,
                    DEV_SIZE,
                    input_path=f"other_corpus",
                    converted=CONVERTED,
                    sequential=SEQUENTIAL
                    )


@app.command()
def build(dataset: str):
    if dataset == "s2_DS":
        datasets = [#"other_corpus/s2_DS/train.tsv",
                    "other_corpus/s2_DS/valid.tsv"]
        for dataset in datasets:
            load_s2(dataset)




if __name__ == "__main__":
    app()
