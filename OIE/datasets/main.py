from src.create_txt_csv import Convert
from src.match import OIE_Match
import typer
import pathlib
from src.pos_tag import PosTag
from src.train_test_dev import train_dev_test
from src.merge_datasets import Merge
from validated_splits.contractions import transform_portuguese_contractions, clean_extraction
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
    fine_tune = ["other_corpus/outputs/saida_match/gamalho/gamalho_corpus.txt",
                  "other_corpus/outputs/saida_match/pragmatic_ceten/pragmatic_ceten_corpus.txt",
                  "other_corpus/outputs/saida_match/pragmatic_wiki/pragmatic_wiki_corpus.txt"]

    trad = [
        "validated_splits/normal/TransAlign2/carb_corpus.txt",
        "validated_splits/normal/TransAlign2/ls_train_corpus.txt",
        "validated_splits/normal/TransAlign2/ls_dev_corpus.txt",
        "validated_splits/normal/TransAlign2/ls_test_corpus.txt",
        "validated_splits/normal/TransAlign2/dev_corpus.txt",
        "validated_splits/normal/TransAlign2/s2_TA_train_corpus.txt",
    ]

    OUTPUT_NAME = "TA2"
    Merge(trad, OUTPUT_NAME)

@app.command()
def split():
    TEST_SIZE = 0.1
    DEV_SIZE = 0.0
    OUTPUT_NAME = "TA2"
    IN_PATH = "validated_splits/normal/TransAlign2/"
    OUT_PATH = "outputs/splits"
    train_dev_test(TEST_SIZE, DEV_SIZE, OUTPUT_NAME, IN_PATH, OUT_PATH)

@app.command()
def load_ptoie():
    criar_conll("PTOIE", "PTOIE/PTOIE.txt", 0.1, 0.1)

@app.command()
def load_s2(dataset_path = "other_corpus/s2/valid.tsv"):
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
                    "sent": transform_portuguese_contractions(sent),
                    "ext": [{"arg1": transform_portuguese_contractions(arg0),
                             "rel": transform_portuguese_contractions(rel),
                             "arg2": transform_portuguese_contractions(arg1)}]
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
def build(dataset: str = "s2"):
    if dataset == "s2":
        datasets = ["other_corpus/s2/train.tsv",
                    "other_corpus/s2/valid.tsv"]
        for dataset in datasets:
            load_s2(dataset)

@app.command()
def load_bia():
    dataset_name = "other_corpus/bia.csv"
    valid = []
    invalid = []
    data_dict = {}
    with open(dataset_name, "r", encoding="utf-8") as f:
        dataset = f.read().splitlines()
        #for i in dataset:
            #print(i)
        dataset = [i.split(";")[0:4] for i in dataset]
        print(dataset)



    counter = 0
    for _,i in enumerate(dataset):
        #print(i)
        try:
            i[0] = i[0].replace('"": {"', "")
            i[0] = i[0].replace('"', "")
            i[0] = i[0].replace("'", "")
            i[0] = i[0].replace("\\", "")
            i[1] = i[1].replace('"": {"', "")
            i[1] = i[1].replace('"', "")
            i[1] = i[1].replace("'", "")
            i[1] = i[1].replace("\\", "")
            i[2] = i[2].replace('"": {"', "")
            i[2] = i[2].replace('"', "")
            i[2] = i[2].replace("'", "")
            i[2] = i[2].replace("\\", "")
            i[3] = i[3].replace('"": {"', "")
            i[3] = i[3].replace('"', "")
            i[3] = i[3].replace("'", "")
            i[3] = i[3].replace("\\", "")
            sent = transform_portuguese_contractions(i[0])
            arg0 = transform_portuguese_contractions(i[1])
            rel = transform_portuguese_contractions(i[2])
            arg1 = transform_portuguese_contractions(i[3])
            data_dict[counter] = {"ID": _,"sent": sent, "ext":[{"arg1": arg0, "rel": rel, "arg2": arg1}]}
            print("\n", _)
            print("sent: ", sent)
            print("arg0: ", arg0)
            print("rel: ", rel)
            print("arg1: ", arg1)
            counter += 1
        except:
            print("invalid")
            invalid.append(i)
            pass

    path = pathlib.Path(f"outputs/bia/saida_match/")
    path.mkdir(parents=True, exist_ok=True)
    with open(f"outputs/bia/saida_match/json_dump.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(data_dict))

    criar_conll("bia", "", 0, 0, converted=True, sequential=True)


if __name__ == "__main__":
    app()
