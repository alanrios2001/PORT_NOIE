from src.create_txt_csv import Convert
from src.match import OIE_Match
import typer
import pathlib
from src.pos_tag import PosTag
from src.train_test_dev import train_dev_test
from src.merge_datasets import Merge

app = typer.Typer()


@app.command()
def criar_conll(out_name: str,
                txt_path: str,
                test_size: float,
                dev_size: float,
                input_path: str = None,
                converted: bool = False,
                sequential: bool = True
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
    datasets = ["splits/ls_train.txt",
                "splits/pud_200.txt",
                "splits/ls_dev.txt"]
    OUTPUT_NAME = "ls_train_plus"
    Merge(datasets, OUTPUT_NAME)

@app.command()
def train_dev_test():
    TEST_SIZE = 0.0
    DEV_SIZE = 0.0
    OUTPUT_NAME = "ls_train_plus"
    IN_PATH = "merges"
    OUT_PATH = "saida_pos_tag"
    train_dev_test.train_dev_test(TEST_SIZE, DEV_SIZE, OUTPUT_NAME, IN_PATH, OUT_PATH)



if __name__ == "__main__":
    app()
