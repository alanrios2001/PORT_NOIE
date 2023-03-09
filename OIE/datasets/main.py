from src.create_txt_csv import Convert
from src.match import OIE_Match
import typer
import pathlib
from src.pos_tag import PosTag
from src.train_test_dev import train_dev_test

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
        input_path = "datasets/"
    path = f"{input_path}/outputs/{out_name}"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    path_saida_match = f"{input_path}outputs/{out_name}"+"/saida_match"
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


if __name__ == "__main__":
    app()
