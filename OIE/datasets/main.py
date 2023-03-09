from OIE.datasets.create_txt_csv import Convert
from OIE.datasets.match import OIE_Match
import typer
import pathlib
from OIE.datasets.pos_tag import PosTag
from OIE.datasets.train_test_dev import train_dev_test

app = typer.Typer()


@app.command()
def criar_conll(out_name: str,
                input_path: str,
                test_size: float,
                dev_size: float,
                converted: bool = False,
                sequential: bool = True
                ):
    path = f"outputs/{out_name}"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    path_saida_match = f"outputs/{out_name}"+"/saida_match"
    pathlib.Path(path_saida_match).mkdir(parents=True, exist_ok=True)

    if not converted:
        Convert(input_path, path, out_name)

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
