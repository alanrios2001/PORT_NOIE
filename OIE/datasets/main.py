from create_txt_csv import Convert
from OIE.datasets.match import OIE_Match
import typer
import pathlib
from pos_tag import PosTag
from train_test_dev import train_dev_test

app = typer.Typer()


@app.command()
def criar_conll(out_name: str,
                json_dir: str,
                input_path: str,
                test_size: float,
                dev_size: float,
                ):

    path = pathlib.Path("saida_match")
    path.mkdir(parents=True, exist_ok=True)
    Convert(input_path, out_name)

    # selecionar e anotar sentenças validas
    oie_match = OIE_Match(out_name, json_dir)
    oie_match.run()

    # POS tagging
    PosTag(f"saida_match\{out_name}_corpus.txt").run(f"{out_name}_corpus.txt")

    # train, dev, test
    train_dev_test(test_size, dev_size, out_name, in_path="saida_pos_tag", out_path="saida_pos_tag")

if __name__ == "__main__":
    app()
