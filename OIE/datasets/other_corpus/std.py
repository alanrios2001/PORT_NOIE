from OIE.datasets.conll2bioes import Conversor
from OIE.datasets.pos_tag import PosTag
import typer
import pathlib

app = typer.Typer()

@app.command()
def main(dataset: str):
    path = pathlib.Path("other_corpus/mod")
    path.mkdir(parents=True, exist_ok=True)

    total = 0

    with open("other_corpus/" + dataset, "r", encoding="utf-8") as file:
        with open("other_corpus/mod/" + dataset, "a", encoding="utf-8") as file2:
            lines = file.read()
            lines = lines.split("\n\n")
            for line in lines:
                if "ARG0" in line and "ARG1" in line and "V" in line:
                    file2.write(line + "\n\n")