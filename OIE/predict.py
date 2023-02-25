from flair.models import SequenceTagger
from flair.data import Sentence
import typer

app = typer.Typer()

class Predictor:
    def __init__(self, model:str):
        try:
            self.oie = SequenceTagger.load("train_output/" + model + "/best-model.pt")
        except:
            self.oie = SequenceTagger.load("train_output/" + model + "/final-model.pt")

    def display(self, maior, exts, sentenca, tripla, sentence: Sentence):
        print("\n" * 1)
        print("| ", "-" * len(maior), " |")
        for ext in exts:
            print("Extração: ", ext[0][0] + " " + ext[1][0] + " " + ext[2][0])
        print("| ", "-" * len(maior), " |")
        print("\n" * 1)
        print("| ", "-" * len(maior), " |")
        print("| ", int((len(maior) - len("MAIS INFO")) / 2 - 1) * "-", "MAIS INFO",
              int((len(maior) - len("MAIS INFO")) / 2) * "-", " |")
        print("| ", "-" * len(maior), " |")
        print("| ", "sentença: ", " " * (len(maior) - (len("sentença: ") + 1)), " |")
        print("| ", sentenca, " " * (len(maior) - (len(sentenca) + 1)), " |")
        print("| ", "-" * len(maior), " |")
        print("| ", "extrações: ", " " * (len(maior) - (len("extrações: ") + 1)), " |")
        print("| ", tripla, " " * (len(maior) - (len(tripla) + 1)), " |")
        print("| ", "-" * len(maior), " |")
        print("| ", "probs: ", " " * (len(maior) - (len("probs: ") + 1)), " |")
        print("| ", sentence.get_spans('label'), " " * (len(maior) - (len(str(sentence.get_spans('label'))) + 1)), " |")
        print("| ", "-" * len(maior), " |")


    def predict(self, text:str, show_output: bool):
        sentence = Sentence(text)
        self.oie.predict(sentence)

        # separa elementos da tripla
        arg0 = [(span.text, span.score, span.end_position) for span in sentence.get_spans('label') if span.tag == "ARG0"]
        rel = [(span.text, span.score, [span.start_position, span.end_position]) for span in sentence.get_spans('label') if span.tag == "V"]
        arg1 = [(span.text, span.score, span.start_position) for span in sentence.get_spans('label') if span.tag == "ARG1"]

        # cria extrações baseadas na proximidade da rel com arg0 e arg1
        exts = []
        if len(rel) > 0 and len(arg0) == 1 and len(arg1) == 1:
            for r in rel:
                if arg0[0][2] < r[2][0] and arg1[0][2] > r[2][1]:
                    exts.append([arg0, r, arg1])
        elif len(rel) > 0 and len(arg0) == 1 and len(arg1) > 1:
            for r in rel:
                if arg0[0][2] < r[2][0]:
                    for a in arg1:
                        if a[2] > r[2][1]:
                            exts.append([arg0, r, a])
        elif len(rel) > 0 and len(arg0) > 1 and len(arg1) == 1:
            for r in rel:
                if arg1[0][2] > r[2][1]:
                    for a in arg0:
                        if a[2] < r[2][0]:
                            exts.append([a, r, arg1])
        elif len(rel) > 0 and len(arg0) > 1 and len(arg1) > 1:
            for r in rel:
                for a in arg0:
                    if a[2] < r[2][0]:
                        for b in arg1:
                            if b[2] > r[2][1]:
                                exts.append([a, r, b])

        return exts


@app.command()
def run(model:str, text:str, show_output: bool = True):
    predictor = Predictor(model)
    predictor.predict(text, show_output)

if __name__ == "__main__":
    app()