from flair.models import SequenceTagger
from flair.data import Sentence
import typer

app = typer.Typer()


@app.command()
def precict(model:str, text:str, show_output: bool):
    try:
        oie = SequenceTagger.load("train_output/"+model+"/best-model.pt")
    except:
        oie = SequenceTagger.load("train_output/"+model+"/final-model.pt")
    if "_lower" in model:
        sentence = Sentence(text.lower())
    else:
        sentence = Sentence(text)
    oie.predict(sentence)

    #separa elementos da tripla
    arg0 = [(span.text, span.score) for span in sentence.get_spans('label') if span.tag == "ARG0"]
    rel = [(span.text, span.score) for span in sentence.get_spans('label') if span.tag == "V"]
    arg1 = [(span.text, span.score) for span in sentence.get_spans('label') if span.tag == "ARG1"]

    # monta as extrações
    if len(arg0) == 0:
        arg0_new = ("", 0)
    if len(rel) == 0:
        rel_new = ("", 0)
    if len(arg1) == 0:
        arg1_new = ("", 0)
    if len(arg0) == 1:
        arg0_new = arg0[0]
    if len(rel) == 1:
        rel_new = rel[0]
    if len(arg1) == 1:
        arg1_new = arg1[0]

    exts = []
    if len(arg0) == 1 and len(rel) == len(arg1) > 1:
        for i in range(len(arg1)):
            ext = []
            ext.append(max(arg0, key=lambda x: x[1]))
            ext.append(max(rel, key=lambda x: x[1]))
            ext.append(max(arg1, key=lambda x: x[1]))
            rel.remove(ext[1])
            arg1.remove(ext[2])
            exts.append(ext)

    if len(arg0) == len(rel) == len(arg1) > 1:
        for i in range(len(arg1)):
            ext = []
            ext.append(max(arg0, key=lambda x: x[1]))
            ext.append(max(rel, key=lambda x: x[1]))
            ext.append(max(arg1, key=lambda x: x[1]))
            arg0.remove(ext[0])
            rel.remove(ext[1])
            arg1.remove(ext[2])
            exts.append(ext)

    if len(exts)==0:
        try:
            arg0_new = max(arg0, key=lambda x: x[1])
        except:
            arg0_new = ("", 0)
        try:
            rel_new = max(rel, key=lambda x: x[1])
        except:
            rel_new = ("", 0)
        try:
            arg1_new = max(arg1, key=lambda x: x[1])
        except:
            arg1_new = ("", 0)
        exts.append([arg0_new, rel_new, arg1_new])

    splited = sentence.to_tagged_string().split("→")
    sentenca = splited[0].split("Sentence:")[1]
    maior = sentenca
    try:
        tripla = splited[1]
    except:
        tripla = splited

    if len(sentenca) < len(tripla):
        maior = tripla
    if len(maior) < len(str(sentence.get_spans('label'))):
        maior = str(sentence.get_spans('label'))

    if show_output:
        print("\n" * 1)
        print("| ", "-" * len(maior), " |")
        for ext in exts:
            print("Extração: ", ext[0][0] + " " + ext[1][0] + " " + ext[2][0])

        print("| ", "-" * len(maior), " |")
        print("\n" * 1)
        print("| ", "-" * len(maior), " |")
        print("| ", int((len(maior) - len("MAIS INFO")) / 2 - 1) * "-", "MAIS INFO", int((len(maior) - len("MAIS INFO")) / 2) * "-", " |")
        print("| ", "-" * len(maior), " |")
        print("| ", "sentença: ", " "* (len(maior) - (len("sentença: ")+1)), " |")
        print("| ", sentenca," " * (len(maior) - (len(sentenca)+1))," |")
        print("| ", "-"*len(maior), " |")
        print("| ", "extrações: ", " " * (len(maior)-(len("extrações: ")+1)), " |")
        print("| ", tripla, " "* (len(maior)- (len(tripla)+1)), " |")
        print("| ", "-"*len(maior), " |")
        print("| ", "probs: ", " " * (len(maior)-(len("probs: ")+1)), " |")
        print("| ", sentence.get_spans('label'), " " * (len(maior)-(len(str(sentence.get_spans('label')))+1)), " |")
        print("| ", "-" * len(maior), " |")

    return exts


if __name__ == "__main__":
    app()


@app.command()
def precict2(oie: SequenceTagger, text:str, show_output: bool):

    sentence = Sentence(text)
    oie.predict(sentence)

    #separa elementos da tripla
    arg0 = [(span.text, span.score) for span in sentence.get_spans('label') if span.tag == "ARG0"]
    rel = [(span.text, span.score) for span in sentence.get_spans('label') if span.tag == "V"]
    arg1 = [(span.text, span.score) for span in sentence.get_spans('label') if span.tag == "ARG1"]

    # monta as extrações
    if len(arg0) == 0:
        arg0_new = ("", 0)
    if len(rel) == 0:
        rel_new = ("", 0)
    if len(arg1) == 0:
        arg1_new = ("", 0)
    if len(arg0) == 1:
        arg0_new = arg0[0]
    if len(rel) == 1:
        rel_new = rel[0]
    if len(arg1) == 1:
        arg1_new = arg1[0]

    exts = []
    if len(arg0) == 1 and len(rel) == len(arg1) > 1:
        for i in range(len(arg1)):
            ext = []
            ext.append(max(arg0, key=lambda x: x[1]))
            ext.append(max(rel, key=lambda x: x[1]))
            ext.append(max(arg1, key=lambda x: x[1]))
            rel.remove(ext[1])
            arg1.remove(ext[2])
            exts.append(ext)

    if len(arg0) == len(rel) == len(arg1) > 1:
        for i in range(len(arg1)):
            ext = []
            ext.append(max(arg0, key=lambda x: x[1]))
            ext.append(max(rel, key=lambda x: x[1]))
            ext.append(max(arg1, key=lambda x: x[1]))
            arg0.remove(ext[0])
            rel.remove(ext[1])
            arg1.remove(ext[2])
            exts.append(ext)

    if len(exts)==0:
        try:
            arg0_new = max(arg0, key=lambda x: x[1])
        except:
            arg0_new = ("", 0)
        try:
            rel_new = max(rel, key=lambda x: x[1])
        except:
            rel_new = ("", 0)
        try:
            arg1_new = max(arg1, key=lambda x: x[1])
        except:
            arg1_new = ("", 0)
        exts.append([arg0_new, rel_new, arg1_new])

    splited = sentence.to_tagged_string().split("→")
    sentenca = splited[0].split("Sentence:")[1]
    maior = sentenca
    try:
        tripla = splited[1]
    except:
        tripla = splited

    if len(sentenca) < len(tripla):
        maior = tripla
    if len(maior) < len(str(sentence.get_spans('label'))):
        maior = str(sentence.get_spans('label'))

    if show_output:
        print("\n" * 1)
        print("| ", "-" * len(maior), " |")
        for ext in exts:
            print("Extração: ", ext[0][0] + " " + ext[1][0] + " " + ext[2][0])

        print("| ", "-" * len(maior), " |")
        print("\n" * 1)
        print("| ", "-" * len(maior), " |")
        print("| ", int((len(maior) - len("MAIS INFO")) / 2 - 1) * "-", "MAIS INFO", int((len(maior) - len("MAIS INFO")) / 2) * "-", " |")
        print("| ", "-" * len(maior), " |")
        print("| ", "sentença: ", " "* (len(maior) - (len("sentença: ")+1)), " |")
        print("| ", sentenca," " * (len(maior) - (len(sentenca)+1))," |")
        print("| ", "-"*len(maior), " |")
        print("| ", "extrações: ", " " * (len(maior)-(len("extrações: ")+1)), " |")
        print("| ", tripla, " "* (len(maior)- (len(tripla)+1)), " |")
        print("| ", "-"*len(maior), " |")
        print("| ", "probs: ", " " * (len(maior)-(len("probs: ")+1)), " |")
        print("| ", sentence.get_spans('label'), " " * (len(maior)-(len(str(sentence.get_spans('label')))+1)), " |")
        print("| ", "-" * len(maior), " |")

    return exts