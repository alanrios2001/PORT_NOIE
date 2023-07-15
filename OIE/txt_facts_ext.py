from predict import Predictor
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions
from openie_helper.triple import OIESentence

model = "TA2/fine_tune"
oie = Predictor(model)
show_triple = True

with open("texto.txt", "r", encoding="utf-8") as f:
    lines = f.read()
    lines = lines.replace("\n", "")
    lines = lines.split(".")[:-1]

exts = []
for line in lines:
    ext = oie.pred(transform_portuguese_contractions(line), False)
    for e in ext:
        ex = []
        for i in e:
            ex.append(i)
        exts.append(ex)

for ex in exts:
    extraction = "extração:"
    if len(ex) > 2:
        for i in range(len(ex)):
            extraction += f" {ex[i][0]}"
        if show_triple:
            print(extraction + " → " + f"(ARG0: {ex[0][0]})" + f"(REL: {ex[1][0]})" + f"(ARG1: {ex[2][0]})")
        else:
            print(extraction)
