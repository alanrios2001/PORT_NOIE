from predict import Predictor
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions
from openie_helper.triple import OIESentence

model = "TA_bertina"
oie = Predictor(model)
show_triple = True

with open("texto.txt", "r", encoding="utf-8") as f:
    lines = f.read()
    lines = lines.replace("\n\n", " ")
    #lines = lines.replace("Dr. ", "Dr.")
    lines = lines.split(". ")[:-1]

exts = []
raw_exts = []
for line in lines:
    ext = oie.pred(transform_portuguese_contractions(line), False)
    raw_exts.append(ext)
    for e in ext:
        ex = []
        for i in e:
            ex.append(i)
        exts.append((transform_portuguese_contractions(line), ex))

print('quantidade de extrações: ', len(exts))
print("-----------------------------------")
for i, ex in enumerate(exts):
    extraction = f"{i} - extração: "
    if show_triple:
        print(extraction + " → " + f"(ARG0: {ex[1][0][0]})" + f"(REL: {ex[1][1][0]})" + f"(ARG1: {ex[1][2][0]})")
    else:
        print(extraction)
