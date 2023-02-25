from predict import Predictor
from flair.models import SequenceTagger

model = "PTOIE"
oie = Predictor(model)

with open("texto.txt", "r", encoding="utf-8") as f:
    lines = f.read()
    lines = lines.split(".")[:-1]

exts = []
for line in lines:
    ext = oie.pred(line, False)
    for e in ext:
        ex = []
        for i in e:
            if type(i) == list:
                ex.append(i[0])
            else:
                ex.append(i)
        exts.append(ex)

for ex in exts:
    n = 0
    for e in ex:
        if e[0] == "":
            try:
                exts.remove(ex)
                n =+ 1
            except:
                pass
    if n == 0:
        print(f"extração: {ex[0][0]} {ex[1][0]} {ex[2][0]}")
