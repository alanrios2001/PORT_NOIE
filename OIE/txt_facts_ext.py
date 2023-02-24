from predict import precict2
from flair.models import SequenceTagger

model = "PTOIE_pos"
oie = SequenceTagger.load("train_output/"+model+"/best-model.pt")


with open("texto.txt", "r", encoding="utf-8") as f:
    lines = f.read()
    lines = lines.split(".")[:-1]

exts = []
for line in lines:
    ext = precict2(oie, line, False)
    for e in ext:
        exts.append(e)
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

