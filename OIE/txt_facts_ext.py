from predict import Predictor
from OIE.datasets.feedback_dataset import FeedBackDataset
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions

model = "TA2"
oie = Predictor(model)
show_triple = True
fb = FeedBackDataset()

with open("text.txt", "r", encoding="utf-8") as f:
    lines = f.read()
    lines = lines.replace("\n", "")
    lines = lines.split(".")[:-1]

exts = []
for line in lines:
    ext = oie.pred(transform_portuguese_contractions(line), False)
    for e in ext:
        ex = [transform_portuguese_contractions(line)]
        for i in e:
            ex.append(i)
        exts.append(ex)

print('quantidade de extrações:', len(exts))
print('-'*50)
displayed_extractions = []
for i, ex in enumerate(exts):
    lenght = len(ex)
    extraction = f"{i} - extração:"
    if lenght > 2:
        displayed_extractions.append([transform_portuguese_contractions(ex[0]),
                                      transform_portuguese_contractions(ex[1][0]),
                                      transform_portuguese_contractions(ex[2][0]),
                                      transform_portuguese_contractions(ex[3][0])])
        for i in range(1, lenght):
            extraction += f" {ex[i][0]}"
        if show_triple:
            print(extraction + " → " + f"(ARG0: {ex[1][0]})" + f"(REL: {ex[2][0]})" + f"(ARG1: {ex[3][0]})")
        else:
            print(extraction)


valid_idx = input('selecione as extrações corretas pelo número da extração, separados por vírgula ex:(0,2,4): ')
#valid_idx = '0,1,2,3,4,5,6,7,8'
valid_idx = valid_idx.split(',')
valid_idx = [int(i) for i in valid_idx]
for i in valid_idx:
    fb.main(displayed_extractions[i][0], displayed_extractions[i][1], displayed_extractions[i][2], displayed_extractions[i][3])

