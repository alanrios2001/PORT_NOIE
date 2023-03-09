import json
import pathlib

dictio = {}
cond = True
i = 0

path = pathlib.Path("saida_criar_corpus")
path.mkdir(parents=True, exist_ok=True)

while cond:
    sent = input("Digite a sentença: ")
    arg1 = input("Digite o argumento 1: ")
    rel = input("Digite a relação: ")
    arg2 = input("Digite o argumento 2: ")

    dictio[i] = {}
    dictio[i]["sent"] = sent
    dictio[i]["ext"] = [{"arg1": arg1, "rel": rel, "arg2": arg2}]

    opc = input("continuar? (s/n)")
    if opc == "s":
        cond = True
        i = i + 1
    else:
        cond = False

with open(f"saida_criar_corpus/json_dump.json", "a", encoding ="utf-8") as file:
    file.write(json.dumps(dictio))
