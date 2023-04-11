import pickle
from evaluations.src.benchmark import Benchmark
from evaluations.src.matcher import Matcher
from OIE.datasets.validated_splits.generative_dataset import TripleExtraction, get_dataset
from OIE.datasets.validated_splits.contractions import transform_portuguese_contractions, clean_extraction
from typing import List
import tqdm
from predict import Predictor
import torch
import re
import pathlib

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "LSOIE4"
engine = Predictor(f"{name}/fine_tune")


def extract_anwsers(anwsers) -> List[TripleExtraction]:
    # Parse anwsers in the format:
    # E 1: ARG0= o máximo V= permitido ARG1= $ 5000"
    # E 2: ARG0= o mínimo V= permitido ARG1= $ 1000"

    # Get by regex E \d+: (.*)
    anwsers_matches = re.split(r'E \d+:', anwsers)
    anwsers = [anwser.strip() for anwser in anwsers_matches if anwser.strip()]

    parsed_anwsers = []

    for anwser in anwsers:
        try:

            # Split using regex ARG0=( .*)V=( .*)ARG1=( .*)
            arg0, rel, arg1 = re.match(r'ARG0=(.*)V=(.*)ARG1=(.*)', anwser).groups()

            if arg0 and rel and arg1:
                parsed_anwsers.append(
                    TripleExtraction(
                        arg1=arg0.strip(),
                        rel=rel.strip(),
                        arg2=arg1.strip(),
                    )
                )

        except Exception as e:
            pass
            # print(f"Invalid anwser: {anwser} - {e}")

    return parsed_anwsers



def generate_extractions(sentence: str):
    raw_result = engine.pred(sentence, False)
    count = 0
    extraction_str = ""
    for ext in raw_result:
        extraction_str += f"E {count}: "
        arg0 = ""
        rel = ""
        arg1 = ""
        for element in ext:
            if element[2] == "ARG0":
                arg0 = element[0]
            elif element[2] == "V":
                rel = element[0]
            elif element[2] == "ARG1":
                arg1 = element[0]
        arg0 = transform_portuguese_contractions(clean_extraction(arg0))
        rel = transform_portuguese_contractions(clean_extraction(rel))
        arg1 = transform_portuguese_contractions(clean_extraction(arg1))
        extraction_str += f"ARG0= {arg0} V= {rel} ARG1= {arg1} "
        count += 1
    return extract_anwsers(extraction_str)

def generate_results():
    train, test = get_dataset()
    for sentence in tqdm.tqdm(test, desc="Generating extractions"):
        result = generate_extractions(sentence.phrase)
        sentence.predicted_extractions = result
        # print(result)
    # Save test as pickle
    path = pathlib.Path("evaluations/benchmark/pickle")
    path.mkdir(parents=True, exist_ok=True)
    with open("evaluations/benchmark/pickle/lsoie2_results.pkl", "wb") as f:
        pickle.dump(test, f)


def evaluate():
    with open("evaluations/benchmark/pickle/lsoie2_results.pkl", "rb") as f:
        test = pickle.load(f)

    b = Benchmark()

    # Transform in dictionary
    predict_dict = dict()
    gold_dict = dict()
    for line in test:
        # print(line)
        gold_str = '\n'.join([str(x) for x in line.gold_extractions])
        predict_str = '\n'.join([str(x) for x in line.predicted_extractions])

        print(f"Sentença: {line.phrase} \n Gold:\n{gold_str} \n Predicted:\n{predict_str} "
              f"\n {'-' * 10}")
        predict_dict[line.phrase] = [x.to_dict() for x in line.predicted_extractions]
        gold_dict[line.phrase] = [x.to_dict() for x in line.gold_extractions]

    b.compare(
        gold=gold_dict,
        predicted=predict_dict,
        matchingFunc=Matcher.identicalMatch,
        output_fn=f"evaluations/benchmark/curve_{name}.txt",
    )

generate_results()
evaluate()