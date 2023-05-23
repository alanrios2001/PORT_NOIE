import spacy
from spacy.matcher import Matcher
import os

class OIE_Match:
    def __init__(self, sequential: bool = True):
        self.sequential = sequential
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except:
            print("Baixando pt_core_news_lg")
            os.system("python -m spacy download pt_core_news_lg")
            self.nlp = spacy.load("pt_core_news_lg")

    def match(self, sent, arg1, rel, arg2):
        # preprocessamento da sentença e dos elementos da tripla
        sentence = self.nlp(sent.lower())
        arg1 = arg1.lower()
        rel = rel.lower()
        arg2 = arg2.lower()

        arg1 = self.nlp(arg1)
        rel = self.nlp(rel)
        arg2 = self.nlp(arg2)

        # encontrar arg1
        arg1_matcher = Matcher(self.nlp.vocab)
        # cria o padrão de busca para o arg1
        pattern = []
        for pos, token in enumerate(arg1):
            # If is not the first or the last
            if pos != 0 and pos != len(arg1) - 1:
                pattern.append(
                    {"IS_PUNCT": True, "OP": "?"}
                )  # This is to handle the case where the arg1 is between two punctuations
            pattern.append({"LOWER": token.text})
        if len(pattern) > 0:
            arg1_matcher.add("arg1", [pattern])
        # faz a busca pelo match
        arg1_match = arg1_matcher(sentence)
        # padroniza o match
        try:
            arg1_match = (arg1_match[0][1], arg1_match[0][2] - 1)
        except:
            arg1_match = ()

        # encontrar relações
        rel_matcher = Matcher(self.nlp.vocab)
        # cria o padrão de busca para o rel
        pattern = []
        for pos, token in enumerate(rel):
            if pos != 0 and pos != len(arg1) - 1:
                pattern.append(
                    {"IS_PUNCT": True, "OP": "?"}
                )  # This is to handle the case where the rel is between two punctuations
            pattern.append({"LOWER": token.text})
        if len(pattern) > 0:
            rel_matcher.add("rel", [pattern])
        # faz a busca pelo match
        rel_match = rel_matcher(sentence)
        # se houver mais de um match, seleciona o que está após o arg1
        if len(rel_match) > 1 and len(arg1_match) > 0:
            for match in rel_match:
                if match[1] > arg1_match[1]:
                    rel_match = [match]
                    break
        # padroniza o match
        try:
            rel_match = (rel_match[0][1], rel_match[0][2] - 1)
        except:
            rel_match = ()

        # encontrar arg2
        arg2_matcher = Matcher(self.nlp.vocab)
        # cria o padrão de busca para o arg2
        pattern = []
        for pos, token in enumerate(arg2):
            if pos != 0 and pos != len(arg1) - 1:
                pattern.append({"IS_PUNCT": True, "OP": "?"})
            pattern.append({"LOWER": token.text})
        if len(pattern) > 0:
            arg2_matcher.add("arg2", [pattern])
        arg2_match = arg2_matcher(sentence)
        # se houver mais de um match, seleciona o que está após a relação
        if len(arg2_match) > 1 and len(rel_match) > 0:
            for match in arg2_match:
                if match[1] > rel_match[1]:
                    arg2_match = [match]
                    break
        # padroniza o match
        try:
            arg2_match = (arg2_match[0][1], arg2_match[0][2] - 1)
        except:
            arg2_match = ()


        #seleciona extração válida por ser sequencial ou não e por ter todos os elementos
        if len(arg1_match) > 0 and len(rel_match) > 0 and len(arg2_match) > 0:
            if self.sequential:
                if arg1_match[1] < rel_match[0] and rel_match[1] < arg2_match[0]:
                    #print("EXTRAÇÃO VÁLIDA", "sent:", sent, "arg1:", arg1, "rel:", rel, "arg2:", arg2)
                    return (arg1_match, rel_match, arg2_match, True)
                else:
                    #print("EXTRAÇÃO INVÁLIDA(NÃO SEQUENCIAL)", "sent:", sent, "arg1:", arg1, "rel:", rel, "arg2:", arg2)
                    return (arg1_match, rel_match, arg2_match, False)
            else:
                #print("EXTRAÇÃO VÁLIDA", "sent:", sent, "arg1:", arg1, "rel:", rel, "arg2:", arg2)
                return (arg1_match, rel_match, arg2_match, True)
        else:
            #print("EXTRAÇÃO INVÁLIDA(ELEMENTO SEM CORRESPONDENCIA)", "sent:", sent, "arg1:", arg1, "rel:", rel, "arg2:", arg2)
            return (arg1_match, rel_match, arg2_match, False)