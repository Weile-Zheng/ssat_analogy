import numpy as np
from cosim import cosine_similarity
from gensim.models import Word2Vec

model = Word2Vec.load("trainedModel.bin")


def questionParsing(string):
    '''
    MAJOR FIXING IS NEEDED. Parsing not versatile enough, wrongly deletes "as" if a word 
    does contain it
    '''
    return string.replace("is to", "").replace("as", "").replace("A)", "").replace("B)", "").replace("C)", "").replace("D)", "").replace("E)", "").replace(":", "").lower()


def solver(list):
    question = model.wv[list[0]] - model.wv[list[1]]
    a = model.wv[list[2]] - model.wv[list[3]]
    b = model.wv[list[4]] - model.wv[list[5]]
    c = model.wv[list[6]] - model.wv[list[7]]
    d = model.wv[list[8]] - model.wv[list[9]]
    e = model.wv[list[10]] - model.wv[list[11]]
    print("Cosine Similarities: ")
    print("Option A " + str(cosine_similarity(a, question)))
    print("Option B " + str(cosine_similarity(b, question)))
    print("Option C " + str(cosine_similarity(c, question)))
    print("Option D " + str(cosine_similarity(d, question)))
    print("Option E " + str(cosine_similarity(e, question)))


def userInput():
    print("Enter your multiline input. To finish, enter a blank line:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main():

    string = userInput()
    string = questionParsing(string)
    list = string.split()
    solver(list)


if __name__ == "__main__":
    main()
