import sys

from language_identification.models import AllInOneClassifier

classifer = AllInOneClassifier.load("data/final-model")
print("model loaded.")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        for filepath in sys.argv[1:]:
            with open(filepath) as f:
                lines = f.readlines()
                doc = "\n".join(lines)
                pred = classifer.predict([doc])[0]
                print("file:", filepath, ", label:", pred)
    else:
        while True:
            doc = input("> ")
            pred = classifer.predict([doc])[0]
            print("label:", pred)
