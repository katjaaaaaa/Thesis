import json
import sys

def main(argv):
    with open(argv[1]) as f:
        results_dict = json.load(f)

    for k, v in results_dict.items():
        print(k)
        print(v)
        print()

if __name__ == "__main__":
    main(sys.argv)