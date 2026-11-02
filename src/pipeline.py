# src/pipeline.py
from src.data import download
from src import evaluate

if __name__ == "__main__":
    download()
    evaluate.main()