from datasets import Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import login
from huggingface_hub import create_repo
from huggingface_hub import upload_file

def main():
    load_dotenv()
    login()
    upload_file(
        path_or_fileobj="data/processed/dataset.json",  # path absoluto o desde raíz del proyecto
        path_in_repo="dataset.json",
        repo_id="kentokamg/ticket-dataset",
        repo_type="dataset"
    )

if __name__ == "__main__":
    main()