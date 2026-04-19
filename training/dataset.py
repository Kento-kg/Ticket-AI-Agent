from datasets import Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import login
from huggingface_hub import create_repo
from huggingface_hub import upload_file

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login()

upload_file(
    path_or_fileobj="../data/processed/dataset.json",
    path_in_repo="dataset.json",
    repo_id="kentokamg/ticket-dataset",
    repo_type="dataset"
)