from typing import List

import bibtexparser
import instructor
import pandas as pd
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field

class PaperInfo(BaseModel):
    """Data model for paper classification."""

    evaluation_example: str
    use_case: str
    application_domain: str
    technology_used: List[str] = Field(
        description="List of acronyms of the technologies used in the paper. DO NOT WRITE OR EXPLAIN THE MEANING OF THE ACRONYMS!")

def send_text_to_chatgpt(api_key, user_message, system_message="You are a helpful assistant.", model="gpt-3.5-turbo"):
    """Send a text prompt to the ChatGPT model and return the response."""

    client = OpenAI(
                api_key=api_key
            )
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )
        
        # Extract the assistant's response
        assistant_message = completion.choices[0].message.content
        return assistant_message

    except Exception as e:
        return f"An error occurred: {e}"

def send_text_to_chatgpt_structured(api_key, user_message, system_message="You are a helpful assistant.", model="gpt-3.5-turbo"):
    client = instructor.from_openai(
        OpenAI(api_key=api_key),
        mode=instructor.Mode.JSON)

    paper_info = client.chat.completions.create(
        model=model,
        response_model=PaperInfo,
        messages=[{"role": "user",
                   "content": user_message}],
    )
    return paper_info.model_dump_json(indent=2)

def read_bibtex(file_path):
    """Reads a BibTeX file and returns a list of entries."""
    try:
        with open(file_path, 'r', encoding='utf-8') as bib_file:
            parser = BibTexParser()
            parser.customization = convert_to_unicode  # Convert LaTeX characters
            bib_data = bibtexparser.load(bib_file, parser=parser)
            
            print(f"Loaded {len(bib_data.entries)} entries from {file_path}.")
            return bib_data.entries

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def bibtex_to_dataframe(bib_entries):
    """Converts a list of BibTeX entries to a Pandas DataFrame."""
    if not bib_entries:
        print("No entries found to convert.")
        return pd.DataFrame()  # Return empty DataFrame if no entries

    # Convert list of dictionaries to a DataFrame
    df = pd.DataFrame(bib_entries)
    
    # Optional: Fill NaN for missing columns if needed
    df.fillna("N/A", inplace=True)
    return df

def retrieve_additional_data(abstract, api_key):
    """Retrieve additional data from the abstract using GPT-3."""
    prompt = f"""Given the abstract below, please provide additional insights or information. 
                present the Information as comma separated values. Give nothing else as output.
                Additional information: Technology that was used, Evaluation Example, Use Case, Application Domain.
                Abstract: 
                {abstract}
                Technology that was used, Evaluation Example, Use Case, Application Domain
                """
    return send_text_to_chatgpt(api_key, prompt)


def retrieve_additional_structured_data(abstract, api_key):
    """Retrieve additional data from the abstract using GPT-3."""
    prompt = f"Analyze the following abstract, try to be short and precise {abstract}"
    return send_text_to_chatgpt_structured(api_key, prompt)

def iterate_over_entries(data, api_key):
    for index, row in data.iterrows():
        abstract = row['abstract']
        additional_data = retrieve_additional_structured_data(abstract, api_key)
        print(f"Abstract: {abstract}\nAdditional Data: {additional_data}\n")