import openai
import configparser
import os
import pandas as pd
from utils import send_text_to_chatgpt, read_bibtex, bibtex_to_dataframe, retrieve_additional_data, iterate_over_entries

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['API']['api_key']
openai.api_key = api_key

if __name__ == "__main__":
    data = read_bibtex('data/Merged_OPC UA.bib')
    data = bibtex_to_dataframe(data)
    print(data.head())  
    iterate_over_entries(data[:1], api_key)
