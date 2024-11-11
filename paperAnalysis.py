import configparser

import openai

from utils import read_bibtex, bibtex_to_dataframe, iterate_over_entries

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['API']['api_key']
openai.api_key = api_key

if __name__ == "__main__":
    data = read_bibtex('data/Merged_OPC UA.bib')
    data = bibtex_to_dataframe(data)
    print(data.head())  
    iterate_over_entries(data[:1], api_key)
