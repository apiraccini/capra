import os
import shutil

from utils.articles import search_articles, download_articles
from utils.llm import ask

def test_pipeline(debug):

    raw_data_path = './data/raw'    
    processed_data_path = './data/processed'
    db_data_path = './data/.chromadb'

    if debug:

        if os.path.exists(raw_data_path):
            shutil.rmtree(raw_data_path)
        if os.path.exists(processed_data_path):
            shutil.rmtree(processed_data_path)
        if os.path.exists(db_data_path):
            shutil.rmtree(db_data_path)
        
        os.makedirs(raw_data_path, exist_ok=True)
        search_articles("generative ai", outpath=f'{raw_data_path}/result_df.csv')
        download_articles(inpath=f'{raw_data_path}/result_df.csv', outpath=f'{raw_data_path}/pdf_articles')

    response, _, _ = ask('what are the latest trends in large language models and generative ai?', raw_data_path, processed_data_path, db_data_path)
    
    print(response)
    return

if __name__ == '__main__':
    
    debug=False
    test_pipeline(debug)