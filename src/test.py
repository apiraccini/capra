import shutil
from pathlib import Path

from utils.articles import search_articles, download_articles
from utils.llm import ask

def test_pipeline(debug):

    data_path = Path('./data')
    raw_data_path = data_path / 'raw'
    processed_data_path = data_path / 'processed'
    db_data_path = data_path / '.chromadb'

    if debug:

        if raw_data_path.exists():
            shutil.rmtree(raw_data_path)
        if processed_data_path.exists():
            shutil.rmtree(processed_data_path)
        if db_data_path.exists():
            shutil.rmtree(db_data_path)
        
        raw_data_path.mkdir(parents=True, exist_ok=True)
        search_articles("generative ai", outpath=raw_data_path / 'result_df.csv')
        download_articles(inpath=raw_data_path /'result_df.csv', outpath=raw_data_path / 'pdf_articles')

    response, _, _ = ask('what are the latest trends in large language models and generative ai?', raw_data_path, processed_data_path, db_data_path)
    
    print(response)
    return

if __name__ == '__main__':
    
    debug=False
    test_pipeline(debug)