import gradio as gr
from utils.app_utils import search_articles, download_articles
import os
import pandas as pd

result_df_path = './data/raw/result_df.csv'
pdf_articles_path = './data/raw/pdf_articles'
os.makedirs(pdf_articles_path, exist_ok=True)

with gr.Blocks(gr.themes.Soft()) as demo:

    gr.Markdown('# APRA - AI Powered Research Assistant')

    with gr.Tab(label='Search for your references'):

        query = gr.Textbox(label='Search query', placeholder='Insert your arXiv query here')
        query_btn = gr.Button(value='Search')
        search_results = gr.Dataframe(headers=['title', 'authors', 'abstract', 'entry_id'])
        filter_btn = gr.Button(value='Download articles')

        query_btn.click(fn=lambda x: search_articles(x, outpath=result_df_path), inputs=query, outputs=search_results)
        filter_btn.click(fn=lambda: download_articles(inpath=result_df_path, outpath=pdf_articles_path))


    with gr.Tab(label='Obtain embedded corpus'):

        out = gr.Textbox()
        def f():
            return os.listdir(pdf_articles_path)
        btn = gr.Button()
        btn.click(fn=f, outputs=out)

if __name__ == '__main__':
    demo.launch()