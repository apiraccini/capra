import gradio as gr
import os

from utils.articles import search_articles, download_articles
from utils.llm import ask

raw_data_path = './data/raw'
os.makedirs(raw_data_path, exist_ok=True)

with gr.Blocks(gr.themes.Soft()) as demo:

    gr.Markdown('# CAPRA - Context-enhanced AI Powered Research Assistant')

    with gr.Tab(label='Search for your references'):

        query = gr.Textbox(label='Search query', placeholder='Insert your arXiv query here')
        query_btn = gr.Button(value='Search')
        search_results = gr.Dataframe(headers=['title', 'authors', 'abstract', 'entry_id'])
        download_btn = gr.Button(value='Download articles')

        query_btn.click(fn=lambda x: search_articles(x, outpath=f'{raw_data_path}/result_df.csv'), inputs=query, outputs=search_results)
        download_btn.click(fn=lambda: download_articles(inpath=f'{raw_data_path}/result_df.csv', outpath=f'{raw_data_path}/pdf_articles'))


    with gr.Tab(label='Ask Capra'):

        question = gr.Textbox(label= 'Your question', placeholder='Insert your question here')
        ask_btn = gr.Button(value='Ask')
        response = gr.Markdown()

        with gr.Accordion(label='Details', open=False):
        
            log = gr.Markdown()
            prompt = gr.Markdown()

    ask_btn.click(fn=ask, inputs=question, outputs=[response, prompt, log], api_name="capra_answer")


if __name__ == '__main__':
    demo.launch()