import os
import chromadb
import pandas as pd

from tqdm import tqdm
from pdf2image import convert_from_path

from transformers import AutoProcessor, VisionEncoderDecoderModel, StoppingCriteriaList
from misc import  StoppingCriteriaScores
import torch


def get_or_create_corpus(raw_data_path, processed_data_path, db_data_path):
    '''Retrieve or create corpus from raw pdf files'''

    if not os.path.exists(db_data_path):
        
        raw_text_df = get_text_df(raw_data_path, processed_data_path)

        print('Creating embedded corpus...')
        corpus = create_corpus(raw_text_df, db_data_path)
    
    else:
        print('Loading corpus...')
        chroma_client = chromadb.PersistentClient(path=db_data_path)
        corpus = chroma_client.get_collection('articles')

    print('Done.')
    return corpus


def create_corpus(raw_text_df, db_data_path, chuck_per_article=6):
    '''Create embeddings using ChromaDB for the corpus from raw text df, with unique IDs based on chunk number'''

    chroma_client = chromadb.PersistentClient(path=db_data_path)
    corpus = chroma_client.create_collection(name="articles")

    batch_size = 50
    for i in range(0, len(raw_text_df), batch_size):

        batch_df = raw_text_df[i:i+batch_size]
        batch_ids = []
        batch_documents = []

        for _, row in batch_df.iterrows():
            
            text = row['article']
            chunk_size = int(len(text)/chuck_per_article)
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            chunk_ids = [f"{str(row['id'])}_chunk_{idx+1}" for idx in range(len(chunks))]

            batch_ids.extend(chunk_ids)
            batch_documents.extend(chunks)

        corpus.add(
            ids=batch_ids,
            documents=batch_documents
        )

    return corpus


def get_text_df(raw_data_path, output_dir):
    '''Process pdf directory to obtain a text dataframe'''

    pdf_data_path = f'{raw_data_path}/pdf_articles'
    if not os.path.exists(pdf_data_path):
        print(f'There are no pdf files in {pdf_data_path}.')
        return

    if not os.path.exists(output_dir):
        
        print('Need to process raw pdf files.') 
        os.makedirs(output_dir, exist_ok=True)

        data = []
        pdf_files = os.listdir(pdf_data_path)
        
        for i, file in enumerate(pdf_files):
            
            pdf =  f'{pdf_data_path}/{file}'
            images = convert_from_path(pdf_path=pdf)
            data.append((f'{i}', get_markdown(images)))

        out_df = pd.DataFrame(data, columns=['id', 'article'])
        output_path = f'{output_dir}/text_df.csv'
        out_df.to_csv(output_path, index=False)

        return out_df
    
    print('Loading processed text data...')
    out_df = pd.read_csv(f'{output_dir}/text_df.csv')

    return out_df


def get_markdown(images):
    '''obtain markdown string from pillow images'''

    # load model and processor
    processor = AutoProcessor.from_pretrained("facebook/nougat-small")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

    # move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text = []
    for image in tqdm(images, total=len(images)):
        
        # prepare image for the model
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # autoregressively generate tokens, with custom stopping criteria (as defined by the Nougat authors)
        outputs = model.generate(
            pixel_values.to(device),
            min_length=1,
            max_length=3584,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
        )

        # decode the generated IDs back to text and postprocess the generation.
        generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
        generated = processor.post_process_generation(generated, fix_markdown=False)
        text.append(generated)

    raw_markdown = ' '.join(text)
    markdown = ' '.join(raw_markdown.split()) # remove excess whitespaces and newlines

    return markdown


if __name__ == '__main__':

    raw_data_path = './data/raw'    
    processed_data_path = './data/processed'
    db_data_path = './data/.chromadb'

    corpus = get_or_create_corpus(raw_data_path, processed_data_path, db_data_path)
    print(corpus.peek())