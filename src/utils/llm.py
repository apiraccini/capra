import os
import tiktoken
import openai

from dotenv import load_dotenv

from corpus import get_or_create_corpus


def ask(question, raw_data_path, processed_data_path, db_data_path):
    '''CAPRA API: anwers a question using the context provided'''

    model = 'gpt-3.5-turbo'
    max_tokens = 512
    temperature = 0.2

    corpus = get_or_create_corpus(raw_data_path, processed_data_path, db_data_path)
    
    raw_result = corpus.query(query_texts=f'{question}', n_results=2)
    result = filter_result(raw_result)

    context = ' '.join([f'DOC: {id}\nCONTENT: {document}\n\n' for id, document in zip(result['ids'][0], result['documents'][0])])
    context = context.rstrip('\n')
    
    prompt = [
        {'role': 'system', 'content': "Take a deep breath and answer the following question based on the context provided. If you can't find any evidence, say you don't know."},
        {'role': 'user', 'content': f"Question: {question}. \nThe context, written in markdown syntax, is the following:\n<<<{context}>>>\n. Anwer:"}
    ]
    
    response =  gpt_response(model, prompt, max_tokens, temperature)
    prompt_string =  ' '.join([str(d) for d in prompt])
    log =  estimate_cost(prompt, model, max_tokens)
    
    return response, prompt_string, log


def gpt_response(model, messages, max_tokens, temperature):
    '''Wrapper for the API call to gpt model'''

    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']

    response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            max_tokens = max_tokens,
            temperature = temperature
        ).choices[0].message.content

    return response


def filter_result(result, threshold=100):
    '''Filter query result based on a threshold'''
    
    for ids, docs, distances in zip(result['ids'], result['documents'], result['distances']):
        for i in range(len(ids)-1, -1, -1):
            if distances[i] > threshold:
                ids.pop(i)
                docs.pop(i)
                distances.pop(i)
    return result


def estimate_cost(messages, model='gpt-3.5-turbo', max_tokens=512):
    '''Estimate cost for the OpenAI API calls'''

    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3

    if model == 'gpt-3.5-turbo':
        cost_per_input_token = 0.0015 / 1000
        cost_per_output_token = 0.002 / 1000
    elif model == 'gpt-3.5-turbo-16k':
        cost_per_input_token = 0.003 / 1000
        cost_per_output_token = 0.004 / 1000
    elif model == 'gpt-4':
        cost_per_input_token = 0.03 / 1000
        cost_per_output_token = 0.06 / 1000

    cost_estimate = num_tokens*cost_per_input_token + max_tokens*cost_per_output_token
    msg = f'Total tokens (input + max output): {num_tokens+max_tokens}; cost estimate: {cost_estimate} dollars.'
    
    return msg


if __name__ == '__main__':

    raw_data_path = './data/raw'    
    processed_data_path = './data/processed'
    db_data_path = './data/.chromadb'

    response, prompt, log = ask('what are the latest trends in large language models and generative ai?', raw_data_path, processed_data_path, db_data_path)
    print(response)