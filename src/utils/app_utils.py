import arxiv
import pandas as pd


def search_articles(query, outpath, max_retries=10):
    retries = 0

    while retries < max_retries:
        client = arxiv.Client(
            page_size=10,
            delay_seconds=4,
            num_retries=30,
        )
        search = arxiv.Search(
            query=query,
            max_results=10,
        )

        data = []
        for result in client.results(search):
            paper_data = {
                'title': result.title,
                'authors': ', '.join([res.name for res in result.authors]),
                'abstract': result.summary,
                'entry_id': result.entry_id
            }
            data.append(paper_data)

        out = pd.DataFrame(data)

        if not out.empty:
            out.to_csv(outpath, index=False)
            return out
        retries += 1
    
    out = pd.DataFrame()
    out.to_csv(outpath, index=False)

    return out

def download_articles(inpath, outpath):

    search_results = pd.read_csv(inpath)
    id_list = [id.split('/')[-1] for id in search_results['entry_id']]
    
    client = arxiv.Client(
        page_size=10,
        delay_seconds=4,
        num_retries=30,
    )
    search = arxiv.Search(
        id_list=id_list
    )

    for result in client.results(search):
        result.download_pdf(dirpath=outpath)
    
    return