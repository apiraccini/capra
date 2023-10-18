# CAPRA - Context AI Powered Research Assistant

Concept:
- search for articles on arxiv
- load each article into a corpus in chunks and obtain embeddings
- when questioned, provide context from the corpus via similarity with the query and answer using a gpt model (RAG)

The pdf articles are processed using the model Nougat, first proposed in [Nougat: Neural Optical Understanding for Academic Documents](https://doi.org/10.48550/arXiv.2308.13418) and accessible via [HuggingFace](https://huggingface.co/) `transformers`, in order to extract the markdown text.

## Notes

You will need to create your own `.env` file inside the root prokect directory with you OpenAI API key inside
As of now, the arXiv API seems unreliable.
Will not make us of docker containers until Nougat is included with a stable version of `transformers`.