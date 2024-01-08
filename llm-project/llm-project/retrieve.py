# import
from argparse import ArgumentParser

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from utils import load_text_data


def main(config):
    retrieve_similar_papers(config.query, config.model_name, config.top_k)


def retrieve_similar_papers(
        query: str, model_name: str = "all-MiniLM-L6-v2", top_k: int = 3, folder_path: str = 'extracted-text'
):
    """
    Retrieves similar papers to a query
    :param query: Query to search for
    :param model_name:  Name of the model to use
    :param top_k:   Number of papers to retrieve
    :param folder_path: Folder containing text files
    """
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

    # load the data
    docs = load_text_data(folder_path)

    # load it into Chroma Vector Database
    db = Chroma.from_documents(docs, embedding_function)

    # query it
    docs = db.similarity_search(query)  # returns a list of Document objects
    for doc in docs[:top_k]:
        print(doc.metadata['source'])
        print(doc.page_content)


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--query", type=str)
    argument_parser.add_argument("--model_name", default="all-MiniLM-L6-v2", type=str)
    argument_parser.add_argument("--top_k", type=int, default=3)
    args = argument_parser.parse_args()
    main(args)
