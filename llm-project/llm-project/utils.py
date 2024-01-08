from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.llms import LlamaCpp


def load_pdf_data(pdf_path: str):
    """
    Loads langchain docs data from a pdf file
    :param pdf_path: File for the pdf paper to load
    :return: Langchain documents containing chunks of the pdf text
    """
    loader = PyPDFLoader(pdf_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs


def load_text_data(folder_path: str = 'extracted-text'):
    """
    Loads langchain docs data from a folder of text files
    :param folder_path:  Folder containing text files
    :return: Langchain documents containing chunks of the text files
    """
    loader = DirectoryLoader(folder_path, glob="*.txt")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)
    return docs


def create_llm(model_path: str, verbose: bool = False):
    """
    Creates a LLM model
    :param model_path:  Path to the model
    :param verbose: Whether to print verbose output
    :return: LLM model
    """
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=500,
        top_p=1,
        verbose=verbose,
        n_ctx=3096,  # Max context allowed
        n_gpu_layers=50,  # Activate the GPU
    )

    return llm
