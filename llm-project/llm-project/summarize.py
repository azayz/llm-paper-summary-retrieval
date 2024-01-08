from argparse import ArgumentParser

from langchain.prompts import PromptTemplate
import textwrap
from langchain.chains.summarize import load_summarize_chain
from utils import load_pdf_data, create_llm


def main(config):
    summarize(config.file_path, config.model_path, config.verbose)


def summarize(pdf_path: str, model_path: str, verbose: bool = False):
    """
    Summarizes a pdf paper
    :param pdf_path:    Path to the pdf paper
    :param model_path:  Path to the model
    :param verbose:    Whether to print verbose output
    :return:    None
    """

    # Prompt for summarization
    prompt_template = """Write a concise summary of the following:

    {text}

    CONSCISE SUMMARY:"""

    docs = load_pdf_data(pdf_path)  # Load the pdf paper that we want to summarize
    llm = create_llm(model_path, verbose=verbose) # Create the LLM model
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        combine_prompt=prompt,
        verbose=verbose,
    )  # Create the summarization chain using the LLM model which will summarize parts of the paper
    # and reduce to a final summary
    summary = chain.run(docs) # Run the summarization chain

    print(f"Chain type: {chain.__class__.__name__}")
    print(f"Summary: {textwrap.fill(summary, width=100)}")


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--file_path", type=str)
    argument_parser.add_argument("--model_path", type=str)
    argument_parser.add_argument("--verbose", type=bool, default=False)
    args = argument_parser.parse_args()
    main(args)
