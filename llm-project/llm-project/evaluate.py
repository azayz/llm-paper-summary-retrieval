from rouge import Rouge
from bert_score import score
from argparse import ArgumentParser


def main(config):
    evaluate(config.model_result, config.ground_truth)


def evaluate(model_result, ground_truth):
    """
    Evaluates a model result against ground truth
    :param model_result:    Model result
    :param ground_truth:    Ground truth
    :return:    Evaluation score
    """
    rouge = Rouge()
    bert_score = score(model_result, ground_truth, lang="en", verbose=True)  # Calculate BERT score
    rouge_score = rouge.get_scores(model_result, ground_truth)  # Calculate ROUGE score
    print(f"BERT score: {bert_score}")
    print(f"ROUGE score: {rouge_score}")
    return bert_score, rouge_score


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--model_result", type=str)
    argument_parser.add_argument("--ground_truth", type=str)
    args = argument_parser.parse_args()
    main(args)
