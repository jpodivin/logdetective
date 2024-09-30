import logging
import sys

from logdetective.utils import (
    process_log, initialize_model, retrieve_log_content, format_snippets, compute_certainty)
from logdetective.extractors import LLMExtractor, DrainExtractor

LOG = logging.getLogger("logdetective")


def run(model_path: str, filename_suffix: str, verbose: int,
        summarizer: str, n_clusters: int, file: str, no_stream: bool):
    """Main execution function."""

    # Primary model initialization
    try:
        model = initialize_model(model_path, filename_suffix=filename_suffix,
                                 verbose=verbose > 2)
    except ValueError as e:
        LOG.error(e)
        LOG.error("You likely do not have enough memory to load the AI model")
        sys.exit(3)

    # Log file summarizer selection and initialization
    if summarizer == "drain":
        extractor = DrainExtractor(verbose > 1, context=True, max_clusters=n_clusters)
    else:
        summarizer_model = initialize_model(summarizer, verbose=verbose > 2)
        extractor = LLMExtractor(summarizer_model, verbose > 1)

    LOG.info("Getting summary")

    try:
        log = retrieve_log_content(file)
    except ValueError as e:
        # file does not exists
        LOG.error(e)
        sys.exit(4)
    log_summary = extractor(log)

    ratio = len(log_summary) / len(log.split('\n'))

    LOG.info("Compression ratio: %s", ratio)

    LOG.info("Analyzing the text")

    log_summary = format_snippets(log_summary)
    LOG.info("Log summary: \n %s", log_summary)

    stream = True
    if no_stream:
        stream = False
    response = process_log(log_summary, model, stream)
    probs = []
    print("Explanation:")
    if no_stream:
        print(response["choices"][0]["text"])
        probs = response["choices"][0]["logprobs"]["top_logprobs"]
    else:
        # Stream the output
        for chunk in response:
            if isinstance(chunk["choices"][0]["logprobs"], dict):
                probs.extend(chunk["choices"][0]["logprobs"]["top_logprobs"])
            delta = chunk['choices'][0]['text']
            print(delta, end='', flush=True)
    certainty = compute_certainty(probs)

    print(f"\nResponse certainty: {certainty:.2f}%\n")
