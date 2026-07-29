import re


def wikitext_detokenizer(doc):
    value = doc["page"]
    value = value.replace("s '", "s'")
    value = re.sub(r"/' [0-9]/", r"/'[0-9]/", value)
    value = value.replace(" @-@ ", "-")
    value = value.replace(" @,@ ", ",")
    value = value.replace(" @.@ ", ".")
    for source, target in (
        (" : ", ": "),
        (" ; ", "; "),
        (" . ", ". "),
        (" ! ", "! "),
        (" ? ", "? "),
        (" , ", ", "),
    ):
        value = value.replace(source, target)
    value = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", value)
    value = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", value)
    value = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", value)
    value = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', value)
    value = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", value)
    value = value.replace("= = = =", "====")
    value = value.replace("= = =", "===")
    value = value.replace("= =", "==")
    value = value.replace(f" {chr(176)} ", chr(176))
    value = value.replace(" \n", "\n")
    value = value.replace("\n ", "\n")
    value = value.replace(" N ", " 1 ")
    return value.replace(" 's", "'s")


def process_results(doc, results):
    (loglikelihood,) = results
    words = len(re.split(r"\s+", doc["page"]))
    byte_count = len(doc["page"].encode())
    return {
        "word_perplexity": (loglikelihood, words),
        "byte_perplexity": (loglikelihood, byte_count),
        "bits_per_byte": (loglikelihood, byte_count),
    }
