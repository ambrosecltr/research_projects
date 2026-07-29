def doc_to_text(doc):
    return {"1": 0, "2": 1}[doc["answer"]]


def doc_to_target(doc):
    index = doc["sentence"].index("_") + 1
    return doc["sentence"][index:].strip()


def doc_to_choice(doc):
    index = doc["sentence"].index("_")
    return [
        doc["sentence"][:index] + doc["option1"],
        doc["sentence"][:index] + doc["option2"],
    ]
