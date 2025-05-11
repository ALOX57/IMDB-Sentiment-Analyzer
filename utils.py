import re

def preprocess(text):
    text = text.lower()
    
    # Join negation words with the word that follows
    text = re.sub(r"\b(not|no|never|n't)\s+(\w+)", r"not_\2", text)

    return text