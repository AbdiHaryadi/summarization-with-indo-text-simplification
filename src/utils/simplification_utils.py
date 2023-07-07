import sys

sys.path.append(".")
from indo_ts.src.utils import word_list_to_sentence, sentence_to_word_list
from indo_ts.src.indo_ts import TextSimplifier

def generate_simplify_corpus_function():
    simplifier = TextSimplifier(tokenize_no_ssplit=True, strategy=5)
    def simplify_corpus(corpus_list):
        new_corpus_list = []
        for single_corpus in corpus_list:
            sentence_list = []
            for word_list in single_corpus:
                sentence = word_list_to_sentence(word_list)
                sentence_list.append(sentence)

            simplified_sentences_list = simplifier.simplify("\n\n".join(sentence_list))

            new_single_corpus = []
            for simplified_sentences in simplified_sentences_list:
                for sentence in simplified_sentences:
                    word_list = sentence_to_word_list(sentence)
                    new_single_corpus.append(word_list)
            
            new_corpus_list.append(new_single_corpus)

        return new_corpus_list
        
    return simplify_corpus
