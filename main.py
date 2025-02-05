import re
import math
import random
from collections import defaultdict, Counter
import nltk

class Spell_Checker:
    """The class implements a context-sensitive spell checker using the Noisy Channel framework,
       based on a language model and an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an instance variable.
        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm  # Language Model
        self.error_tables = {}

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
        Args:
            lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """Adds the specified dictionary of error tables as an instance variable.
        Args:
            error_tables (dict): a dictionary of error tables (confusion matrices).
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language model in use.
           when texts containing OOV words Using Smoothing
           Args:
               text (str): Text to evaluate.
           Returns:
               Float. The float should reflect the (log) probability.
        """
        if not self.lm:
            raise ValueError("Language model not loaded.")
        return self.lm.evaluate_text(text)

    def spell_check(self, text, alpha=0.95):
        """Returns the most probable fix for the specified text using a noisy channel model.
        This method performs spell checking on the given text using a combination of a
        language model and error models based on insertion, deletion, substitution,
        and transposition. It identifies potential spelling errors, generates candidate
        corrections, and selects the best correction based on a probabilistic evaluation
        of both the likelihood of the sentence and the likelihood of the error.
        Args:
            text (str): the text to spell check.
            alpha (float): the probability of keeping a lexical word as is.
        Return:
            A modified string (or a copy of the original if no corrections are made.)
        """

        if self.lm is None or self.error_tables is None:
            return text

        words = normalize_text(text).split()
        best_sentence = words.copy()
        for i, word in enumerate(words):
            if word not in self.lm.vocab:
                candidates = self.get_candidates(word)
                max_p_candidate = None
                max_p = float('-inf')

                for candidate, edit_type, letters in candidates:
                    candidate_sentence = words[:i] + [candidate] + words[i + 1:]
                    candidate_text = " ".join(candidate_sentence)
                    P_ngram = self.evaluate_text(candidate_text)
                    Prob_of_word = 1.0
                    if word != candidate:
                        error_score = 0.0
                        # Set the error from the error_table
                        if edit_type in self.error_tables:
                            error_table = self.error_tables[edit_type]
                            if letters in error_table:
                                error_score = error_table[letters]

                        # Compute the probability of the word based on error score
                        Prob_of_word = error_score if error_score > 0 else 1.0

                    # calculate the total probability
                    candidate_p = (1 - alpha) * P_ngram + (alpha) * Prob_of_word

                    # Select the candidate with the highest probability
                    if candidate_p >= max_p:
                        max_p = candidate_p
                        max_p_candidate = candidate

                # If a better candidate is found, replace the word in the best sentence
                if max_p_candidate:
                    best_sentence[i] = max_p_candidate

        return " ".join(best_sentence)

    def get_candidates(self, word):
        """Generates possible candidates for a misspelled word considering at most two edits."""
        # Define the alphabet to generate possible candidates
        en_letters = 'abcdefghijklmnopqrstuvwxyz'

        def first_change(word):
            """Generates edits with Levenshtein distance 1 and includes the type of edit."""
            splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
            deletes = [(L + R[1:], 'deletion', L[-1] + R[0] if len(L) > 0 else '#' + R[0]) for L, R in splits if R]
            transposes = [(L + R[1] + R[0] + R[2:], 'transposition', R[0] + R[1]) for L, R in splits if len(R) > 1]
            replaces = [(L + c + R[1:], 'substitution', R[0] + c) for L, R in splits if R for c in en_letters]
            inserts = [(L + c + R, 'insertion', L[-1] + c if len(L) > 0 else '#' + c) for L, R in splits for c in en_letters]
            return set(inserts + deletes + transposes + replaces)

        def seconed_change(word):
            """Generates edits with Levenshtein distance of 2 and includes the type of edit."""
            result = set()
            for target1, action, letters in first_change(word):
                for target2, _, _ in first_change(target1):
                    result.add((target2, action, letters))
            return result

        # Generate possible candidates within edit distance 1 and 2, and include the original word
        candidates = set(first_change(word)).union(seconed_change(word))
        candidates.add((word, 'no_edit', ''))

        final_candidates = []
        for candidate, edit_type, letters in candidates:
            if candidate in self.lm.vocab:
                final_candidates.append((candidate, edit_type, letters))

        return final_candidates

##########################----Inner class----##########################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text
            (on both word level and character level).
           It supports language generation and evaluation of a given string.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (n-gram). Defaults to 3.
                chars (bool): True if the model consists of ngrams of characters rather than word tokens.
                              Defaults to False.
            """
            self.n = n
            self.chars = chars
            self.model_dict = defaultdict(Counter)  # Stores n-gram counts
            self.model_dict_n_minus_one = defaultdict(Counter)
            # self.dict_prefix_ngram = defaultdict(list)  # Stores n-gram prefixes to next words mapping
            self.vocab = set()

        def build_model(self, text):
            """Populates the instance variable model_dict with n-grams.
            Args:
                text (str): the text to construct the model from.
            """
            text = normalize_text(text)
            tokens = self.tokenize(text)
            ngrams = list(nltk.ngrams(tokens, self.n))
            for ngram in ngrams:
                prefix = ngram[:len(ngram)-1]
                token = ngram[len(ngram)-1]
                self.model_dict[prefix][token] += 1
                self.vocab.update(ngram)

            if self.n > 1:
                for ngram in ngrams:
                    prefix1 = ngram[:len(ngram) - 2]
                    token1 = ngram[len(ngram) - 2]
                    self.model_dict_n_minus_one[prefix1][token1] += 1
                    self.vocab.update(ngram)

        def get_model_dictionary(self):
            """Returns the dictionary class object."""
            return dict(self.model_dict)

        def get_model_window_size(self):
            """Returns the size of the context window (the n in "n-gram")."""
            return self.n

        def generate(self, context=None, n=20):
            """
            Generates a string based on the language model using the specified context.
            This method generates text using the trained n-gram language model.
            It accepts an optional seed context to start the generation and generates
            `n` tokens (words or characters depending on the model type). If no context
            is provided, the method selects a random context from the language model.
            Args:
                context (str): A seed context to start the generated string from. Defaults to None.
                n (int): The length of the string to be generated.
            Returns:
                String. The generated text.
            Edge Cases:
            - If `n` is less than 1 or the language model (`model_dict`) is empty, the function
              returns the input `context` unchanged.
            - If no `context` is provided, the function selects a random seed context
              from the keys in the `model_dict`.
            - If the length of `context` exceeds `n`, the context is truncated and returned.
            - **Case when the n in the n-gram model is larger than the length of the context**:
              In this case, the function pads the context by generating tokens to fill the gap.
              It compares the prefix (last few tokens in the context) with available n-grams
              in the model. If an exact match is found, it predicts the next token based on
              probabilities. If not, a random token from the model's vocabulary is selected as a fallback.
              This ensures that even with short contexts, the generation process can proceed.
            """
            ####----edge cases----####
            # If n is less than 1 or the model_dict is empty, return the context as it is
            if n < 1 or self.model_dict is None:
                return context
            # If no context is provided, choose a random context from the model's keys
            if context is None:
                context = random.choice(list(self.model_dict.keys()))
            # Split context into words or characters depending on the model type (word-based or char-based)
            context_split = list(context) if self.chars else context.split()
            # If the context length exceeds n, truncate it to the required length
            if len(context_split) >= n:
                return "".join(context_split[:n]) if self.chars else " ".join(context_split[:n])

            # Initialize the generated sequence with the seed context
            generated_sequence = context_split[:]
            # Generate words/characters until the desired length is reached

            # # Case when the n in the n-gram model is bigger the length of the context
            if self.n > len(context_split):
                # space = self.n - len(context_split) - 1
                while len(generated_sequence) < n:
                    # Get the current prefix (the last self.get_model_window_size() tokens)
                    prefix = tuple(generated_sequence[-self.get_model_window_size():])
                    next_word = None
                    keys = {}
                    # Create a mapping of the model_dict keys to their sliced portions
                    for key in self.model_dict.keys():
                        keys[key] = key[-len(context_split):]  # Slice the last 'space' elements of each key
                    # Check if the prefix matches any of the values in the keys dictionary
                    options = keys.values()
                    if len(prefix) > len(context_split):
                        prefix = prefix[-len(context_split):]
                    if prefix in options:
                        # Find the original key where the sliced value matches the prefix
                        original_key = [k for k, v in keys.items() if v == prefix][0]
                        # Use the original key to get the corresponding prefix counter from model_dict
                        prefix_counter = self.model_dict[original_key]
                        # Get the next word based on the probabilities
                        words, frequencies = zip(*prefix_counter.items())
                        next_word = random.choices(words, weights=frequencies)[0]
                    # If next_word is found, append it to the generated sequence
                    if next_word:
                        generated_sequence.append(next_word)
                    if next_word is None:
                        next_word = random.choice(list(self.vocab))
                        generated_sequence.append(next_word)

            # Base case
            while len(generated_sequence) < n:
                # Get the current prefix: the last n-1 elements of the generated sequence
                prefix = tuple(generated_sequence[-self.get_model_window_size():])
                # Try to find a matching next word by progressively shortening the prefix
                next_word = None
                for i in range(len(prefix)):
                    current_prefix = prefix[i:]  # Shorten the prefix step by step
                    if current_prefix in self.model_dict:
                        # Found a matching prefix in the model_dict
                        prefix_counter = self.model_dict[current_prefix]
                        words, frequencies = zip(*prefix_counter.items())
                        next_word = random.choices(words, weights=frequencies)[0]
                        break  # Exit the loop once a match is found
                # If no match is found, pick a random word from the model's vocabulary as a fallback
                if next_word is None:
                    next_word = random.choice(list(self.vocab))
                # Append the predicted next word to the generated sequence
                generated_sequence.append(next_word)
            # Join the sequence into a string and return
            return "".join(generated_sequence) if self.chars else " ".join(generated_sequence)

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text.
            Args:
                text (str): Text to evaluate.
            Returns:
                Float. The (log) probability.
            """
            text = normalize_text(text)
            tokens = self.tokenize(text)
            ngrams = list(nltk.ngrams(tokens, self.n))
            if text is None:
                return 0.0
            # Split the text into tokens or characters
            if not self.chars:
                split_text = text.split()
            else:
                split_text = [char for char in text]
            # Return 0.0 if the text is shorter than the n-gram size
            if len(split_text) < self.n:
                return 0.0

            log_prob = 0
            for ngram in ngrams:
                prefix = ngram[:-1]  # Get the prefix (n-1 words)
                token = ngram[-1]  # Get the next word (the last word in the n-gram)
                prefix1 = ngram[:-2]  # Get the prefix (n-2 words)
                token1 = ngram[-2]  # Get the next word (the last word in the n-1-gram)

                # Check if the prefix is in the model_dict and has the token
                if prefix in self.model_dict and token in self.model_dict[prefix]:
                    token_count = self.model_dict[prefix][token]
                    prefix_count = self.model_dict_n_minus_one[prefix1][token1]
                    prob = token_count / prefix_count
                    log_prob += math.log(prob)
                else:
                    log_prob += math.log(self.smooth(ngram))
            return log_prob


        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.
            Args:
                ngram (tuple): the ngram to have its probability smoothed
            Returns:
                float. The smoothed probability.
            """
            prefix = ngram[:-1]
            token = ngram[-1]
            token_count = self.model_dict[prefix][token]
            prefix_count = sum(self.model_dict[prefix].values())
            vocab_size = len(self.vocab)
            smoothed_prob = (token_count + 1) / (prefix_count + vocab_size) # Applying Laplace smoothing
            return smoothed_prob

        def tokenize(self, text):
            """Tokenizes text into words or characters depending on the model."""
            if self.chars:
                return list(text)
            return text.split()


##########################----global functions----##########################
def normalize_text(text):
    """Returns a normalized version of the specified string.
       Args:
           text (str): the text to normalize.
       Returns:
           string. the normalized text.
    """
    text = text.lower()
    text = text.replace('<s>', ' ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\n', '.')
    text = text.replace('\t', ' ')
    text = text.replace('_', ' ')
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'(?<=\w)\.(?=\w)', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



