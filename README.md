# Spell-Checker-using-a-Noisy-Channel-Model

## Overview

This project implements a spell checker using a Noisy Channel Model. It corrects spelling errors in text by leveraging an n-gram language model and common error type matrices. The model identifies potential corrections and ranks them based on probability.

## Features

Handles Non-word and Real-word Errors: Identifies misspelled words and suggests corrections based on context.

Uses a Noisy Channel Model: Incorporates a probabilistic approach to maximize the likelihood of the correct word.

Supports Up to Two Edits: Detects insertion, deletion, substitution, and transposition errors.

Context-Aware Correction: Uses an n-gram language model to assess corrections based on surrounding words.

Customizable Parameters: Allows tuning of alpha (probability weight) for better correction accuracy.

## Usage

**1. Training the Language Model**

Before using the spell checker, train the language model on a corpus:

```python
from language_model import Language_Model

lm = Language_Model(n=3)  # Trigram model
lm.train('path/to/corpus.txt')
```

**2. Initializing the Spell Checker**
```python
from spell_checker import Spell_Checker

spell_checker = Spell_Checker(lm, error_tables)
```

**3. Correcting a Sentence**
```python
sentence = "Ths is an exmple sentnce."
corrected_sentence = spell_checker.correct(sentence)
print(corrected_sentence)
```

**4. Evaluating Sentence Probability**
```python
prob = lm.evaluate_text("This is an example sentence.")
print(f"Sentence probability: {prob}")
```

## File Structure:
```python
├── spell_checker.py       # Implements the Spell_Checker class
├── language_model.py      # Implements the n-gram Language_Model class
├── error_tables.py        # Stores common spelling error probabilities
├── corpus.txt            # Sample training corpus
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
```

## Customization:

Modify n-gram size: Adjust n in Language_Model for different context sizes.

Change alpha: Fine-tune the probability weight in the spell_check function.

Expand the corpus: Improve accuracy by training on a larger dataset.

Set chars=True for character-level language model
