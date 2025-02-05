from main import *
from spelling_confusion_matrices import error_tables
from spelling_confusion_matrices import *


b = open("big.txt")
data = b.read()
normalized_corpus = normalize_text(data)

lm1 = Spell_Checker.Language_Model(n=3, chars=False)   # Set chars=True for character-level language model
lm1.build_model(normalized_corpus)

print(lm1.evaluate_text("the rare and radiant"))
print(lm1.evaluate_text("the rare and radiant"))


print(lm1.generate(context="gutenberg ebook of",n=10))
print(lm1.generate(context="gutenberg book of",n=4))
print(lm1.generate(context="His manner was not ",n=5))
print(lm1.generate(context="I was half-dragged up to the altar", n=9))
print(lm1.generate(context="", n=4))


lm2 = Spell_Checker.Language_Model(n=6, chars=True)
lm2.build_model(normalized_corpus)
print(lm2.generate(context="His manner was not ",n=70))
print(lm2.generate(context="", n=4))

add2 = Spell_Checker.Language_Model()

lm = Spell_Checker.Language_Model(n=3, chars=False)   # Set chars=True for character-level language model
lm.build_model(normalized_corpus)
spell_checker = Spell_Checker()
spell_checker.add_language_model(lm)
spell_checker.add_error_tables(error_tables)
input_text = "the project gutenberg book"
alpha = 0.1 # Adjust the alpha value based on your preference (higher alpha values give more weight to the original word)
print("==================================")
corrected_text = spell_checker.spell_check(input_text, alpha)
print("Original text:", input_text)
print("Corrected text:", corrected_text)
print("==================================")
input_text = "This is an exmple sentence with errors"
alpha = 0.95 # Adjust the alpha value based on your preference (higher alpha values give more weight to the original word)
corrected_text = spell_checker.spell_check(input_text, alpha)
print("Original text:", input_text)
print("Corrected text:", corrected_text)
print("==================================")
input_text = "you are the royal ling of england"
alpha = 0.9 # Adjust the alpha value based on your preference (higher alpha values give more weight to the original word)
corrected_text = spell_checker.spell_check(input_text, alpha)
print("Original text:", input_text)
print("Corrected text:", corrected_text)
print("\n==================================")
input_text = "haunts of the whalec"
alpha = 0.95
corrected_text = spell_checker.spell_check(input_text, alpha)
print("Original text:", input_text)
print("Corrected text:", corrected_text)
print("==================================")


test_cases = [
    ("beutiful", "beautiful"),
    ("baautiful", "beautiful"),
    ("bautiful", "beautiful"),
    ("beutifull", "beautiful"),
    ("beuatiful", "beautiful"),
    ("inteligence", "intelligence"),
    ("intellgence", "intelligence"),
    ("intelligenc", "intelligence"),
    ("intellignce", "intelligence"),
    ("intelliegnce", "intelligence"),
    ("intelilgence", "intelligence"),
    ("recomend", "recommend"),
    ("reccommend", "recommend"),
    ("reccomend", "recommend"),
    ("reecommand", "recommend"),
    ("envirnoment", "environment"),
    ("environmet", "environment"),
    ("envronment", "environment"),
    ("envirnomnt", "environment"),
    ("enviroonmet", "environment"),
    ("acess", "access"),
    ("acount", "account"),
    ("acomplish", "accomplish"),
    ("adres", "address"),
    ("adresable", "addressable"),
    ("definate", "definite"),
    ("definitly", "definitely"),
    ("definitiv", "definitive"),
    ("recive", "receive"),
    ("recived", "received"),
    ("recieving", "receiving"),
    ("ling", "king"),
]

# test_cases = [
#     ('acess', 'access'),
# ]

# Initialize counters
intx = 0
correct_count = 0
total_count = len(test_cases)
# Loop through test cases and check the spell check results
for i in test_cases:
    a = spell_checker.spell_check(i[0], alpha=0.95)
    is_correct = a == i[1]
    # Print the details of each test case
    print(intx, i, "------->", a, " --> ", is_correct)
    # Increment correct count if the result is correct
    if is_correct:
        correct_count += 1
    intx += 1
# Calculate success rate
success_rate = correct_count / total_count * 100
# Print the success rate
print(f"Success rate: {correct_count}/{total_count} ({success_rate:.2f}%)")

print("==================================")
input_text = "The Projecc Gutenberg EBook of The Adventures"
alpha = 0.95 # Adjust the alpha value based on your preference (higher alpha values give more weight to the original word)
corrected_text = spell_checker.spell_check(input_text, alpha)
print("Original text:", input_text)
print("Corrected text:", corrected_text)


text1 = """
U.S.A $12.40 Once upon a midnight dreary, while pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
    While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.
“’Tis some visitor,” I muttered, “tapping at my chamber door—
            Only this and nothing more.”


    Ah, distinctly I remember it was in the bleak December;
And each separate dying ember wrought its ghost upon the floor.
    Eagerly I wished the morrow;—vainly I had sought to borrow
    From my books surcease of sorrow—sorrow for the lost Lenore—
For the rare and radiant maiden whom the angels name Lenore—
            Nameless here for evermore.
"""


lm = Spell_Checker.Language_Model(n=3, chars=False)
lm.build_model(text1)
print(lm.generate("tis some visitor", n=20))
print(lm.evaluate_text("my books legs"))
print(lm.evaluate_text("the rare and radiant"))
print(lm.evaluate_text("rapping at my chamber door."))
print(lm.evaluate_text("rapping my chambel door. _mohana lok__"))



C = open("corpus.txt")
dataC = C.read()
normalized_corpusC = normalize_text(dataC)
lmC = Spell_Checker.Language_Model(n=4, chars=False)
lmC.build_model(normalized_corpusC)
print(lmC.generate(context="laid open the haunts",n=5))
spell_checker = Spell_Checker()
spell_checker.add_language_model(lmC)
spell_checker.add_error_tables(error_tables)
input_text = "haunts of the whalec"
alpha = 0.95
corrected_text = spell_checker.spell_check(input_text, alpha)
print("Original text:", input_text)
print("Corrected text:", corrected_text)
print("==================================")
print(lmC.evaluate_text("haunts of the whale"))


TEXTX= ("<s> emma by jane austen 1816 <s> volume i <s> chapter i <s> "
        "emma woodhouse handsome clever and rich with a comfortable home and happy disposition seemed to unite some "
        "of the best blessings of existence and had lived nearly twenty one years in the world with "
        "very little to distress or vex her <s> she was the youngest of the two daughters of a most "
        "affectionate indulgent father and had in consequence of her sister s marriage been mistress of "
        "his house from a very early period <s> her mother had died too long ago for her to have more than "
        "an indistinct remembrance of her caresses and her place had been supplied by an excellent woman as "
        "governess who had fallen little short of a mother in affection <s> sixteen years had miss taylor been in mr "
        "woodhouse s family less as a governess than a friend very fond of both daughters but particularly of emma <s> "
        "between _them_ it was more the intimacy of sisters <s> even before miss taylor")

print(normalize_text(TEXTX))