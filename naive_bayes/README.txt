To train model, run "python nb_train.py"
To test model, run "python nb_test.py"
To test model on handcrafted adversarial input, run "python nb_test.py --handcrafted"
To test model on training set, run "python nb_test.py --training"
To test model on basic naive-Bayes-targeted adversary, run "python nb_test.py --basic_adversary"
    This adversary replaces the first word of each positive training review with one of the (randomly
    selected) top-5 negative-leaning words (and vice versa for negative training reviews).
To test model on strongest-word-swap adversary, run "python nb_test.py --strongest_word_adversary"
    This adversary replaces the word that most strongly aligns with the correct classification
    with a word that most strongly aligns with the incorrect classification.
To test model on strongest-3-word-swap adversary, run "python nb_test.py --strongest_3_word_adversary"
    This adversary replaces the three words that most strongly align with the correct classification
    with three words that most strongly align with the incorrect classification.
To test model on strongest-5-word-swap adversary, run "python nb_test.py --strongest_5_word_adversary"
    This adversary replaces the five words that most strongly align with the correct classification
    with five words that most strongly align with the incorrect classification.
To test model on JSMA adversary, run "python nb_test.py --JSMA_adversary"
    This adversary replaces words at the end of the review with adversarial words based on the JSMA
    method. It replaces words until the review becomes adversarial or the 35-word limit is reached.

To get most positive-leaning and most negative-leaning words, run 
    "python model_analysis.py --number k" where k is the number of top
    words you'd like to see. (e.g. "python model_analysis.py --number 5")

No need to train model, though, because model weights are stored in model_data folder.
