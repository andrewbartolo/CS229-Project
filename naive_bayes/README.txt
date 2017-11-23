To train model, run "python nb_train.py"
To test model, run "python nb_test.py"
To test model on handcrafted adversarial input, run "python nb_test.py --handcrafted"
To test model on basic naive-Bayes-targeted adversary, run "python nb_test.py --basic_adversary"
    This adversary replaces the first word of each positive training review with one of the (randomly
    selected) top-5 negative-leaning words (and vice versa for negative training reviews).

To get most positive-leaning and most negative-leaning words, run 
    "python model_analysis.py --number k" where k is the number of top
    words you'd like to see. (e.g. "python model_analysis.py --number 5")

No need to train model, though, because model weights are stored in model_data folder.
