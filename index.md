# Leveraging Model (Un)certainty
_Can a model "know" when it's predictions are incorrect or not?_ 

This work explores model uncertainty scoring for bias/variance and error analysis, using Machine Translation (NN Attention) as a use case.  

![Image](https://github.com/mahaley22/Uncertainty-Sampling/blob/master/Keep%20your%20mask%20on!.PNG?raw=true)  ![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Aslightconfusion.PNG?raw=true) 

The above two examples are teaser examples of model uncertainty for a couple of translated sentences.  Basically, the higher the uncertainty bars for a given token, the higher the uncertainty. The first shows an acceptable translation that wasn't too confident: note the "put/keep" uncertainty, and that "mask" shows even higher uncertainty (but I guess we're all still getting used to the mask thing).  The second shows low uncertainty despite the "puzzled"/"confused" switch.  I guess it's certain that we're confused!  :confused:

The notebook in this repo demonstrates that not only is uncertainty positively correlated with mismatches from the target translation, but also correlated with mismatches that are actually True Negatives, i.e. not acceptable alternate translations.  This work was inspired in part by <a href="https://www.manning.com/books/human-in-the-loop-machine-learning">*Human-in-the-Loop Machine Learning* by Robert Munro © 2020 </a>

## Introduction

1) "Where": more automatically detecting errors on "in the wild" unlabelled sets and production

2) "What": targetted sampling for enhancing training or dev/test sets

3) "Why": "Why did the model get this wrong?"  (everybody's favorite question in ML)

4) "When": detecting model drift by measuring aggregate uncertainty

5) "How":  to improve the model as quickly and cost-effectively as possible?

As a practitioner of Applied ML for a number of years now, I'm not alone in having these questions posed to me at various times (by myself, or worse  by others like internal stakeholders or customers).  So I wanted to try NN model uncertainty to see if it can be useful, even if the model itself is (initially) weaker than we would like.  In fact, that's the whole point: we want to improve the model using all the means we have at our disposal: hyperparameter tuning, training, etc. as part of the Active Learning iterative process.  Bear in mind that information from inside the model is certainly not the only tool to leverage for things like Active Learning.

We all know the aphorism: all models are wrong, but some are useful. Raw accuracy on seen or unseen data is just one metric of usefulness.   Besides, no matter how accurate (and however imperfectly that in itself might be measured), ML models can be subject to data drift, concept drift, or generalization issues and even outright bugs on unseen data.  Following software engineering best practices, how can we apply practices like unit testing, bug detection, early exception detection and CI/CD in production be accomplished when working with Neural Nets, which are nondeterministic, nested non-linear structures optimized for accuracy, not scrutability?

Active Learning practice attempts to answer some of these questions: How do we optimize for humans in the loop?  How much hand-labeled training do you need up front and on an ongoing basis?  What types of errors can be discovered and analyzed earlier and easier?  Which subsets of outputs should humans be targetting for possible correction and training?  Active Learning can employ metrics derived from the models themselves, and also external to the model.  This notebook looks at some in-model metrics. 

## Method
This notebook trains a Attention-based Encoder-Decoder MT model, and teases out several in-model metrics from the last layer (logits) of the decoder outputs.
Instead of looking at Attention plots, we plot and measure several Uncertainty Scores, which are simply a measure of finding predictions that are near a decision boundary.  This is done at a couple of levels:

1) Explore the data by aggregate uncertainty for analyzing avoidable bias, variance, and "most uncertain" Uncertainty Sampling.

2) Use Uncertainty plots in order to drive error analysis and interpretibility of results.  Uncertainty measureed at the token/word level can yield information

The choice to build a toy MT model affords some fun and interesting examples of how for a given translation output the model may be trying to say something about its own uncertainty.  The choice of MT affords a look at not just on the overall aggregate output sentence uncertainty, but on the constituent tokens which can lend to some interpretibility.  

The original model's output just selected the maximum raw score (logits) from each timestamp.  Afer that (i.e. post-optimization) this notebook applies softmax normalization to these scores, so that for a given timestamp, all the scores add up to one.  Then, this notebook uses the normalized softmax scores for 3 different measures of uncertainty.

(Note: the first third or so of this notebook is all about setting up the training and model and actually doing the training, and was adapted and lightly modified from a reference google demo notebook.  Also for reasons having to do with having an in-house native speaking spouse, this happens to use Hebrew as the source language, but hopefully this shouldn't matter since most of the specific examples just compare the English outputs.  Remember, Google Translate is still your friend for comparison!)

## Metrics
In the notebook there are many examples of different metrics of the translation (from left to right):

_Competing information:_

1) "Least Confidence" absolute difference between the top score and 1 (this is a somewhat confusing term having to do with the sampling method, i.e. picking the "least confident" unlabelled data ranked in descending order)

2) Margin of Confidence (difference between the top score and its runner-up). 

3) Overall uncertainty due to competing information (combining a) and b)), and that is what's used in the rest of the notebook for competing information analysis.

_Level of information:_

4) Level of information (raw logit scores)

_Combining Competing information and Level of Information:_

5) Level of information combined with competing information (c)

Mainly this notebook uses uncertainty due to the combined competing information (c) metric above.  However, we that a lack of information could be a used as another selection criterion in some cases.

No matter what the uncertainty score used, let's say a) or b) above instead of c), or even using a different custom softmax for scoring itself, *can* change the overall uncertainty rankings of multiple outputs.    There is nothing probabilistic or magical about softmax for this purpose, but it is terribly useful for normalizing for uncertainty and also having the raw logits available for more information. 

That all the scores add up to 1 leads some to that "probabilistic" confusion, and also potential confusion (pardon the pun) among terms like "uncertainty" and "confidence" and "probability".  Especially since the model used conditional (discriminative) model, a given uncertainty score of 0.6  is *not* an indication that there is a 60% probability that this is wrong.  In fact, the point is to try different metrics in order to gain more insights in our error analysis by uncertainty, as we do in the notebook.  Keep in mind that different metrics can yield different rankings of uncertainty, which with enough examples *should* clear one's mind of some probabilistic delusions.  :smiley:

## Some Uncertainty sampling classes
One challenge with this datset is that there is usually exactly one reference translation.  As a crude start then we can simply consider all word-for-word matches with the single target as correct, and then explore the mismatches for errors.  (In other words, there are no False Positives among Matches)  Let's consider positives and negatives in the context of both uncertainty and matching:

1) *True Negatives with respect to mis-matches and uncertainty* (this is easy as there are many of them where the translation is just wrong and there is high aggregate uncertainty):

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Mistranslation1.PNG?raw=true)

2) *False Negatives with respect to mis-matches* (good alternate translations with low uncertainty): when we explore this seeming variance in the low uncertainty validation set, we often find acceptable alternate translations, e.g. replacement with an synonymous word or words like "perplexed/confused" (above example), "this/that", "keep" vs. "put" (image above).  These acceptable replacements can be recast as *True Positives* w.r.t. uncertainty scores and added to our reference translations ground truth.

Some interesting outlier cases:
- Partially FN, partially TN hybrid, where a whole subclause can be correct and then another goes off the rails: 
    
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Long%20sentence%20started%20out%20ok.PNG?raw=true)

- Mislabelled ground truth!  Usually we can live with these random labelling errors in Deep Learning training with lots of data, unless there is a more systematic error underlying these.  However, this is more important for dev/test sets (Hebrew is *not* ambiguous between animate and inanimate objects!):
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Wrong%20ground%20truth!.PNG?raw=true)

3) *False Negatives w.r.t. uncertainty* can arise, like the "mask" example above, or here (flight/hotel), which offers up a another class of potential errors (or where the model more or less got "lucky") to work on for model refinement/training:

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Flight%20vs.%20Hotel.PNG?raw=true)

4) *False Positives w.r.t. uncertainty*: mis-translations with low uncertainty, we do find a few in our exploration of underfitting of the training set and variance of the validation set, again offering up samples we might not have considered otherwise for training or model refinement.  These will be harder to find with this method but if they do arise in a low uncertainty context, this can be possibly prioritized for discovering training or model flaws.

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Believable%20versus%20reliable.PNG?raw=true)

5) *True Positives w.r.t. uncertainty*: these would be the many examples of matches with low uncertainty in the notebook.

For error analysis, its interesting to discover using uncertainty which individual words/tokens will have high uncertainty, often indicating at the token level where the translation went awry. This is often indicated by the "runner-up" (2nd highest scoring) translation for that token(s).  This could be of help for humans in the loop correcting these translations using a manual interface, for example.  Thus the outright wrong results are at least somewhat interpretible.  Besides training on more data, its possible that knowing more about the model confusion info itself to try different things, like increasing the beam width.

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Runner-up%20was%20correct!.PNG?raw=true)

### A note on Level of Information:
So far we've seen that competing information using scores that add up to one using softmax.  Raw logits on the other hand, can indicate the level of information in a particular token or sentence.  While using softmax is very convenient, it is at least interesting to look at the highly variant raw logit scores from the model's last layer:

1. Low Information examples which can indicate problematic or incomplete translations:
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Low%20Information%201.PNG?raw=true)

2. High information examples indicating possibly fluent mis-translations: 
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/High%20Information%20and%20Competition%204%20-%20He%20caught%20a%20Poet!.PNG?raw=true)
High Information combined with high uncertainty for that token(s) can yield some pretty egregious replacements!


## Aggregate Results

With some variation in the ratios, the density of raw mis-matches (True or False Negatives) is positively correlated with uncertainty.  For example, in one run:

**33.6%** of the potential errors are found by **20.0%** of the target sentences with the highest uncertainty score.

Results may vary somewhat depending on the training/dev sets, but always the following holds true:

a) Not only is the distribution of mismatches (presumptive Negatives) skewed toward the high uncertainty percentiles, moreover

b) The distribution of *mistranslations* (True Negatives) is markedly skewed toward the high uncertainty percentiles, whereas the *alternate acceptable translations* have lower uncertainty.


## Conclusions
This is an illustration (with graphs even!) of using uncertainty in ML, using using a MT system as an example.  These types of exploration can lead to better error analysis and Active Learning:
1. This work shows some tooling for exploration and sampling with for underfitting and variance, using uncertainty.  Even if a translation output matches a reference translation, uncertainty can be used for analysis and sampling.
2. Aggregate Uncertainty is positively correlated with True Negatives; both as an aid for human correct and for the purposes of error analysis and iterating on the model itself.
3. Interpretibility is aided to some extent with word/token level uncertainty scoring, and score graphs and "runner-up" token translations logging.
4. This type of analysis can be a part of a virtuous cycle of model-based Active Learning, with new fixed reference translations to improve our ground truth for training/dev/test, as well as prioritizing the team's model iterations.
5. Exploring the space of different scoring and aggregation methods would seem to be worthwhile for future work.  For example, instead of the means, use the minimums as the aggregation score for sentences.

Feel free to explore some interesting (and funny) translations here: <a href="https://github.com/mahaley22/Uncertainty-Scoring/tree/gh-pages/images">More examples</a>
Also feel free to download and play with this notebook for other languages and metrics!

## References:
1. <a href="https://www.manning.com/books/human-in-the-loop-machine-learning">*Human-in-the-Loop Machine Learning* by Robert Munro © 2020 </a>
2. <a href="https://www.aclweb.org/anthology/W19-8671.pdf">*Modeling Confidence in Sequence-to-Sequence Models* Niehues, Pham 2019</a>
