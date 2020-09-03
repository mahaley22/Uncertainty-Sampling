# Leveraging Model (Un)certainty
_Can a model "know" when it's predictions are incorrect or not?_ 

This work explores model uncertainty scoring for bias/variance and error analysis, using Machine Translation (NN Attention) as a use case.  

![Image](https://github.com/mahaley22/Uncertainty-Sampling/blob/master/Keep%20your%20mask%20on!.PNG?raw=true)  ![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Aslightconfusion.PNG?raw=true) 

The above two examples are teaser examples of model uncertainty for a couple of translated sentences.  Basically, the higher the uncertainty bars for a given token, the higher the uncertainty. The first shows an acceptable translation that wasn't too confident: note the "put/keep" uncertainty, and that "mask" shows even higher uncertainty (but I guess we're all still getting used to the mask thing).  The second shows low uncertainty despite the "perplexed"/"confused" switch.  I guess it's certain that we're confused!

The notebook in this repo demonstrates that not only is uncertainty positively correlated with mismatches from the target translation, but also correlated with mismatches that are actually True Negatives, i.e. not acceptable alternate translations.  This work was inspired in part by Human-in-the-Loop Machine Learning by Robert Munro © 2020

## Intro/Summary

    1) "Where": more automatically detecting errors on "in the wild" unlabelled sets and production

    2) "What": targetted sampling for enhancing training or dev/test sets

    3) "Why": "Why did the model get this wrong?"  (everybody's favorite question in ML)

    4) "When": detecting model drift by measuring aggregate uncertainty

    5) "How":  to improve the model as quickly and cost-effectively as possible?

As a practitioner of Applied ML for a number of years now, I'm not alone in having these questions posed to me at various times (by myself, or worse  by others like internal stakeholders or customers).  So I wanted to try NN model uncertainty to see if it can be useful, even if the model itself is weaker than we would like.  In fact, that's the whole point: we want to improve the model using all the means we have at our disposal: hyperparameter tuning, training, etc. as part of the Active Learning iterative process.  Bear in mind that information from inside the model is certainly not the only tool to leverage for things like Active Learning.

So we all know the saying: all models are wrong, but some are useful. Raw accuracy using a single metric is usually not the only measure of a model.   So how else can a model be useful?

No matter how accurate or *good* one's model is, not only will there will always be things like data drift, concept drift, or simply generalization issues on things the model hasn't seen or tested for before (see checklist paper).  As in other types of software, how does CI/CD and maintenance come in, and how does a model help/hinder that?

Active Learning comes in to answer some of these questions: How do we optimize for humans in the loop?  How much hand-labeled training do you need up front and on an ongoing basis?  F which outputs should a human have a look at outputs for possible correction and training?  Active Learning has a lot of tools in the toolkit for sampling and iterating, both within the models and outside them.  this model will look at just 2 or 3 in-model metrics to show that 

And yet, by definition the goal of optimizing any machine learning model is not primarily (at least when we're talking about conditional modeling, like in Machine Translation) in the business of generating accurate "probabilities" or confidences for those predictions.  And how explainable/interpretible are the results?

And worse yet, how can one even tease out such information of a deep learning algorithm, which by its nature is a nested non-linear structure? 
This notebook is an example of a NN model for MT that can fairly easily yield useful information and metrics that are helpful for error analysis and Active Learning.

## Method
This notebook trains a sequence to sequence (seq2seq) model for machine translation, using Attention. However, instead of looking at the Attention plots, we plot and measure Uncertainty scores, which are simply a measure of finding predictions that are near a decision boundary.

1) Explore the data by aggregate uncertainty for analyzing avoidable bias, variance, and sampling for Active Learning
2) Use Uncertainty plots in order to drive analysis and interpretibility of results.

For this notebook I've chosen a toy Machine Translation example, which affords some fun and interesting examples of how for a given translation output the model may be trying to say something about its own uncertainty.  The choice of MT affords a look at not just on the overall aggregate output sentence uncertainty, but on the constituent tokens which can lend to some interpretibility. 

So in an active learning cycle using model uncertainty sampling, you want to rank the "most uncertain" outputs (in this case, sentences) in order to gain a better understanding prioritization for error analysis, human review and possible (re)training, as well as iterating on the model itself, e.g. hyperparameter tuning.  

The original model's output just selected the maximum raw score (logits) from each timestamp.  Afer that (i.e. post-optimization) this notebook softmax normalization to these scores, so that for a given timestamp, all the scores add up to one.  Then, this notebook uses the normalized softmax scores for three somewhat different measures of uncertainty:
a) "Least Confidence" absolute difference between the score and 1 (this is a somewhat confusing term having to do with the sampling method, i.e. picking the "least confident" unlabelled data ranked in descending order)

b) Margin of Confidence (difference between the top score and its runner-up). 

c) is simply an aggregation of the first two (by multiplying), and that is what's used in the rest of the notebook for analysis.

No matter what the uncertainty score used, let's say a) or b) above instead of c), or even using a different custom softmax for scoring itself, *can* change the overall uncertainty rankings of multiple outputs.    That Munro book I cite at the top of the nb emphasizes that there's nothing probabilistic or magical about softmax for this purpose, but its especially useful for uncertainty when softmax is not originally used as part of the optimization of the final layer.  That all the scores add up to 1 leads some to that "probabilistic" confusion, but it doesn't matter.

Here I'll just pause to note that the potential confusion (pardon the pun) among terms like "uncertainty" and "confidence" and "probability".  That all the scores add up to 1 leads some to that "probabilistic" confusion.  Especially since the model used conditional (discriminative) model, a given uncertainty score let's say 0.6, is *not* an indication that there is a 60% probability that this is wrong.  In fact, the point is to try different metrics in order to gain more insights in our error analysis by uncertainty, as we do in the notebook.  Keep in mind that different metrics can yield different rankings of uncertainty, which with enough examples *should* clear one's mind of any probabilistic delusions. ![Image](data:image/jpeg;base64,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)

(Note: the first third or so of this notebook is mostly setting up the training and model and actually doing the training using an Attention model, adapted and slightly modified from a reference google demo notebook.  Also for reasons having an in-house native speaking spouse, this happens to use Hebrew as the source language, but shouldn't matter since most of the specific examples just compare the English outputs.  Remember, to verify Google Translate is your friend!)

## Some Uncertainty sampling classes
One challenge with this datset is that there is usually exactly one reference translation.  As a crude start then we can simply consider all word-for-word matches with the single target, andt otherwise these are mismatches.  (In other words, there are no False Positives among Matches)  Let's consider positives and negatives in the context of both uncertainty and matching:

1) *True Negatives with respect to mis-matches* (this is easy as there are many of them):

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Mistranslation1.PNG?raw=true)

2) *False Negatives with respect to mis-matches* (good alternate translations with low uncertainty): when we explore this seeming variance in the low uncertainty validation set, we often find acceptable alternate translations, e.g. replacement with an synonymous word or words like "perplexed/confused" (above example), "this/that", "keep" vs. "put" (image above).  These acceptable replacements can be recast as *True Positives* w.r.t. uncertainty scores and added to our reference translations ground truth.

Some interestomg outlier cases:
- Partially FN, partially TN hybrid, where a whole subclause can be correct and then another goes off the rails: 
    
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Long%20sentence%20started%20out%20ok.PNG?raw=true)

- Mislabelled ground truth!  Usually we can live with these random labelling errors in Deep Learning training with lots of data, unless there is a more systematic error underlying these.  However, this is more important for dev/test sets (Hebrew is *not* ambiguous between animate and inanimate objects!):
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Wrong%20ground%20truth!.PNG?raw=true)

3) *False Negatives w.r.t. uncertainty* can arise, like the "mask" example above, or here (flight/hotel), which offers up a another class of potential errors (or where the model more or less got "lucky" to work on for model refinement/training:

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Flight%20vs.%20Hotel.PNG?raw=true)

4) *False Positives w.r.t. uncertainty*: mis-translations with low uncertainty, we do find a few in our exploration of underfitting of the training set and variance of the validation set, again offering up samples we might not have considered otherwise for training or model refinement.  These will be harder to find with this method but if they do arise in a low uncertainty context, this can be possibly prioritized for discovering training or model flaws.

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Believable%20versus%20reliable.PNG?raw=true)

5) *True Positives w.r.t. uncertainty*: these would be the many examples of matches with low uncertainty in the notebook, .

For error analysis, its interesting to discover using uncertainty which individual words/tokens will have high uncertainty, often indicating at the token level where the translation went awry. This is often indicated by the "runner-up" (2nd highest scoring) translation for that token(s).  This could be of help for humans in the loop correcting these translations using a manual interface, for example.  Thus the outright wrong results are at least somewhat interpretible.  Besides training on more data, its possible that knowing more about the model confusion info itself to try different things, like increasing the beam width.

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Runner-up%20was%20correct!.PNG?raw=true)


## Aggregate Results
With some variation in the ratios, the density of raw mis-matches (True or False Negatives) is positively correlated with uncertainty.  In one run 

**32.1%** of the non-matches (potential errors) are found by **10.0%** of the target sentences with the highest uncertainty score.

But that is not usually the case.  
But it usually holds true that both the distribution of mis-matched bad translations is weighted in the high uncertainty.  


But then with a quick tool for exploration, it's easy to examine the presumptive Negatives to see if they are True or False Negatives (bad vs. good translations).  There, we find a marked concentration of True Negatives in high uncertainty.

mismatched but "good" mismatched but "bad" - low confidence, so it makes sense for example in an Active Learning scenario to go after the low confident mismatches first.  Put another way, our True Negatives are overwhelmingly concentrated at the high uncertainty percentiles.

## Conclusions:
This is an illustration (with graphs even!) of using uncertainty in ML, using using a MT system as an example.  These types of exploration can lead to better error analysis and Active Learning:
1. Uncertainty is positively correlated with True Negatives; both as an aid for human correct and for the purposes of error analysis and iterating on the model itself.
2. Interpretibility is aided to some extent with score graphs and "runner-up" token translations
4. We present a bit of tooling for (better, easier?) exploration and sampling with for underfitting and variance.  Even if a translation output matches a reference translation, uncertainty can be used for analysis and sampling.
5. Part of an early stage of a virtuous cycle of model-based Active Learning, with new fixed reference translations to improve our ground truth for training/dev/test, as well as prioritizing the team's model iterations.
6. Exploring the space of scoring and aggregation methods would seem to be worthwhile.  For example, instead of the means, use the minimums as the aggregation score for sentences.

## References:
1. *Human-in-the-Loop Machine Learning* by Robert Munro © 2020
2. *Modeling Confidence in Sequence-to-Sequence Models* Niehues, Pham 2019 



```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mahaley22/Uncertainty-Sampling/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### References

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
