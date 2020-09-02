# Leveraging Model (Un)certainty
_Or: : can a model "know" when it's predictions are incorrect or not?_ 

This work explores model uncertainty scoring in Neural Net Machine Learning error analysis, using Machine Translation (Attention) as a use case.  

![Image](https://github.com/mahaley22/Uncertainty-Sampling/blob/master/Keep%20your%20mask%20on!.PNG?raw=true&width="500"&height="450")  ![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Aslightconfusion.PNG?raw=true&width="400"&height="450")

The above two examples are teaser examples uncertainty for a couple of translated sentences.  Basically, the higher the uncertainty for given output token. The first shows an acceptable translation that wasn't too confident: note the "put/keep" uncertainty, and that "mask" shows even higher uncertainty (but I guess we're all still getting used to the mask thing).  The second shows low uncertainty despite the "perplexed"/"confused" switch.  I guess it's certain that we're confused!

The notebook in this repo demonstrates that not only is uncertainty positively correlated with mismatches from the target translation, but also correlated with mismatches that are actually True Negatives, i.e. not acceptable alternate translations.

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

And yet, by definition the goal of optimizing any machine learning model is not primarily (at least when we're talking about conditional modeling, like in Machine Translation) in the business of generating accurate "probabilities" or confidences for those predictions.  And how explainable/interpretibleare the results?

And worse yet, how can one even tease out such information of a deep learning algorithm, which by its nature is a nested non-linear structure? 
This notebook strongly indicates a NN model for MT can yield useful information and metrics that are helpful for analysis and Active Learning.

## Methods used
This notebook trains a sequence to sequence (seq2seq) model for machine translation, using Attention. However, instead of looking at the Attention plots, we do the following:

1) Explore the data by aggregate uncertainty for analyzing avoidable bias, variance, and sampling for Active Learning
2) Use Uncertainty plots in order to drive analysis and interpretibility of results.

This work was inspired in part by Human-in-the-Loop Machine Learning by Robert Munro © 2020
For this little exercise I've chosen a toy Machine Learning example, which affords some fun and interesting examples of how for a given translation output the model may be trying to say ... something about its own uncertainty.  The choice of MT affords a look at not just on the overall aggregate output sentence uncertainty, but on the constituent tokens which can lend to some interpretibility. 

So as part of
Let's say you want to rank and find the "most uncertain" outputs (in this case, sentences)  for human review and possible (re)training.   Interestingly enough, using a custom softmax, or using a the first or second bar chart instead of the 3rd combination as I do in the notebook, *can* change the overall uncertainty rankings of multiple outputs.    That Munro book I cite at the top of the nb emphasizes that there's nothing probabilistic or magical about softmax for this purpose, but its especially useful for uncertainty when softmax is not originally used as part of the optimization of the final layer.  That all the scores add up to 1 leads some to that "probabilistic" confusion, but it doesn't matter.

The original model's output just selected the maximum raw score (logits) from each timestamp.  Afer that (post-optimization) a softmax normalization is added, so that for a given timestamp, all the scores add up to one.

Here I'll just pause to note that the potentital confusion (pardon the pun) among terms like "uncertainty" and "confidence" and "probability".  Since this is a conditional (distributive) model, the a given uncertainty score let's say 0.6, is *not* an indication that there is a 60% probability that this is wrong.  In fact, the point is to try different metrics in order to gain more insights in our error analysis by uncertainty, as we do in the notebook.  Keep in mind that different metrics can yield different rankings of uncertainty 

(Note: the first third or so of this notebook is mostly setting up the training and model and actually doing the training using an Attention model, adapted and slightly modified from a reference google demo notebook.  Also for reasons having an in-house native speaking spouse, this happens to use Hebrew as the source language, but shouldn't matter since most of the specific examples just compare the English outputs.  Remember, to verify Google Translate is your friend!)

## Some Uncertainty sampling classes
One challenge with this datset is that there is usually exactly one reference translation.  As a crude start then we can simply consider all word-for-word matches with the single target, andt otherwise these are mis=matches.  (There are no False Positives among Matches)  Let's consider positives and negatives in the context of both uncertainty and matching:

1) *True Negatives with respect to mis-matches*:

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Mistranslation1.PNG?raw=true&width=300&height=300)

2) *False Negatives with respect to mis-matches* (good alternate translations with low uncertainty): often acceptable replacement with an synonymous word or words, e.g. "perplexed/confused" (above example), "this/that", "keep" vs. "put" (image above).  These acceptable replacements can be recast as *True Positives* w.r.t. uncertainty scores and added to our reference translations ground truth.

    2a) Partially FN, partially TN hyvrid, where a whole subclause can be correct and then another goes off the rails: 
    
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Long%20sentence%20started%20out%20ok.PNG?raw=true&width="1000"&height="500")

    2b) Mislabelled ground truth!  Usually we can live with these random labelling errors in Deep Learning training with lots of data, unless there is a more systematic error underlying these.  However, this is more important for dev/test sets:

3) *False Negatives w.r.t. uncertainty* can arise, like the "mask" example above, or here (flight/hotel), which offers up a another class of potential errors (or where the model more or less got "lucky" to work on for model refinement/training:

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Flight%20vs.%20Hotel.PNG?raw=true&width="500"height="400")

4) *False Positives w.r.t. uncertainty*: mis-translations with low uncertainty, we do find a few in our exploration of underfitting of the training set and variance of the validation set, again offering up samples we might not have considered otherwise for training or model refinement.  These will be harder to find with this method but if they do arise in a low uncertainty context, this can be prioritized.

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Believable%20versus%20reliable.PNG?raw=true&width="500"height="400")

5) *True Positives w.r.t. uncertainty*: these would be the many examples of matches with low uncertainty in the notebook.




Its interesting to note sometimes which individual words/tokens will have high uncertainty, often indicating at the token level where the translation went awry. This is often indicated by the "runner-up" (2nd highest scoring) translation for that token(s).  This could be of help for humans in the loop correcting these translations using a manual interface, for example.  Thus the outright wrong results are at least somewhat explainable.  Also, can knowing more about the model confusion info itself to try different things, like in this case increase the beam width?

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Runner-up%20was%20correct!.PNG?raw=true)


## Aggregate Results
With some variation in the ratios, the density of raw mis-matches (True or False Negatives) is positively correlated with uncertainty.  In one run 

**32.1%** of the non-matches (potential errors) are found by **10.0%** of the target sentences with the highest uncertainty score.

But that is not usually the case.  
But it usually holds true that both the distribution of mis-matched bad translations is weighted in the high uncertainty.  


But then with a quick tool for exploration, it's easy to examine the presumptive Negatives to see if they are True or False Negatives (bad vs. good translations).  There, we find a marked concentration of True Negatives in high uncertainty.

mismatched but "good" mismatched but "bad" - low confidence, so it makes sense for example in an Active Learning scenario to go after the low confident mismatches first.  Put another way, our True Negatives are overwhelmingly concentrated at the high uncertainty percentiles.

## Conclusions:
This is an illustration (with graphs even!) of using uncertainty in ML, using ML as an example.  This type of exploration can lead to better Error analysis:
1. Uncertainty is positively correlated with True Negatives.
2. Interpretibility is aided to some extent with score graphs and "runner-up" token translations
3. Better exploration and sampling with for underfitting and variance.  Even if a translation output matches a reference translation, uncertainty can be used for analysis and sampling.
3. Part of an early stage of a virtuous cycle of model-based Active Learning, with new fixed reference translations to improve our ground truth for training/dev/test, as well as prioritizing the team's

## References:
1. Human-in-the-Loop Machine Learning by Robert Munro © 2020
2. 



```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mahaley22/Uncertainty-Sampling/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### References

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
