# Leveraging Model (Un)certainty: can a model "know" when it's predictions are incorrect or not?

This work explores model uncertainty scoring in Neural Net Machine Learning, using Machine Translation (Attention) as a use case.  

<img src="https://github.com/mahaley22/Uncertainty-Sampling/blob/master/Keep%20your%20mask%20on!.PNG" width="200" height="200" />
<img src="https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/A%20slight%20confusion" width="200" height="200" />



Show that not only is uncertainty positively correlated with mismatches from the target translation, but also correlated with mismatches that are actually True Negatives, i.e. not acceptable alternate translations.

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

For this little exercise I've chosen a toy Machine Learning example, which affords some fun and interesting examples of how for a given translation output the model may be trying to say ... something about its own uncertainty.  The choice of MT affords a look at not just on the overall aggregate output sentence uncertainty, but on the constituent tokens which can lend to some interpretibility. 

So as part of
Let's say you want to rank and find the "most uncertain" outputs (in this case, sentences)  for human review and possible (re)training.   Interestingly enough, using a custom softmax, or using a the first or second bar chart instead of the 3rd combination as I do in the notebook, *can* change the overall uncertainty rankings of multiple outputs.    That Munro book I cite at the top of the nb emphasizes that there's nothing probabilistic or magical about softmax for this purpose, but its especially useful for uncertainty when softmax is not originally used as part of the optimization of the final layer .  That all the scores add up to 1 leads some to that "probabilistic" confusion, but it doesn't matter.

The original model's output just selected the maximum raw score (logits) from each timestamp.  To that this notebook adds (post-optimization) a softmax normalization, so that for a given timestamp, all the scores add up to one.
Without making too much fuss about it, I'll just pause to note that the potentital confusion (pardon the pun) among terms like "uncertainty" and "confidence" and "probability".  Since this is a conditional (distributive) model, the a given uncertainty score let's say 0.6, is *not* an indication that there is a 60% probability that this is wrong.  In fact, the point is to try different metrics MORE HERE
I'm not necessarily breaking new ground here for using uncertainty in MT or ML, but I've never yet seen an implementation that "paints a picture" for practitioners in industry, maybe just another widget in their toolkit to bear in mind as we consider more things  than raw accuracy or throughput or computational cost.  

(Note: the first third or so of this notebook is mostly setting up the training and model and actually doing the training using an Attention model, adapted and slightly modified from a reference google demo notebook.  Also for reasons having an in-house native speaking spouse, this happens to use Hebrew as the source language, but shouldn't matter since most of the specific examples just compare the English outputs.  Remember, to verify Google Translate is your friend!)

## Some interesting examples
One challenge with this datset is that there is a lack of complete alternate reference translations.  So when mismatches between the source and target do occur, 
1) acceptable replacement with an acceptable synonym, e.g. "perplexed/confused", "this/that".  These acceptable replacements might have little uncertainty, but plenty do have higher uncertainty

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Flight%20vs.%20Hotel.PNG = 100x100)
<img src="https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Flight%20vs.%20Hotel.PNG" width="500" height="400" />

2) non-acceptable

![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Mistranslation1.PNG?raw=true)

3) whole section of a sentence that are problematic 
<img src="https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Long%20sentence%20started%20out%20ok.PNG" width="1000" height="500" />


Its interesting to note sometimes which individual words/tokens will have high uncertainty, often indicating at the token level where the translation went awry. This is often indicated by the "runner-up" (2nd highest scoring) translation for that token(s).  This could be of help for humans in the loop correcting these translations using a manual interface, for example.  Thus the outright wrong results are at least somewhat explainable.  Also, can knowing more about the model confusion infoitself to try different things, like in this case increase the beam width?
![Image](https://github.com/mahaley22/Uncertainty-Scoring/blob/gh-pages/images/Runner-up%20was%20correct!.PNG)


Put "This vs. that" vs. "confused vs. embarassed" 
Admittedly your plan makes sense

You can use the [editor on GitHub](https://github.com/mahaley22/Uncertainty-Sampling/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

## Overall Results

**32.1%** of the non-matches (potential errors) are found by **10.0%** of the target sentences with the highest uncertainty score.

mismatched but "good" mismatched but "bad" - low confidence, so it makes sense for example in an Active Learning scenario to go after the low confident mismatches first.  Put another way, our True Negatives are overwhelmingly concentrated at the high uncertainty percentiles.

## Conclusions:


**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mahaley22/Uncertainty-Sampling/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### References

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
