# Leveraging Model (Un)certainty: can a model "know" when it's"bad" or "good"?
(warning: I may over-use "scare quotes" in this piece)

This work explores model uncertainty scoring in Machine Learning, using Machine Translation using Neural Nets (Attention) as a toy example.  

![Image](https://github.com/mahaley22/Uncertainty-Sampling/blob/master/Keep%20your%20mask%20on!.PNG?raw=true)

- Here is an example of how one might try it with common tools.

    - High level description of non-uncertainty preserving system with an example
    - High level description of your uncertainty preserving system with an example
It looks like the first half of the note book sets up a standard system. Boy what a train set--I thought these packages were easier to setup.
On your work, if you are using softmax() like I think you are then you are correct to have scare quotes around "probabilistic". The issue is calibration of estimates, a .6 probability means that in the wild there will be .6 of the cases will apply. I think a more accurate way of describing things is that your scores impose a ranking but you don't need softmax for that--the original scores will do.
I think what would help is a high level description requested above will help me a ton but maybe that is because I am the wrong reader. Is this for fun? For getting a job?

# Intro/Summary

    1) "Where": more automatically detecting errors in unlabelled sets and production

    2) "What": targetted sampling for enhancing training or dev/test sets

    3) "Why": "Why did the model get this wrong?"  (everybody's favorite question in ML)

    4) "When": detecting model drift by measuring aggregate uncertainty

    5) "How":  to improve the model as quickly and cost-effectively as possible?

As a practitioner of Applied ML for a number of years now, I'm not alone in having these questions posed to me, by myself, or worse at times, by others like internal stakeholders or customers.  So I wanted to try NN model uncertainty to see if it can be useful, even if the model itself is weaker than we would like.  In fact, that's the whole point: we want to improve the model using all the means we have at our disposal: hyperparameter tuning, training, etc. as part of the Active Learning iterative process.  Bear in mind that information from inside the model is certainly not the only tool to leverage for things like Active Learning.

So we all know the saying: all models are wrong, but some are useful.  
These days, we hear a lot about things that may or may not be useful about models other than their raw accuracy.  No matter how accurate or *good* one's model is, not only will there will always be things like data drift, concept drift, or simply generalization issues on things the model hasn't seen or tested for before (see checklist paper). So that's where Active Learning comes in to answer some of these questions: And, where do you have humans in the loop?  How much hand-labeled training do you need up front and on an ongoing basis?  When should a human cgecj the output that needs it for possible correction and training?  How do we know what the model(s) know they know, know what they don't know, and don't know either? 

And yet, by definition the goal of optimizing a model is not primarily (at least when we're talking about conditional modeling, like in Machine Translation) in the business of generating accurate "probabilities" or confidences for those predictions.  And how explainable are the results?

And worse yet, how can one even tease out such information  of a deep learning algorithm, which by its nature is a nested non-linear structure? 
 the process I hope to make the case that yes, the model "knows" what it does and doesn't know, and that this follows a pattern that is helpful for analysis, in 

# Methodology
Note: the first third or so of this notebook is mostly setting up the training and model and actually doing the training using an Attention model, adapted and slightly modified from a reference google demo notebook.
For this little exercise I've chosen a toy Machine Learning example, which affords some fun and interesting examples of how for a given translation output the system can be trying to say " something about its own uncertainty - not just on the overall output sentence let's say, but on the constituent sub-tokens.  

Let's say you want to rank and find the "most uncertain" outputs (in this case, sentences)  for human review and possible (re)training.   Interestingly enough, using a custom softmax, or using a the first or second bar chart instead of the 3rd combination as I do in the notebook, *can* change the overall uncertainty rankings of multiple outputs.    That Munro book I cite at the top of the nb emphasizes that there's nothing probabilistic or magical about softmax for this purpose, but its especially useful for uncertainty when softmax is not originally used as part of the optimization of the final layer .  That all the scores add up to 1 leads some to that "probabilistic" confusion, but it doesn't matter.

By the way there can be lot of potential confusion, pardon the pun among uncertainty and confidence and probability
I'm not necessarily breaking new ground here for using uncertainty in MT or ML, but I've never yet seen an implementation that "paints a picture" for practitioners in industry, maybe just another widget in their toolkit to bear in mind as we consider more things  than raw accuracy or throughput or computational cost.  

## Some interesting examples

Its interesting to note sometimes which individual words/tokens will have high uncertainty, often indicating at the token level where the translation went awry. This is often indicated by the "runner-up" (2nd highest scoring) translation for that token(s).  This could be of help for humans in the loop correcting these translations using a manual interface, for example.  Thus the "wrong" results are at least somewhat explainable.  Also, can knowing more about the model confusion infoitself to try different things, like in this case increase the beam width?

Put "This vs. that" vs. "confused vs. embarassed" 
Admittedly your plan makes sense

You can use the [editor on GitHub](https://github.com/mahaley22/Uncertainty-Sampling/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Overall Results

**32.1%** of the non-matches (potential errors) are found by **10.0%** of the target sentences with the highest uncertainty score.

mismatched but "good" mismatched but "bad" - low confidence, so it makes sense for example in an Active Learning scenario to go after the low confident mismatches first.  Put another way, our True Negatives are overwhelmingly concentrated at the high uncertainty percentiles.

### Conclusions:


**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mahaley22/Uncertainty-Sampling/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### References

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
