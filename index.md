#What does a model know about its own confidence in its predictions?
For me I could use an overview of what you are up to. I am confused between the implementation details of using the package and your goal of bringing in uncertainty.
    I know nothing about ML so I may well be the wrong demographic, but for me I'd could use a simple summary of what problem you are trying to solve. Is it something like:

- Typical ML does not attempt to handle uncertainty in any sort of well defined way.

- Here is an example of how one might try it with common tools.

    - High level description of non-uncertainty preserving system with an example
    - High level description of your uncertainty preserving system with an example
It looks like the first half of the note book sets up a standard system. Boy what a train set--I thought these packages were easier to setup.
On your work, if you are using softmax() like I think you are then you are correct to have scare quotes around "probabilistic". The issue is calibration of estimates, a .6 probability means that in the wild there will be .6 of the cases will apply. I think a more accurate way of describing things is that your scores impose a ranking but you don't need softmax for that--the original scores will do.
I think what would help is a high level description requested above will help me a ton but maybe that is because I am the wrong reader. Is this for fun? For getting a job?

# Intro/Summary
It seems like these days we hear a lot about things other than accuracy.  Did an "accurate" model fail miserably on a "checklist" test?  No matter how accurate or *good* one's model is, not only will there will always be things like data drift, concept drift, or simply generalization issues on things the model hasn't seen before.  And, where do you have humans in the loop?  How much hand-labeled training do you need up front and on an ongoing basis?  When should a human cgecj the output that needs it for possible correction and training?  How do we know what the model(s) know they know, know what they don't know, and don't know either? 

And yet, by definition the goal of optimizing a model is not primarily (at least when we're talking about conditional modeling, like in Machine Translation) in the business of generating accurate "probabilities" or confidences for those predictions.

And worse yet, how can one even tease such information out of a Neural Net, which by its nature is a giant non-linear function? 

For this little exercise I've chosen a toy Machine Learning example, which affords some fun and interesting examples of how for a given translation output the system can seem " something about its own uncertainty - not just on the overall output sentence let's say, but on the constituent sub-tokens.  In the process I hope to make the case that yes, the model "knows" what it does and doesn't know, and that this follows a pattern that is helpful for analysis, in 

Let's say you want to rank and find the "most uncertain" outputs (in this case, sentences)  for human review and possible (re)training.   Interestingly enough, using a custom softmax, or using a the first or second bar chart instead of the 3rd combination as I do in the notebook, *can* change the overall uncertainty rankings of multiple outputs.    That Munro book I cite at the top of the nb emphasizes that there's nothing probabilistic or magical about softmax for this purpose, but its especially useful for uncertainty when softmax is not originally used as part of the optimization of the final layer .  That all the scores add up to 1 leads some to that "probabilistic" confusion, but it doesn't matter.

I'm not necessarily breaking new ground here for using uncertainty in MT or ML, but I've never yet seen an implementation that "paints a picture" for practitioners in industry, maybe just another widget in their toolkit to bear in mind as we consider more things  than raw accuracy or throughput or computational cost.  

## Some interesting examples

You can use the [editor on GitHub](https://github.com/mahaley22/Uncertainty-Sampling/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Overall Results

**32.1%** of the non-matches (potential errors) are found by **10.0%** of the target sentences with the highest uncertainty score.

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mahaley22/Uncertainty-Sampling/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### References

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
