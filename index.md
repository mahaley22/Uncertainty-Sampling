 It's just something i tacked on to the outputs ("logits") of the final layer as an implementation detail. The original code just spat out the raw scores, so my applying softmax after optimization does not change the individual tokens ranking. 

However,  let's say you want to rank and find the "most uncertain" outputs (sentences)  for human review.   Interestingly enough, using a custom softmax, or using a the first or second bar chart instead of the 3rd combination as I do in the notebook, *can* change the overall uncertainty rankings of multiple outputs.    That Munro book I cite at the top of the nb emphasizes that there's nothing probabilistic or magical about softmax for this purpose, but its especially useful for uncertainty when softmax is not originally used as part of the optimization of the final layer .  That all the scores add up to 1 leads some to that "probabilistic" confusion, but it doesn't matter.

I'm not necessarily breaking new ground here for using uncertainty in MT or ML, but I've never yet seen an implementation that "paints a picture" for practitioners in industry, maybe just another widget in their toolkit to bear in mind as we consider more things  than raw accuracy or throughput or computational cost.  

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/mahaley22/Uncertainty-Sampling/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mahaley22/Uncertainty-Sampling/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
