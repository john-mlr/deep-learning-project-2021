# deep-learning-project-2021

#### Background:
Our deep learning project spawned from previous work Gabe and I have performed in an attempt to leverage modern techniques in deep learning (such as self-supervision and attention) into a cohesive model which can accurately predict Braak score.

Braak is a measure commonly used in the neurodegeneration field to grade how much tau has accumulated in the brain. These Braak assessments are commonly made on wholse-slide-images (WSIs) at a low magnification and can often be unreliable and subjective between observers. Additionally, the sheer size of WSIs (often measuring over 40,000 by 40,000 pixels) makes computational modelling difficult. 

Given these issues with a computational model of Braak stage, we sought out to construct a model which could learn the features of a hippocampal WSI and predict Braak score. To do so, we generated 256x256 pixel patches drawn from the WSIs, and trained the unsupervised model Bootstrap Your Own Latent (BYOL) on them. Using an unsupervised method for feature learning has many distinct advantages, primary of which is lack of need for patch level labels. This allowed us to train a Resnet-50 backbone to recognize features using these patches, greatly expanding the specificity of our model.

#### This Project:
As a part of this larger goal, our focus has been two pronged:

1. Train an attention based model to evaluate our embeddings.
2. Use KMeans clustering to gain a deeper understanding of the information encoded by BYOL.

These aims are descirbed further, along with their results, in 2 notebooks:

1. attention_eval.ipynb
