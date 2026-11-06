Thoughts and notes on the topic, as well as our specific experiment.

# Notes on feasibility


# Feasibility of Qwen3-4B as a teacher
A valid concern was raised: can our Qwen3-0.6B student actually learn anything from the teacher? The Qwen3-4B teacher is a general model not an expert in a particular topic, so does it actually have knowledge that will be helpful to the student?
If this is not the case we would have to find another teacher to use or look at more difficult datasets that reveal the gap between the models.

In order to determine this I ran a benchmark test, evaluating both the teacher and student zero-shot on the training data (GSM8K). The related notebook can be accessed on Kaggle [here](https://www.kaggle.com/code/redkitex/headroom-test) or in the Github under `code/headroom-test.ipynb`. 
This test determined that there is sufficient headroom between the teacher and the student for us to proceed with using Qwen3-4B as a teacher.
