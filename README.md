# nlp-lime-ig
Project on using model explanation techniques such as LIME and Integrated gradients on long sequences of text.

Resources:

- (Integrated Gradients Model):  https://captum.ai/
```
conda install captum -c pytorch
```
- (Tokenization (Hugging Face)):  https://github.com/huggingface/tokenizers/
```
pip install transformers
```
- (Pretrained Models (Hugging Face)):  https://huggingface.co/docs/transformers/index
  * Probably would want to change the head for continuous/binary scoring, fine-tune on our datasets.
