# nlp-lime-ig
Project on using model explanation techniques such as LIME and Integrated gradients on long sequences of text.

Resources:

- (Integrated Gradients Model):  https://captum.ai/
- (Tokenization; Sentence Piece):  https://github.com/google/sentencepiece

```
conda install -c conda-forge sentencepiece
```

- (Pretrained Models (Hugging Face)):  https://huggingface.co/docs/transformers/index
  * Probably want to change the head for continuous/binary scoring, fine-tune on our datasets.
