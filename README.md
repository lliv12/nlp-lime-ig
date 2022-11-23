# nlp-lime-ig
Project on using model explanation techniques such as LIME and Integrated gradients on long sequences of text.

### Resources:

- Integrated Gradients Model:  https://captum.ai/
- Tokenization (Hugging Face):  https://github.com/huggingface/tokenizers/
- Pretrained Models (Hugging Face):  https://huggingface.co/docs/transformers/index
  * Probably would want to change the head for continuous/binary scoring, fine-tune on our datasets.

### Getting started
```
<create conda env>
conda install -c pytorch pandas numpy matplotlib pytorch captum
pip install transformers
```

### Training (Examples)
```
# Train binary DAN classifier on the reviews dataset
python train.py -d reviews -t dan -n reviews_classifier -s binary -e 10 -b 32

# Train categorical transformer classifier on the essays dataset
python train.py -d essays -t transformer -n essays_classifier -s categorical -e 5 -b 32 -sq 500
```
