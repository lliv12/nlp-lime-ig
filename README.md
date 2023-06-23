# nlp-lime-ig
In this project, we attempt to use a method called Integrated Gradients in order to extract explanations from a language model by highlighting pieces of text that contributed positively or negatively to a prediction. We test this for language models trained on two different datasets: Hewlett Foundation essays and Amazon reviews.

## Details
**Integrated Gradients** (https://arxiv.org/pdf/1703.01365.pdf) is a technique for extacting the relative importance of parts of the input to a model for a given prediction. It works by interpolating between a baseline input (Ex: *all padding tokens* in the case of NLP), and a given input example; and computing the gradient of the model output with respect to the input for each interpolation, and summing over those (*integrating*) to get a score for each input component. Integrated gradients is model-agnostic, and also works in other modalities such as image data. We utilize the Captum library (https://captum.ai/docs/extension/integrated_gradients) to make inference using integrated gradients.

We provide some **pre-trained models** that can be downloaded as described below. These are transformer models, and each has the following architecture:  *128 embedding dimension*, *512 feed-forward dimension*, *2 attention heads*, *2 encoder layers*. This architecture seemed to perform the best in our experiments. Variations of each model are largely along the embedding size. We also tried using a special loss function that better takes into account ordinal classes (https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99), and it worked better for models trained on essays.

The original datasets were heavily pre-processed in order to make training feasible. Most pressing was the issue of class imbalance in both datasets. This was addressed by *undersampling* majority classes during training. The essays dataset was quite particular in that there were different sets with their own ranges of scores. We distilled these into four (*equally sized*) bins so that we could use a dense prediction head to make inference. You can refer to this notebook file: **datasets_overview.ipynb** for more details and analysis about the dataset and some decisions we made for pre-processing.

## Getting started
### Create environment (Anaconda)
```
conda env create --file environment.yml
conda activate nlp-lime-ig
```
### Download Amazon Reviews
Reference:  https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Use the module **data/download_dataset.py** to download the Amazon reviews. The dataset itself is huge, so it is suggested that you download only a subset of the review categories, or a subset of the review files themselves. Use **--cat** to specify which review categories you'd like to download (*refer to AMAZON_DICT in data/download_dataset.py for a full list of categories*). Use **--limit** to limit how many examples you save from each review file. Resulting files will be saved as jsons in *dataset/amazon/*.

**Example:  (download all reviews; up to 5,000 examples from each)**
```
python -m data.download_dataset --catspec all --limit 5000
```

### Download Essays
Reference:  https://www.kaggle.com/competitions/asap-aes

The Hewlett Foundation Essays dataset is available on Kaggle. You must create a Kaggle account and manually download it yourself. Here are the steps:
1) Create a Kaggle account
2) Navigate to the page for *The Hewlett Foundation: Automated Essay Scoring* challenge (*linked in the reference above*)
3) Download the associated zip file (https://www.kaggle.com/competitions/asap-aes/data); extract into **dataset/kaggle**

### Download Pre-trained Models
Use the module **download_models.py** to download pre-trained models for inference. Available models are the following:
* **basic_transformer_reviews**:   (*40.2*% accuracy, *0.04* average f1 trained on all reviews limited to 5,000 examples)
* **transformer_reviews_5000**:   (*30.4*% accuracy, *0.04* average f1 trained on all reviews limited to 5,000 examples; 5,000-token)
* **basic_transformer_essays**:   (*44.0*% accuracy, *0.15* average f1 on essays; ordinal loss)
* **transformer_essays_5000**:   (*40.6*% accuracy, *0.20* average f1 on essays; ordinal loss 5,000-token)

**Example:  (downloads all pre-trained models to *saved_models/reviews* and *saved_models/essays*)**
```
python -m download_models.py
```

## Launching the App
An application has been included which will allow you to preview an example and see the model's prediction, along with a visual interface depicting the relative importance of each token with respect to the prediction. Highlighted in green are tokens that contribute positively to the model's prediction, and in red are tokens that contribute negatively towards this prediction. (*Ex*: the word '*terrible*' might be highlighted in red more often for reviews with a rating of *5*, while '*great*' might be highlighted in green more often). The different options in the application allow you to choose a model, dataset, and example based on several parameters. Click on the "**GO**" button to begin inferencing.

**Run the following to launch the app:**
```
python app.py
```
<p align="center">
  <img src="assets/app_screenshot.png?raw=true" alt="Image" width="540" height="575"/>
</p>

## Training
To train your own model, refer to **train.py** (*detailed instructions are included at the top of the file*). Default model definitions are included in **models.py**. . The basic usage for **train.py** is like this:  (*Refer to **train.py** for all of the options*)
```
python -m train --dataset <reviews/essays> --metric f1 --tb_vis_interval 150 ... <options>
```
The above command will train a fresh transformer model for 10 epochs and validate using the average reciprocal of the f1 score (*more useful to take into account class imbalance*). It will also save both a bar graph and confusion matrix inside of *logs/<reviws/essays>/<model_name>/<train/val>/*, showing the model's performance on each class. This will also be viewable in Tensorboard. Model checkpoints will be saved in *saved_models/<reviws/essays>/*

### Tensorboard:
In a separate terminal run:
```
tensorboard --logdir logs
```
You will be able to see the model's performance, including plots for it's per-class performance (*if you specify --tb_vis_interval for training*)

### Example Plots:
Specify **--tb_vis_interval <global_step>** to see these in the logs folder during training, or on Tensorboard.
<p align="center">
  <img src="assets/viz_metrics.png?raw=true" alt="Image" width="658" height="400"/>
</p>

## Evaluation
Refer to **evaluate.py** for a complete overview of the module and available options. You can evaluate a given model (*must be present in saved_models/*) on the entire dataset it was trained on (train + val). You will need to specify the **model_name**, **dataset**, and the **loss_func** that was used to train it. For the pre-trained reviews models the command would look like this:
```
python -m evaluate basic_transformer_reviews reviews --loss_func cross_entropy ... <options>
```
For the essays models it would look like this: (*since these were trained using an ordinal loss*)
```
python -m evaluate basic_transformer_essays essays --loss_func ordinal ... <options>
```
The results will be saved in *logs/<reviws/essays>/<model_name>/<eval>/* by default.
