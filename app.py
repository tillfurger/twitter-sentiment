from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from io import BytesIO
import base64
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize': (10.5, 5)})

matplotlib.use('Agg')

app = Flask(__name__)

output_dir = "data/"

LABEL_NAMES = ['Negative', 'Neutral', 'Positive']

model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=3, output_attentions=False, output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
model.eval()


def tokenize_text(review_text):
    encoded_review = tokenizer.encode_plus(review_text,
                                           add_special_tokens=True,
                                           max_length=512,
                                           padding='max_length',
                                           return_attention_mask=True,
                                           return_token_type_ids=True,
                                           return_tensors='pt')
    return encoded_review


def get_prediction(encoded_review):
    device = torch.device("cpu")
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    model.to(device)
    output = model(input_ids, attention_mask)
    m = torch.nn.Softmax(dim=1)
    probas = m(output[0])
    _, prediction = torch.max(probas, dim=1)
    return LABEL_NAMES[prediction], probas


def create_figure(model_output):
    outputs_df = pd.DataFrame({
        'class_names': LABEL_NAMES,
        'values': model_output.cpu().detach().numpy()[0]
    })
    img = BytesIO()
    clrs = ['red', 'blue', 'green']
    sns.barplot(x='values', y='class_names', data=outputs_df, orient='h', palette=clrs)
    plt.xlabel('Probability')
    plt.ylabel('Sentiment')
    plt.xlim([0, 1])
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'text' not in request.form:
            return 'No Text entered'
        text = request.form['text']
        tok_text = tokenize_text(text)
        prediction, probas = get_prediction(encoded_review=tok_text)
        plot = create_figure(model_output=probas)
        return render_template('result.html', raw_text=text, prediction=prediction.upper(), plot=plot)


if __name__ == '__main__':
    app.run(debug=True)
