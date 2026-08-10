import pandas as pd
from preprocess import preprocess_text, print_data_info, print1, read_dataset
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


data_path = "D://download//training_AI//omar5//dataset//Dataset---Hate-Speech-Detection-using-Deep-Learning.csv"
data_scource = read_dataset(data_path)
preprocessed_text = data_scource['tweet'].apply(preprocess_text)
print(preprocessed_text.head(1))
print1()
