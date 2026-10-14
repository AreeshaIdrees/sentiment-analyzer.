print("Areesha Idrees B04-0923-0058")
from transformers import pipeline
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(clf("I love this course"))
print("Model:", clf.model.name_or_path)
