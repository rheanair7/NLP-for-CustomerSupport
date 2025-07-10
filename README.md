# NLP-for-CustomerSupport
> An NLP-powered system to extract actionable insights from telecom customer call transcripts using BERT and BERTopic.

## Overview

**NLP-for-CustomerSupport** is a smart, scalable NLP pipeline built to analyze customer service conversations in the telecom industry. The system leverages transformer-based sentiment analysis and unsupervised topic modeling to uncover what customers are talking about — and how they feel about it.

Built using real-world call transcript data, this project helps organizations move beyond manual review by transforming **unstructured call logs** into **clear, visual insights** that support smarter decisions in product, operations, and support coaching.



## Business Need

Telecom providers receive **thousands of customer service calls daily**, rich with feedback — but most of this data is **unstructured** and **unused**. Manual review is expensive and slow.

This project addresses key business problems:

- What topics are generating the most complaints?
- Which issues are driving customer dissatisfaction?
- How can we coach agents based on actual conversation patterns?

With this project, teams can automatically monitor service quality, detect emerging issues, and identify training needs all from raw text.



## Key Objectives

- Perform sentiment analysis using BERT on customer utterances  
- Identify dominant themes using BERTopic and sentence embeddings  
- Visualize patterns across topics, sentiment, and agent/customer speech  
- Enable data-driven decisions in support operations


## 🛠️ Tech Stack

| Tool/Library            | Purpose                             |
|-------------------------|-------------------------------------|
| Python (Google Colab)   | Scripting and experimentation       |
| Hugging Face Transformers | Sentiment analysis with BERT       |
| BERTopic                | Topic modeling from text data       |
| Sentence Transformers   | Dense text embeddings               |
| Pandas, Matplotlib      | Data handling and visualization     |
| MongoDB (optional)      | Scalable storage for annotated calls|
| AWS S3/EC2 (optional)   | Hosting and storage infrastructure  |



## Dataset

- **Source**: [Talkmap Telecom Conversation Corpus – Hugging Face](https://huggingface.co/datasets/talkmap/telecom-conversation-corpus)
- **Content**: Real-world, anonymized customer service dialogues from the telecom industry  
- **Columns**: `conversation_id`, `speaker`, `date_time`, `text`  
- ~10,000+ utterances



## Pipeline Overview

### 1. Data Preprocessing
- Remove noise and clean text
- Tokenize and prepare for embedding

### 2. Sentiment Analysis with BERT
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Output: `Positive` or `Negative` label per utterance

```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
df['sentiment'] = df['text'].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])
