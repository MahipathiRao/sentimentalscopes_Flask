import re
import string
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import PyPDF2
from flask import Flask, render_template, request, send_file
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing whitespace
    return text


def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def extract_sentences_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split text into sentences
    return sentences


def analyze_pdf_sentiment(file):
    sentences = extract_sentences_from_pdf(file)
    results = []
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for sentence in sentences:
        cleaned_sentence = preprocess_text(sentence)
        sentiment = get_vader_sentiment(cleaned_sentence)
        results.append({'sentence': sentence, 'sentiment': sentiment})
        sentiment_counts[sentiment] += 1

    # Calculate performance metrics
    total = sum(sentiment_counts.values())
    positive_percentage = (sentiment_counts['positive'] / total) * 100 if total > 0 else 0
    negative_percentage = (sentiment_counts['negative'] / total) * 100 if total > 0 else 0
    neutral_percentage = (sentiment_counts['neutral'] / total) * 100 if total > 0 else 0

    # Append performance metrics to results
    results.append({
        'sentence': 'Performance Metrics',
        'sentiment': f"Positive: {positive_percentage:.2f}%, Negative: {negative_percentage:.2f}%, Neutral: {neutral_percentage:.2f}%"
    })

    return pd.DataFrame(results)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'pdf_file' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['pdf_file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file and file.filename.endswith('.pdf'):
            try:
                # Perform sentiment analysis
                sentiment_results = analyze_pdf_sentiment(file)

                # Convert DataFrame to HTML table
                results_html = sentiment_results.to_html(classes='table table-striped table-bordered', index=False)

                # Save results to an in-memory Excel file
                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                    sentiment_results.to_excel(writer, index=False)
                output_buffer.seek(0)

                # Store the buffer in a global variable (or session) for download
                app.config['excel_buffer'] = output_buffer

                return render_template('index.html', results=results_html, success=True)
            except Exception as e:
                return render_template('index.html', error=f"Error processing file: {str(e)}")
        else:
            return render_template('index.html', error="Please upload a valid PDF file")

    return render_template('index.html')


@app.route('/download')
def download_excel():
    # Retrieve the Excel buffer from app config
    if 'excel_buffer' not in app.config:
        return "No results available for download", 400

    buffer = app.config['excel_buffer']
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="sentiment_analysis_results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == '__main__':
    app.run(debug=True)