
# Sentiment Analysis with SpaCy

This project demonstrates a sentiment analysis application built using **Streamlit**, **spaCy**, and a pre-trained machine learning model. The application allows users to input movie or product reviews and predicts whether the sentiment is positive or negative.

---

## Features

- **Interactive UI**: Built using Streamlit for a user-friendly interface.
- **Data Cleaning**: Uses spaCy for text processing, including lemmatization, stopword removal, and punctuation handling.
- **Machine Learning**: Leverages a pre-trained model saved as `sentiment_model.sav` to classify sentiments.
- **Real-time Prediction**: Provides instant feedback on the sentiment of the entered review.

---

## How It Works

1. The user enters a review text in the provided input box.
2. The text is cleaned using spaCy by removing stopwords, punctuation, and applying lemmatization.
3. The cleaned text is passed to the pre-trained machine learning model for prediction.
4. The app displays the result as either **Positive Sentiment** or **Negative Sentiment**.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-spacy.git
   cd sentiment-analysis-spacy
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download SpaCy's small English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## File Structure

- **`app.py`**: Main script containing the Streamlit application code.
- **`sentiment_model.sav`**: Pre-trained machine learning model for sentiment classification.
- **`requirements.txt`**: List of dependencies required to run the project.

---

## Dependencies

- **Streamlit**: For building the web-based user interface.
- **spaCy**: For natural language processing and text cleaning.
- **Joblib**: For loading the pre-trained sentiment analysis model.

---

## Usage Example

1. Launch the app in your local browser by running the Streamlit command.
2. Enter a review text in the input box.
3. Click on the **Predict** button.
4. View the prediction result (Positive or Negative sentiment).

---

## Screenshots

### Application Interface
![App Interface](screenshot.png)

---

## Future Enhancements

- Adding support for more nuanced sentiment categories (e.g., neutral).
- Enhancing the model with additional training data for better accuracy.
- Deploying the app to a cloud platform for wider accessibility.

---



## Contributing

Contributions are welcome! Feel free to submit a pull request or report issues in the repository.

---

## Acknowledgments

- **spaCy**: For providing robust NLP capabilities.
- **Streamlit**: For simplifying the creation of interactive web applications.
- **Scikit-learn/Joblib**: For enabling easy model saving and loading.

---

## Contact

For any questions or feedback, feel free to reach out:

- **GitHub**: [yourusername](https://github.com/pranit144)
- **Email**: pranitchilbule1@gmail.com
