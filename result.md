# Results for hate-speech-classification

## distilBERT

![Confusion matrix_bert](images/distilbert.png)
![Confusion matrix_bert2](images/distilbert_norm.png)
    
                precision    recall  f1-score   support
        hate       0.60      0.40      0.48       139
    offensive      0.95      0.97      0.96      1895
     neither       0.90      0.93      0.91       445

    accuracy                            0.93      2479
    macro avg       0.81      0.77      0.78      2479
    weighted avg    0.92      0.93      0.92      2479

## LSTM
![Confusion matrix_lstm](images/lstm.png)
![Confusion matrix_lstm2](images/lstm_norm.png)

                 precision    recall  f1-score   support
        hate        0.62      0.12      0.19       139
    offensive       0.94      0.96      0.95      1895
     neither        0.83      0.93      0.88       445

    accuracy                            0.91      2479
    macro avg       0.79      0.67      0.67      2479
    weighted avg    0.90      0.91      0.89      2479
