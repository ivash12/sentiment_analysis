from predict import predict_sentiment_en
message = str(input("Enter your message here to get its grade: "))
print(predict_sentiment_en(message))

