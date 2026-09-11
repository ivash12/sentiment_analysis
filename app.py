from predict import predict_sentiment_en, predict_sentiment_nl

lan = input("choose your language. type 'en' for english and 'nl' for dutch: ")
while True:
    message = input("if you want to exit type: 'exit' \nenter your text: ")
    if message == "exit":
        print("thank you for using the model")
        break
    if lan == "en":
        print(predict_sentiment_en(message))
    elif lan == "nl":
        print(predict_sentiment_nl(message))



