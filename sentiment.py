from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import os
# from google.cloud import storage
from google.cloud.language_v1.proto import language_service_pb2
path = os.getcwd()
config_path = path + '/sentiment-configuration.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config_path

def analyzeText(text):
    print "Performing sentiment analysis."
    API_SIZE_LIMIT = 1000000
    text = text[:API_SIZE_LIMIT]
    # language_client = language.Client()
    # document = language_client.document_from_text(text)
    # sentiment = document.analyze_sentiment()

    # client = language_v1.LanguageServiceClient()
    # type_ = enums.Document.Type.PLAIN_TEXT
    # document = {'content': text, 'type': type_}
    # sentiment = client.analyze_sentiment(document)

    # Instantiates a client
    client = language.LanguageServiceClient()

    # The text to analyze
    # text = u'Hello, world!'
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment

    return sentiment
