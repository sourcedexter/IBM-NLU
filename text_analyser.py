"""
Script to analyse text and gain insights into unstructured data such as Sentiment and Emotion.
The complete tutorial can be found at: https://sourcedexter.com/product-review-sentiment-analysis-with-ibm-nlu

Author: Akshay Pai
Twitter: @sourcedexter
Website: https://sourcedexter.com
Email: akshay@sourcedexter.com
"""
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import operator


def read_csv_file(file_path):
    """
    method to read a csv file and return an iterable object
    :param file_path: path to the dataset file
    :return: iterable object
    """
    # read the file and store it as a dataframe
    csv_iterator = pd.read_csv(file_path)
    # print the number of rows found in the file:
    num_rows, num_cols = csv_iterator.shape
    print(f"the number of rows found in file: {num_rows}")
    # print all the column headings
    print("column headings from raw dataset: ", list(csv_iterator.columns.values))

    return csv_iterator


def get_distribution(dataframe, target_column):
    """
    method to find the distribution of a certain column in a given dataframe.
    Shows the generated visualization to the user.
    :param dataframe:
    :param target_column: column upon which the distribution needs to be applied
    :return: dictionary of unique values from target column and its count in the dataset.
    """
    # get the count of unique products in the dataset
    df_clean = dataframe[target_column].value_counts()
    print("number of unique products found: ", len(df_clean.values))

    # building a scatter plot to show the distribution of products
    x = df_clean.values  # the x axis shows the count of reviews per product
    y = np.random.rand(len(df_clean.values))  # y axis does not have any significance here. so setting random values
    z = df_clean.values  # the size of each bubble in the scatter plot corresponds to the count of reviews.

    # use the scatter function to create a plot and show it.
    plt.scatter(x, y, s=z * 5, alpha=0.5)
    plt.show()

    # return the aggregation as a dictionary
    return df_clean.to_dict()


def preprocess_data(dataset_file_path, features_included):
    """
    :param dataset_file_path: path to the dataset
    :param features_included: list of column names to keep. For example : ["name", "review.txt", "date"]
    :return: python dict with product name as key and dataframe with reviews in date sorted order.
    """
    # read the dataset file
    csv_dataframe = read_csv_file(dataset_file_path)
    # keep only those columns which we need
    cleaned_frame = csv_dataframe[features_included]
    # check to see if the column names are what we wanted
    print("column headings from cleaned frame: ", list(cleaned_frame.columns.values))

    # get the count of reviews for each product
    distribution_result = get_distribution(cleaned_frame, "name")

    # get the names of products who have more than 300 reviews
    products_to_use = []
    for name, count in distribution_result.items():
        if count < 300:
            products_to_use.append(name)

    # get only those rows which have the products that we want to use for our analysis
    cleaned_frame = cleaned_frame.loc[cleaned_frame['name'].isin(products_to_use)]

    # data structure to store the individual product details dataframe
    product_data_store = {}
    for product in products_to_use:
        # get all rows for the product
        temp_df = cleaned_frame.loc[cleaned_frame["name"] == product]
        # the date column is in string format, convert it to datetime
        temp_df["date"] = pd.to_datetime(temp_df["reviews.date"])
        # sort the reviews in reverse chronological order
        temp_df.sort_values(by='date')
        # store the dataframe to the product store
        product_data_store[product] = temp_df.copy()

    return product_data_store


def perform_text_analysis(text):
    """
    method that accepts a piece of text and returns the results for sentiment analysis and emotion recognition.
    :param text: string that needs to be analyzed
    :return: dictionary with sentiment analysis result and emotion recognition result
    """
    # initialize IBM NLU client
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2018-11-16',
        iam_apikey='your_api_key_here',
        url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api'
    )

    # send text to IBM Cloud to fetch analysis result
    response = natural_language_understanding.analyze(text=text, features=Features(
        emotion=EmotionOptions(), sentiment=SentimentOptions())).get_result()

    return response


def aggregate_analysis_result(product_dataframe):
    """
    method to analyse and aggregate analysis results for a given product.

    :param product_dataframe: preprocessed dataframe for one product
    :return:
    """

    # data structure to aggregated result
    product_analysis_data = {}

    count = 0
    print("shape of dataframe", product_dataframe.shape)
    # iterate through the reviews in the dataframe row-wise
    for row_index, row in product_dataframe.iterrows():
        print(count + 1)
        count += 1
        review_text = row["reviews.text"]
        date = row["reviews.date"]

        # get the sentiment result.
        analysis = perform_text_analysis(review_text)

        sentiment_value = analysis["sentiment"]["document"]["score"]

        # emotion of the text is the emotion that has the maximum value in the response.
        # Example dict: {"joy":0.567, "anger":0.34, "sadness":0.8,"disgust":0.4}.
        # in the dict above, the emotion is "Sadness" because it has the max value of 0.8
        emotion_dict = analysis["emotion"]["document"]["emotion"]

        # get emotion which has max value within the dict
        emotion = max(emotion_dict.items(), key=operator.itemgetter(1))[0]

        # check if review on date exists. if yes: update values, if no: create new entry in dict
        if date in product_analysis_data:
            product_analysis_data[date]["sentiment"].append(sentiment_value)
            product_analysis_data[date]["emotion"].append(emotion)
        else:
            product_analysis_data[date] = {}
            product_analysis_data[date]["sentiment"] = [sentiment_value]
            product_analysis_data[date]["emotion"] = [emotion]

    # find the average sentiment for each date and update the dict.
    for date in product_analysis_data.keys():
        sentiment_avg = sum(product_analysis_data[date]["sentiment"]) / len(
            product_analysis_data[date]["sentiment"])

        product_analysis_data[date]["sentiment"] = sentiment_avg

    return product_analysis_data


def visualize_sentiment_data(prod_sentiment_data):
    """
    takes in the sentiment data and produces a time series visualization.
    :param prod_sentiment_data:
    :return: None. visualization is showed
    """
    # to visualize, we will build a data frame and then plot the data.

    # initialize empty dataframe with columns needed
    df = pd.DataFrame(columns=["date", "value"])

    # add data to the data frame
    dates_present = prod_sentiment_data.keys()
    for count, date in enumerate(dates_present):
        df.loc[count] = [date, prod_sentiment_data[date]["sentiment"]]

    # set the date column as a datetime field
    df["date"] = pd.to_datetime(df['date'])
    # convert dataframe to time series by setting datetime field as index
    df.set_index("date", inplace=True)
    # convert dataframe to series and plat it.
    df_series = pd.Series(df["value"], index=df.index)
    df_series.plot()
    plt.show()


def visualize_emotion_data(prod_emotion_data):
    """
    method that takes in emotion data and generates a pei chart that represnts the count of each emotion.
    IBM provides data for 5 types of emotions: Joy, Anger, Disgust, Sadness, and fear
    :param prod_emotion_data:
    :return:
    """
    # data structure to hold emotions data
    prod_emotions = {}

    for key in prod_emotion_data.keys():
        emotions = prod_emotion_data[key]["emotion"]

        # update emotion count in the emotions data store
        for each_emotion in emotions:
            if each_emotion in prod_emotions:
                prod_emotions[each_emotion] += 1
            else:
                prod_emotions[each_emotion] = 1

    # define chart properties
    labels = tuple(prod_emotions.keys())
    sizes = list(prod_emotions.values())

    # initialize the chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# starting point of script execution
if "__main__" == __name__:
    # pass the location of the dataset file and the list of columns to keep to the pre-processing method
    dataframe_clean = preprocess_data("./Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
                                      ["id", "name", "reviews.date", "reviews.text"])

    # get the list of product names
    prod_names = list(dataframe_clean.keys())
    for each_prod in prod_names:
        # get the dataframe
        prod_dataframe = dataframe_clean[each_prod]
        # start analysis of all the reviews for the product
        result_analysis = aggregate_analysis_result(each_prod)
        print("product analysis complete")
        # visualize both sentiment and emotion results
        visualize_sentiment_data(result_analysis)
        visualize_emotion_data(result_analysis)
