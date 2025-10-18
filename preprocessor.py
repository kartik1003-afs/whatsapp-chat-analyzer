import re
import pandas as pd
import emoji
from textblob import TextBlob

def preprocess(data):

    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*(?:[APMapm]{2})?\s-\s'


    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)


    df = pd.DataFrame({'message_date': dates, 'user_message': messages})


    df['message_date'] = df['message_date'].str.replace('\u202f', '', regex=False)
    df['date'] = pd.to_datetime(df['message_date'].str.strip(' -'), errors='coerce')


    users = []
    messages_clean = []

    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1])
            messages_clean.append(entry[2])
        else:
            users.append('group_notification')
            messages_clean.append(entry[0])

    df['user'] = users
    df['message'] = messages_clean


    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['day_name'] = df['date'].dt.day_name()
    df['only_date'] = df['date'].dt.date


    df['char_count'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['emoji_count'] = df['message'].apply(lambda x: sum(1 for c in x if emoji.is_emoji(c)))


    def get_part_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    df['part_of_day'] = df['hour'].apply(get_part_of_day)

    # ✅ Sentiment
    def get_sentiment(text):
        if '<Media omitted>' in text or 'http' in text:
            return 'Neutral'
        score = TextBlob(text).sentiment.polarity
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    df['sentiment'] = df['message'].apply(get_sentiment)


    df.drop(columns=['user_message', 'message_date'], inplace=True)


    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append(f"00-{hour+1}")
        else:
            period.append(f"{hour}-{hour+1}")
    df['period'] = period

    return df
