import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload your exported chat (.txt)", type=["txt"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Analyze messages from:", user_list)

    if st.sidebar.button("Show Analysis"):
        num_messages, words, media, links = helper.fetch_stats(selected_user, df)

        st.title("Key Chat Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Messages", num_messages)
        with c2:
            st.metric("Total Words", words)
        with c3:
            st.metric("Media Shared", media)
        with c4:
            st.metric("Links Shared", links)


        st.header("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        st.header("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        st.header("Activity Maps")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Most Active Day")
            day_map = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(day_map.index, day_map.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with c2:
            st.subheader("Most Active Month")
            month_map = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(month_map.index, month_map.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        st.subheader("Weekly Heatmap")
        heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap, ax=ax)
        st.pyplot(fig)


        if selected_user == "Overall":
            st.header("👥 Most Active Users")
            x, new_df = helper.most_busy_users(df)
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with c2:
                st.dataframe(new_df)


        st.header("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)


        st.header("Most Common Words")
        mc = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(mc[0], mc[1], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        st.header("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(emoji_df)
        with c2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%.2f")
            st.pyplot(fig)


        st.header("Sentiment Analysis")
        sentiment_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%.1f%%",
            colors=['lightgreen', 'lightcoral', 'lightgray']
        )
        st.pyplot(fig)


        st.header("Messages by Part of Day")
        part_counts = df['part_of_day'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(part_counts.index, part_counts.values, color='skyblue')
        st.pyplot(fig)