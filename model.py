# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

# Importing Datasets
Trump_Responses = pd.read_csv('Trump_dataset.csv', encoding = 'utf-8')
Biden_Responses = pd.read_csv('Biden_dataset.csv', encoding = 'utf-8')

# Getting Sentiment Poalrity for datasets
def find_pol(responses):
    return TextBlob(responses).sentiment.polarity

Trump_Responses['Sentiment_Polarity'] = Trump_Responses['text'].apply(find_pol)
Biden_Responses['Sentiment_Polarity'] = Biden_Responses['text'].apply(find_pol)

# Expression Label Attribute
Trump_Responses['Expression Label'] = np.where(Trump_Responses['Sentiment_Polarity']>0,'positive', 'negative')
Trump_Responses['Expression Label'][Trump_Responses.Sentiment_Polarity ==0] = "Neutral"
Trump_Responses.tail()

Biden_Responses['Expression Label'] = np.where(Biden_Responses['Sentiment_Polarity']>0,'positive', 'negative')
Biden_Responses['Expression Label'][Biden_Responses.Sentiment_Polarity ==0] = "Neutral"
Biden_Responses.tail()

# Trump Tweet Analysis
exp_label_trump = Trump_Responses.groupby('Expression Label').count()
x = list(exp_label_trump['Sentiment_Polarity'])
y = list(exp_label_trump.index)
tuple_list = list(zip(x,y))
df = pd.DataFrame(tuple_list, columns=['x','y'])
df['color'] = 'blue'
df['color'][1] = 'red'
df['color'][2] = 'green'
trump_fig = go.Figure(go.Bar(x=df['x'],
                y=df['y'],
                orientation ='h',
                marker={'color': df['color']}))
trump_fig.update_layout(title_text='Trump\'s Tweet Responses Analysis' )
trump_fig.show()



# Biden Analysis
exp_label_biden = Biden_Responses.groupby('Expression Label').count()
x = list(exp_label_biden['Sentiment_Polarity'])
y = list(exp_label_biden.index)
tuple_list = list(zip(x,y))
df = pd.DataFrame(tuple_list, columns=['x','y'])
df['color'] = 'blue'
df['color'][1] = 'red'
df['color'][2] = 'green'
biden_fig = go.Figure(go.Bar(x=df['x'],
                y=df['y'],
                orientation ='h',
                marker={'color': df['color']}))
biden_fig.update_layout(title_text='Biden\'s Tweet Responses Analysis' )
biden_fig.show()
      


# Cleanp of Zero Polarity
CleanUpResponses_Trump = Trump_Responses[Trump_Responses['Sentiment_Polarity'] == 0.0000]
CleanUpResponses_Trump.shape
cond1 = Trump_Responses['Sentiment_Polarity'].isin(CleanUpResponses_Trump['Sentiment_Polarity'])
Trump_Responses.drop(Trump_Responses[cond1].index, inplace = True)
Trump_Responses.shape 


CleanUpResponses_Biden = Biden_Responses[Biden_Responses['Sentiment_Polarity'] == 0.0000]
CleanUpResponses_Biden.shape
cond2 = Biden_Responses['Sentiment_Polarity'].isin(CleanUpResponses_Biden['Sentiment_Polarity'])
Biden_Responses.drop(Biden_Responses[cond2].index, inplace = True)
Biden_Responses.shape


# Normalization
np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(Trump_Responses.index, remove_n, replace=False)
df_subset_trump = Trump_Responses.drop(drop_indices)
df_subset_trump.shape


np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(Biden_Responses.index, remove_n, replace=False)
df_subset_biden = Biden_Responses.drop(drop_indices)
df_subset_biden.shape



# Visualiization
sns.distplot(df_subset_trump['Sentiment_Polarity'])
sns.boxplot([df_subset_trump.Sentiment_Polarity])
plt.show()


sns.distplot(df_subset_biden['Sentiment_Polarity'])
sns.boxplot([df_subset_biden.Sentiment_Polarity])
plt.show()

# Donald Trump
count_1 = df_subset_trump.groupby('Expression Label').count()
print(count_1)
negative_per1 = (count_1['Sentiment_Polarity'][0]/1000)*100
positive_per1 = (count_1['Sentiment_Polarity'][1]/1000)*100

# Joe Biden
count_2 = df_subset_biden.groupby('Expression Label').count()
print(count_2)
negative_per2 = (count_2['Sentiment_Polarity'][0]/1000)*100
positive_per2 = (count_2['Sentiment_Polarity'][1]/1000)*100


Politicians = ['Donald Trump', 'Joe Biden']
lis_pos = [positive_per1, positive_per2]
lis_neg = [negative_per1, negative_per2]

fig = go.Figure(data=[
    go.Bar(name='Positive', x=Politicians, y=lis_pos),
    go.Bar(name='Negative', x=Politicians, y=lis_neg)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()
      

# Donald Trump
# Most positive replies
most_positive1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == 1].text.head()
pos_txt1 = list(most_positive1)
pos1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == 1].Sentiment_Polarity.head()
pos_pol1 = list(pos1)
fig = go.Figure(data=[go.Table(columnorder = [1,2], 
                               columnwidth = [50,400],
                               header=dict(values=['Polarity','Most Positive Responses on Trump\'s Handle'],
                               fill_color='paleturquoise',
                               align='left'),
               cells=dict(values=[pos_pol1, pos_txt1],
                               fill_color='lavender',
                               align='left'))])
 
fig.show()

# Most Negative Responses
most_negative1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == -1].text.head()
neg_txt1 = list(most_negative1)
neg1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == -1].Sentiment_Polarity.head()
neg_pol1 = list(neg1)
fig = go.Figure(data=[go.Table(columnorder = [1,2],
                               columnwidth = [50,400],
                               header=dict(values=['Polarity','Most Negative Responses on Trump\'s handle'],
                               fill_color='paleturquoise',
                               align='left'),
                cells=dict(values=[neg_pol1, neg_txt1],
                           fill_color='lavender',
                           align='left'))])

fig.show()

# Joe Biden
# Most Positive replies
most_positive2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == 1].text.tail()
pos_txt2 = list(most_positive2)
pos2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == 1].Sentiment_Polarity.tail()
pos_pol2 = list(pos2)
fig = go.Figure(data=[go.Table( columnorder = [1,2],
                                columnwidth = [50,400],
                                header=dict(values=['Polarity','Most Positive Responses on Biden\'s handle'],
                                fill_color='paleturquoise',
                                align='left'),
                                cells=dict(values=[pos_pol2, pos_txt2],
                                fill_color='lavender',
                                align='left'))])

fig.show()

# Most negative replies
most_negative2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == -1].text.head()
neg_txt2 = list(most_negative2)
neg2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == -1].Sentiment_Polarity.head()
neg_pol2 = list(neg2)
fig = go.Figure(data=[go.Table( columnorder = [1,2],
                                columnwidth = [50,400],
                                header=dict(values=['Polarity','Most Negative Responses on Biden\'s handle'],
                                fill_color='paleturquoise',
                                align='left'),
                                cells=dict(values=[neg_pol2, neg_txt2],
                                fill_color='lavender',
                                align='left'))])

fig.show()

# WordCloud for Donald Trump
text = str(df_subset_biden.text)
wordcloud = WordCloud(max_font_size=100, max_words=500, scale=10, relative_scaling=.6, background_color="black", colormap = "rainbow").generate(text)

plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# WordCloud for Joe Biden
text = str(Biden_Responses.text)
wordcloud = WordCloud(max_font_size=100, max_words=500,scale=10,relative_scaling=.6,background_color="black", colormap = "rainbow").generate(text)

plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


labels =  ['Negative_Trump', 'Negative_Biden'] 
sizes = lis_neg
explode = (0.1, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle=0)
ax1.set_title('Negative tweets on both the handles')
plt.show()


labels =  ['Positive_Trump', 'Positive_Biden'] 
sizes = lis_pos
explode = (0.1, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle=0)
ax1.set_title('Positive tweets on both the handles')
plt.show()