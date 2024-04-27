from rss_parser import RSSParser
from requests import get
import pandas as pd


def load_financial_rss_feeds():
    rss_url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"
    response = get(rss_url)

    rss = RSSParser.parse(response.text)

    latest_rss_feeds = [f'{item.title.content}\n{item.link.content}' for item in rss.channel.items]

    dataframe = pd.DataFrame({
        'feeds': latest_rss_feeds
    })

    return dataframe

def load_financial_rss_feeds_dict() -> dict:
    rss_url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"
    response = get(rss_url)

    rss = RSSParser.parse(response.text)

    latest_rss_feeds = [{'content': f'*:gray[[{item.pub_date.content}]]*  \n{item.title.content}', 'link': f'For more details: [Click here]({item.link.content})'} for item in rss.channel.items]

    return latest_rss_feeds
