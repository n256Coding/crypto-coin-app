from rss_parser import RSSParser
from requests import get

from config import RSS_FEED_SOURCE_URL


def load_financial_rss_feeds_dict() -> dict:
    response = get(RSS_FEED_SOURCE_URL)

    rss = RSSParser.parse(response.text)

    latest_rss_feeds = [
        {
            'content': f'*:gray[[{item.pub_date.content}]]*  \n{item.title.content}', 
            'link': f'For more details: [Click here]({item.link.content})'
        } for item in rss.channel.items
    ]

    return latest_rss_feeds
