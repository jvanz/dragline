# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Gazette(scrapy.Item):
    date = scrapy.Field()
    power = scrapy.Field()
    scraped_at = scrapy.Field()
    file_urls = scrapy.Field()
    files = scrapy.Field()
    category = scrapy.Field()
    entity = scrapy.Field()
    autopublicacao = scrapy.Field()
    title = scrapy.Field()
