import re
from datetime import datetime
from urllib.parse import urlparse

import dateparser
import scrapy

from dragline.items import Gazette


class FecamGazetteSpider(scrapy.Spider):

    name = "fecam"
    URL = "https://www.diariomunicipal.sc.gov.br/site/"
    total_pages = None

    def __init__(self, start_date=None, end_date=None, category=None, *args, **kwargs):
        super(FecamGazetteSpider, self).__init__(*args, **kwargs)

        self.category = category
        self.start_date = None
        self.end_date = None

        if start_date is not None:
            try:
                self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                self.logger.info(f"Collecting gazettes from {self.start_date}")
            except ValueError:
                self.logger.exception(
                    f"Unable to parse {start_date}. Use %Y-%m-d date format."
                )
                raise
        else:
            self.logger.info("Collecting all gazettes available from the beginning")

        if end_date is not None:
            try:
                self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                self.logger.info(f"Collecting gazettes until {self.end_date}")
            except ValueError:
                self.logger.exception(
                    f"Unable to parse {end_date}. Use %Y-%m-d date format."
                )
                raise
        elif hasattr(self, "end_date"):
            self.logger.info(f"Collecting gazettes until {self.end_date}")
        else:
            self.end_date = datetime.today().date()
            self.logger.info("Collecting all gazettes available until today")

    def _build_query_string(self, search_page=None):

        query = "q="
        if self.category is not None:
            query = f"{query}+categoria:{self.category}"
        start_date_string = "*"
        end_date_string = "*"
        if self.start_date is not None:
            start_date_string = self.start_date.strftime("%Y-%m-%dT00:00:00Z")
        if self.end_date is not None:
            end_date_string = self.end_date.strftime("%Y-%m-%dT23:59:00Z")
        query = f"{query}+data:[{start_date_string}+TO+{end_date_string}]"

        if search_page is not None:
            query = f"{query}&AtoASolrDocument_page={search_page}"

        self.logger.debug(query)
        return query

    def start_requests(self):
        query_string = self._build_query_string()
        yield scrapy.Request(
            f"{self.URL}?{query_string}", callback=self.parse_pagination
        )

    def parse_pagination(self, response):
        """
        This parse function is used to get all the pages available and
        return request object for each one
        """

        requests = []
        for i in range(1, self.get_last_page(response) + 1):
            query_string = self._build_query_string(i)
            requests.append(
                scrapy.Request(f"{self.URL}?{query_string}", callback=self.parse)
            )
        return requests

    def parse(self, response):
        """
        Parse each page from the gazette page.
        """
        documents = self.get_documents(response)
        for d in documents:
            yield self.get_gazette(d)

    def get_file_link(self, title):
        """
        Get the file link from the gazette's title.
        """
        link_from_title = self.get_file_link_from_title(title)
        if urlparse(link_from_title).hostname is not None:
            return link_from_title
        return self.get_file_link_from_metadata(title)

    def get_file_link_from_title(self, title):
        return title.xpath("./a/@href").get().strip()

    def get_file_link_from_metadata(self, title):
        title_sibling_link = title.xpath("following-sibling::a[2]")
        if "[Abrir/Salvar Original]" in title_sibling_link.xpath("./text()").get():
            link = title_sibling_link.xpath("./@href").get().strip()
            self.logger.debug(f"Metadata link: {link}")
            return link

    def is_autopublicacao(self, metadata):
        """
        Checks if the item from the given metadadata is an "autopublicação".
        """
        autopublicacao = metadata.xpath("./span/text()").get()
        return (
            autopublicacao is not None
            and autopublicacao.strip().lower() == "autopublicação"
        )

    def get_documents(self, response):
        """
        Method to get all the relevant documents list and their dates from the page
        """
        documents = []
        titles = response.css("div.row.no-print h4")
        for title in titles:
            metadata = title.xpath("following-sibling::span[1]")

            link = self.get_file_link(title)
            date = metadata.re_first(r"\d{2}/\d{2}/\d{4}").strip()
            category = metadata.xpath("./text()").get().split("-")[1].strip().lower()
            entity = metadata.xpath("./text()").get().split("-")[-1].strip().lower()
            is_autopublicacao = self.is_autopublicacao(metadata)
            title = title.xpath("./a/text()").get()
            documents.append(
                {
                    "title": title,
                    "link": link,
                    "date": date,
                    "category": category,
                    "entity": entity,
                    "is_autopublicacao": is_autopublicacao,
                    "scraped_at": datetime.utcnow()
                }
            )
        return documents

    @staticmethod
    def get_last_page(response):
        """
        Get the last page number available in the pages navigation menu
        """
        href = response.css(
            "div.pagination.pagination-centered ul#yw4 li.last a::attr(href)"
        ).get()
        result = re.search(r"AtoASolrDocument_page=(\d+)", href)
        if result is not None:
            return int(result.groups()[0])

    def get_gazette(self, document):
        """
        Transform the tuple returned by get_documents_links_date and returns a
        Gazette item
        """
        self.logger.debug(f"Creating gazette: {document}")
        if "date" not in document:
            raise "Missing document date"
        if "link" not in document:
            raise "Missing document URL"

        return Gazette(
            date=dateparser.parse(document["date"], languages=("pt",)).date(),
            file_link=(document["link"],),
            power="unknown",
            category=document.get("category", "unknown"),
            entity=document.get("entity", "unknown"),
            autopublicacao=document.get("is_autopublicacao", False),
            title=document.get("title", "")
        )
