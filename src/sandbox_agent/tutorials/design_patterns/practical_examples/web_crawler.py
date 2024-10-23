# pragma: exclude file
"""
Building a Web Crawler using the Template Method

The Template Method design pattern provides a method in a superclass, usually an abstract superclass, and defines the skeleton of an operation in terms of a number of high-level steps. These steps are methods that are often empty or default implementations which subclasses can optionally override to provide concrete behavior.

In this section, we'll be implementing a Web Crawler using the Template Method design pattern. A Web Crawler is a bot that systematically browses the World Wide Web for the purpose of indexing, data mining, and more. Let's construct a basic crawler that fetches a webpage's content and extracts all URLs from the page.

Concept of the Web Crawler with Template Method

Our abstract base class will define the template method crawl that follows a general workflow:
1. Fetch the webpage content
2. Extract the URLs
3. Store the URLs

Each of these steps will be implemented as a separate method, and subclasses will override these methods to provide concrete implementations.

Code Example in Python
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import requests

from bs4 import BeautifulSoup


class Crawler(ABC):
    """Abstract base class for web crawlers."""

    def __init__(self, url: str):
        """
        Initialize the Crawler with a starting URL.

        Args:
            url (str): The starting URL for the crawler.
        """
        self.url = url

    def crawl(self) -> None:
        """
        Perform the crawling operation.

        This is the template method that defines the skeleton of the crawling operation.
        """
        content = self.fetch_content()
        urls = self.extract_urls(content)
        self.store_urls(urls)

    @abstractmethod
    def fetch_content(self) -> str:
        """
        Fetch the content of the webpage.

        This is an abstract method that must be implemented by subclasses.

        Returns:
            str: The content of the webpage.
        """
        pass

    @abstractmethod
    def extract_urls(self, content: str) -> list[str]:
        """
        Extract URLs from the webpage content.

        This is an abstract method that must be implemented by subclasses.

        Args:
            content (str): The content of the webpage.

        Returns:
            List[str]: A list of URLs extracted from the webpage content.
        """
        pass

    @abstractmethod
    def store_urls(self, urls: list[str]) -> None:
        """
        Store the extracted URLs.

        This is an abstract method that must be implemented by subclasses.

        Args:
            urls (List[str]): A list of URLs to store.
        """
        pass


class SimpleCrawler(Crawler):
    """A simple web crawler that fetches content, extracts URLs, and prints them."""

    def fetch_content(self) -> str:
        """
        Fetch the content of the webpage.

        Returns:
            str: The content of the webpage.
        """
        response = requests.get(self.url)
        return response.text

    def extract_urls(self, content: str) -> list[str]:
        """
        Extract URLs from the webpage content.

        Args:
            content (str): The content of the webpage.

        Returns:
            List[str]: A list of URLs extracted from the webpage content.
        """
        soup = BeautifulSoup(content, "html.parser")
        urls = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and not href.startswith("#"):
                urls.append(href)
        return urls

    def store_urls(self, urls: list[str]) -> None:
        """
        Store the extracted URLs by printing them.

        Args:
            urls (List[str]): A list of URLs to store.
        """
        for url in urls:
            print(url)


# We can use our SimpleCrawler like this:
crawler = SimpleCrawler("https://www.example.com")
crawler.crawl()

"""
The Template Method pattern enables us to provide part of an algorithm in a base class and lets subclasses fill in the details. This design gives us a high level of code reuse and flexibility, allowing us to build diverse web crawlers while maintaining a consistent structure.
"""
