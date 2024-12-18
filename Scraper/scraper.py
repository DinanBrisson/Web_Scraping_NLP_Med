import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def save_to_json(data, filename):
    """
    Save data to a JSON file.

    Parameters:
    data (list): The data to save in JSON format.
    filename (str): The path and filename where to save the data.

    Returns:
    None
    """
    with open(filename, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")

def json_to_csv(input_json_path, output_csv_path):
    """
    Convert a JSON file to a CSV file.

    Parameters:
    input_json_path (str): Path to the input JSON file.
    output_csv_path (str): Path to the output CSV file.

    Returns:
    None
    """
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

def fetch_page(url):
    """
    Fetch the HTML content of a web page.

    Parameters:
    url (str): The URL of the web page to fetch.

    Returns:
    str: HTML content of the page.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception if the request fails
    return response.text

class PubMedScraper:
    """
    A class to scrape articles from PubMed.

    Attributes:
    pages (int): Number of pages to scrape.
    delay (int): Delay in seconds between requests.
    url (str): Base URL for scraping (set externally).
    """
    def __init__(self, pages, delay):
        """
        Initialize the scraper with the number of pages to scrape and delay.

        Parameters:
        pages (int): Number of pages to scrape.
        delay (int): Delay in seconds between requests.
        """
        self.pages = pages
        self.delay = delay
        self.url = None  # Url is set externally.

    def parse_article_data(self, article):
        """
        Parse individual article data from a BeautifulSoup object.

        Parameters:
        article (BeautifulSoup object): HTML content of the article.

        Returns:
        dict: A dictionary containing the parsed article data.
        """
        try:
            title = article.find("a", class_="docsum-title").text.strip()
            article_url = "https://pubmed.ncbi.nlm.nih.gov" + article.find("a", class_="docsum-title")["href"]
            article_page = BeautifulSoup(fetch_page(article_url), "html.parser")
            abstract = article_page.find("div", class_="abstract-content").text.strip() if article_page.find("div", class_="abstract-content") else "No abstract found"
            authors = ", ".join([author.text.strip() for author in article_page.find_all("span", class_="authors-list-item")]) if article_page.find_all("span", class_="authors-list-item") else "No authors found"
            year = article_page.find("span", class_="cit").text.strip().split(";")[0] if article_page.find("span", class_="cit") else "Year not precised"
            journal = article_page.find("button", class_="journal-actions-trigger").text.strip() if article_page.find("button", class_="journal-actions-trigger") else "Journal not precised"
            keywords = "No keywords found"
            keywords_tag = article_page.find("strong", class_="sub-title")
            if keywords_tag and "Keywords:" in keywords_tag.text:
                keywords_paragraph = keywords_tag.find_parent("p")
                if keywords_paragraph:
                    keywords = ", ".join(
                        [kw.strip() for kw in keywords_paragraph.get_text(strip=True).replace("Keywords:", "").split(";")]
                    )
            return {
                "Title": title,
                "Abstract": abstract,
                "Authors": authors,
                "Year": year,
                "Journal": journal,
                "Keywords": keywords,
                "URL": article_url,
            }
        except Exception as e:
            print(f"Error parsing article data: {e}")
            return {
                "Title": "Error",
                "Abstract": "Error",
                "Authors": "Error",
                "Year": "Error",
                "Journal": "Error",
                "Keywords": "Error",
                "URL": "Error",
            }

    def scrape_articles(self):
        """
        Scrape articles from PubMed based on the set URL and number of pages.

        Returns:
        list: A list of dictionaries containing article data.
        """
        if not self.url:
            raise ValueError("URL not set")

        data = []
        parsed_url = urlparse(self.url)
        query_params = parse_qs(parsed_url.query)

        for page in range(1, self.pages + 1):
            print(f"Scraping page {page}...")
            query_params['page'] = [str(page)]
            new_query = urlencode(query_params, doseq=True)
            current_url = urlunparse(parsed_url._replace(query=new_query))

            try:
                soup = BeautifulSoup(fetch_page(current_url), "html.parser")
                for article in soup.find_all("article", class_="full-docsum"):
                    try:
                        article_data = self.parse_article_data(article)
                        data.append(article_data)
                    except Exception as e:
                        print(f"Error parsing article: {e}")
                time.sleep(self.delay)
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break

        return data
