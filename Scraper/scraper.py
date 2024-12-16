import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")

def json_to_csv(input_json_path, output_csv_path):
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

def fetch_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception if the request fails
    return response.text  # Return the HTML content of the page

class PubMedScraper:
    def __init__(self, pages, delay):
        self.pages = pages  # Number of pages to scrape
        self.delay = delay
        self.url = None  # Url is set in main.py

    def parse_article_data(self, article):
        try:
            # Extract title
            title = article.find("a", class_="docsum-title").text.strip()
            # Construct the full URL for the article
            article_url = "https://pubmed.ncbi.nlm.nih.gov" + article.find("a", class_="docsum-title")["href"]

            # Fetch the article page for detailed data
            article_page = BeautifulSoup(fetch_page(article_url), "html.parser")

            # Extract abstract
            abstract = article_page.find("div", class_="abstract-content").text.strip() if article_page.find("div", class_="abstract-content") else "No abstract found"

            # Extract authors
            authors = ", ".join([author.text.strip() for author in article_page.find_all("span", class_="authors-list-item")]) if article_page.find_all("span", class_="authors-list-item") else "No authors found"

            # Extract publication year
            year = article_page.find("span", class_="cit").text.strip().split(";")[0] if article_page.find("span", class_="cit") else "Year not precised"

            # Extract the journal name
            journal = article_page.find("button", class_="journal-actions-trigger").text.strip() if article_page.find("button", class_="journal-actions-trigger") else "Journal not precised"

            # Extract keywords
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
        if not self.url:
            raise ValueError("URL not set")

        data = []
        parsed_url = urlparse(self.url)  # Parse the base URL into components
        query_params = parse_qs(parsed_url.query)  # Extract query parameters

        # Loop through the specified number of pages
        for page in range(1, self.pages + 1):
            print(f"Scraping page {page}...")
            query_params['page'] = [str(page)]
            new_query = urlencode(query_params, doseq=True)  # Reconstruct the query string
            current_url = urlunparse(parsed_url._replace(query=new_query))  # Build the new URL

            try:
                soup = BeautifulSoup(fetch_page(current_url), "html.parser")

                # Find all articles on the current page
                for article in soup.find_all("article", class_="full-docsum"):
                    try:
                        article_data = self.parse_article_data(article)
                        data.append(article_data)
                    except Exception as e:
                        print(f"Error parsing article: {e}")

                time.sleep(self.delay)  # Delay

            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break

        return data

