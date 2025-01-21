import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class JournalImpactFactorScraper:
    def __init__(self, url="https://ooir.org/journals.php?metric=jif", delay=1):
        """
        Initializes the scraper with the target URL and a delay between requests.
        """
        self.url = url
        self.delay = delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        self.journals = []
        self.impact_factors = []

    def fetch_page(self):
        """
        Retrieves the HTML content of the page while following best practices.
        """
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving data: {e}")
            return None

    def parse_data(self, html):
        """
        Extracts journal names and their Impact Factors from the HTML.
        """
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")[1:]  # Skip the header

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                journal_name = cols[1].text.strip()
                impact_factor = cols[2].text.strip()

                self.journals.append(journal_name)
                self.impact_factors.append(impact_factor)

            time.sleep(self.delay)  # Pause to avoid overloading the server

    def save_to_csv(self, filename="impact_factors.csv"):
        """
        Saves the extracted data to a CSV file.
        """
        df = pd.DataFrame({"journal_name": self.journals, "impact_factor": self.impact_factors})
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"File '{filename}' has been successfully created.")

    def run(self):
        """
        Runs the scraper through all necessary steps.
        """
        print("Starting scraping process...")
        html = self.fetch_page()
        if html:
            self.parse_data(html)
            self.save_to_csv()
        else:
            print("Failed to retrieve the page. Check your connection or the URL.")
