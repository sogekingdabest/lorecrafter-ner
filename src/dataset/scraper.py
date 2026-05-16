import logging
import random
import time
from pathlib import Path
import urllib.robotparser
from urllib.parse import urlparse
import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LoreScraper:
    def __init__(self, config_path="configs/llm_generation.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["scraper"]
        
        self.output_path = self.config["output_path"]
        self.max_chars = self.config.get("max_chars_per_page", 3000)
        self.delay = self.config.get("delay", 2.0)
        
        self.session = self._create_session()
        self.robot_parsers = {}
        
        # User-Agent representing our crawler
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 LoreCrafterBot/1.0"
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _create_session(self):
        """Create a requests session with robust retry logic."""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _check_robots_txt(self, url):
        """Check if we are allowed to scrape the URL according to robots.txt"""
        parsed_url = urlparse(url)
        
        # API endpoints are typically meant for programmatic access
        if "api.php" in parsed_url.path:
            return True
            
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self.robot_parsers:
            robots_url = f"{base_url}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robot_parsers[base_url] = rp
                logger.info(f"Loaded robots.txt for {base_url}")
            except Exception as e:
                logger.warning(f"Failed to read robots.txt for {base_url}: {e}. Assuming allowed.")
                self.robot_parsers[base_url] = None
                
        rp = self.robot_parsers[base_url]
        if rp is None:
            return True
            
        return rp.can_fetch(self.user_agent, url)

    def _fetch_with_delay(self, url, verify=True, params=None):
        """Fetch a URL with random delay to avoid rate limiting."""
        if not self._check_robots_txt(url):
            logger.warning(f"Scraping disallowed by robots.txt for URL: {url}")
            return None
            
        # Randomize delay slightly to look more human (e.g., +/- 50% of base delay)
        actual_delay = self.delay * random.uniform(0.5, 1.5)
        logger.debug(f"Sleeping for {actual_delay:.2f}s before fetching {url}")
        time.sleep(actual_delay)
        
        try:
            response = self.session.get(url, timeout=30, verify=verify, params=params)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _clean_html_text(self, html_content):
        """Clean HTML content to extract readable text."""
        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        paragraphs = soup.find_all(["p", "li", "td", "h2", "h3"])
        texts = [p.get_text(separator=" ", strip=True) for p in paragraphs]
        full_text = " ".join(filter(None, texts))

        sentences = full_text.replace("\n", " ").split(". ")
        cleaned = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) < self.max_chars:
                current += sentence + ". "
            else:
                if current.strip():
                    cleaned.append(current.strip())
                current = sentence + ". "
        if current.strip():
            cleaned.append(current.strip())

        return " ".join(cleaned[:5])

    def scrape_mediawiki_api(self, base_url, pages):
        """Scrape data using MediaWiki API (e.g. Fandom LOTR Wiki)"""
        texts = []
        for page in pages:
            logger.info(f"Scraping (API): {page}")
            params = {
                "action": "parse",
                "page": page,
                "prop": "text",
                "format": "json"
            }
            response = self._fetch_with_delay(base_url, params=params)
            
            if response:
                data = response.json()
                
                if "error" in data:
                    logger.warning(f"  -> Page {page} error: {data['error'].get('info', 'Unknown error')}")
                    continue
                    
                html_content = data.get("parse", {}).get("text", {}).get("*", "")
                
                if not html_content:
                    logger.warning(f"  -> Page {page} not found or no HTML available.")
                    continue
                    
                extract = self._clean_html_text(html_content)
                
                if len(extract) > 100:
                    texts.append(extract)
                    logger.info(f"  -> {len(extract)} chars extracted")
                else:
                    logger.warning(f"  -> Skipped {page} (too short)")
        return texts

    def scrape_html_site(self, base_url, pages=None):
        """Scrape regular HTML sites (like D&D SRD)"""
        texts = []
        # Some sources might not use pages, just the base_url
        if not pages or pages == ["srd"]:
            urls = [base_url]
        else:
            urls = [f"{base_url}{page}" for page in pages]

        for url in urls:
            logger.info(f"Scraping (HTML): {url}")
            response = self._fetch_with_delay(url, verify=False)
            
            if response:
                cleaned = self._clean_html_text(response.text)
                if cleaned and len(cleaned) > 100:
                    texts.append(cleaned)
                    logger.info(f"  -> {len(cleaned)} chars extracted")
                else:
                    logger.warning(f"  -> Skipped {url} (too short or empty)")
        return texts

    def run(self):
        all_texts = []

        for source in self.config["sources"]:
            name = source["name"]
            base_url = source["base_url"]
            pages = source.get("pages", [])
            
            if name == "lotr_fandom":
                texts = self.scrape_mediawiki_api(base_url, pages)
                all_texts.extend(texts)
            elif name == "dnd_srd":
                texts = self.scrape_html_site(base_url, pages)
                all_texts.extend(texts)
            else:
                logger.warning(f"Unknown source name: {name}")

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for text in all_texts:
                f.write(text.strip() + "\n\n---SEPARATOR---\n\n")

        logger.info(f"Scraping complete: {len(all_texts)} text blocks saved to {self.output_path}")
        total_chars = sum(len(t) for t in all_texts)
        logger.info(f"Total characters: {total_chars}")
        return all_texts

if __name__ == "__main__":
    scraper = LoreScraper()
    scraper.run()
