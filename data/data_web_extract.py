import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script/style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose()

        # Extract text from main content tags
        paragraphs = soup.find_all('p')
        text = "\n".join(p.get_text() for p in paragraphs)
        
        # Clean up whitespace
        text = " ".join(text.split())
        return text

    except Exception as e:
        print(f"Failed to extract from {url}: {e}")
        return None

# Example usage
url = "https://www.mayoclinic.org/diseases-conditions/heart-attack/symptoms-causes/syc-20373106"
raw_text = extract_text_from_url(url)
print(raw_text[:500])  # print first 500 chars
