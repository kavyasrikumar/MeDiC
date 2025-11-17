import requests
from bs4 import BeautifulSoup
import json
import unicodedata
import os

def extract_text_from_url(url, document_id, output_dir="annotations"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script/style and common non-content elements
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        # Extract text from paragraphs (main content)
        paragraphs = soup.find_all('p')
        raw_text = "\n".join(p.get_text() for p in paragraphs)

        # Normalize unicode (NFKC)
        raw_text = unicodedata.normalize('NFKC', raw_text)

        # Clean up excessive whitespace but preserve newlines between paragraphs
        lines = [ " ".join(line.split()) for line in raw_text.splitlines() if line.strip() ]
        clean_text = "\n".join(lines)

        # Prepare annotation JSON structure
        annotation_data = {
            "document_id": document_id,
            "text": clean_text,
            "annotations": []
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save to file
        output_path = os.path.join(output_dir, f"{document_id}_annotations.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)

        print(f"Saved extracted text and annotation template to: {output_path}")
        return clean_text, output_path

    except Exception as e:
        print(f"Failed to extract from {url}: {e}")
        return None, None

if __name__ == "__main__":
    url = input("Enter the URL to scrape: ").strip()
    document_id = input("Enter a document ID (unique filename): ").strip()

    raw_text, filepath = extract_text_from_url(url, document_id)
    if raw_text:
        print(f"\n--- Extracted Text Preview ---\n{raw_text[:500]}...\n")
