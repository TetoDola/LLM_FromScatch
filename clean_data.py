import os
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

BLOCK_LEVEL_TAGS = ['p', 'div', 'blockquote', 'li', 'section']


def extract_paragraphs_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    all_paragraphs = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')

            for tag_name in BLOCK_LEVEL_TAGS:
                for block in soup.find_all(tag_name):
                    paragraph_text = block.get_text(strip=True)
                    if paragraph_text:
                        all_paragraphs.append(paragraph_text)
    return all_paragraphs


def reflow_paragraph(paragraph, max_width=80):
    words = paragraph.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + (1 if current_line else 0) > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def reflow_all_paragraphs(paragraphs, max_width=80):
    reflowed = [reflow_paragraph(p, max_width) for p in paragraphs]
    return "\n\n".join(reflowed)


##############################
# Main script logic below
##############################

def main():
    BASE_DIR = 'C:\\Users\\Teto\\Downloads\\Officially Translated Light Novels'  # or wherever your ebooks are
    OUTPUT_FILE = 'all_epubs.txt'
    processed_epubs = set()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        for root, dirs, files in os.walk(BASE_DIR):
            for filename in files:
                file_path = os.path.join(root, filename)

                # 2) Process EPUBs
                if filename.lower().endswith('.epub'):
                    if file_path in processed_epubs:
                        continue
                    processed_epubs.add(file_path)

                    print(f"\nProcessing EPUB: {file_path}")
                    try:
                        paragraphs = extract_paragraphs_from_epub(file_path)
                        if paragraphs:
                            text_output = reflow_all_paragraphs(paragraphs, max_width=80)

                            out_file.write(text_output)
                            print(f"Extracted {len(paragraphs)} paragraphs from {file_path}")
                        else:
                            print(f"No paragraphs extracted from {file_path}")
                    except Exception as e:
                        print(f"Error processing EPUB {file_path}: {e}")

    print(f"\nAll EPUB text has been combined into '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
