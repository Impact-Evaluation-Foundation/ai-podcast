import asyncio
import json
import re
import logging
from typing import Dict, List, Set
from urllib.parse import urlparse

import aiohttp

def save_to_file(file_path, content):
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(content)
    print(f"Saved content to {file_path}")

def handle_question_generator(llm_output: str) -> Dict[str, str]:
    """
    Parses LLM output for question generator prompts and extracts structured data.
    Expected format:
        [Any optional header text]
        
        1. First question?
        2. Second question?
        ...
        10. Tenth question?
        
        Tavily Search Query: [query text]
    
    Returns:
        Dictionary with question_1->question_10 and tavily_question keys
    """
    result = {f"question_{i}": None for i in range(1, 11)}
    result["tavily_question"] = None
    
    # Normalize line endings and split into lines
    lines = llm_output.replace('\r\n', '\n').split('\n')
    
    # Extract numbered questions (1-10)
    question_pattern = re.compile(r'^\s*(\d+)\.?\s+(.+?)\s*$')
    found_questions = 0
    
    for line in lines:
        match = question_pattern.match(line)
        if match:
            q_num = int(match.group(1))
            q_text = match.group(2).strip()
            
            if 1 <= q_num <= 10:
                result[f"question_{q_num}"] = q_text
                found_questions += 1
            elif q_num > 10:
                logging.warning(f"Ignoring extra question #{q_num} beyond 10")
    
    # Extract Tavily search query using multiple possible patterns
    tavily_patterns = [
        r'^Tavily Search Query:\s*(.+?)\s*$',
        r'^Search Query:\s*(.+?)\s*$',
        r'^Query:\s*(.+?)\s*$'
    ]
    
    for pattern in tavily_patterns:
        matches = re.search(pattern, llm_output, re.MULTILINE | re.IGNORECASE)
        if matches:
            result["tavily_question"] = matches.group(1).strip()
            break
    
    # Validation and error reporting
    if found_questions < 5:
        logging.error(f"Only found {found_questions} questions in output")
    if not result["tavily_question"]:
        logging.warning("No Tavily query found in output")
    
    # Fill missing questions with placeholders
    for i in range(1, 11):
        if not result[f"question_{i}"]:
            result[f"question_{i}"] = f"Question {i} not generated"
            logging.info(f"Added placeholder for missing question {i}")
    
    return result

async def check_urls(tavily_references: str, perplexity_references: str) -> List[str]:
    """
    Validate URLs from both reference sources and return non-working URLs.
    Handles redirects, timeouts, and various error cases.
    """
    # Unified URL validation with retry logic
    async def validate_url(session, url, sem):
        async with sem:
            try:
                # Validate URL format first
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    return url  # Invalid format counts as bad URL

                # Use HEAD for faster validation (fallback to GET if needed)
                async with session.head(
                    url,
                    allow_redirects=True,
                    timeout=aiohttp.ClientTimeout(total=15),
                    ssl=False  # Allows inspection of sites with SSL issues
                ) as response:
                    if 400 <= response.status < 600:
                        return url
                    return None

            except aiohttp.ClientError as e:
                return url  # Consider connection errors as invalid
            except Exception as e:
                return url  # Broad catch for other errors

    # Extract unique URLs using improved regex
    def extract_urls(text: str) -> Set[str]:
        url_pattern = r'\bhttps?://[^\s<>"]+|www\.[^\s<>"]+'
        return set(re.findall(url_pattern, text, re.IGNORECASE))

    # Get all unique URLs from both sources
    all_urls = extract_urls(tavily_references or "") | extract_urls(perplexity_references or "")

    # Configure concurrent validation
    sem = asyncio.Semaphore(15)  # Concurrent request limit
    bad_urls = []

    async with aiohttp.ClientSession(
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    ) as session:
        tasks = [validate_url(session, url, sem) for url in all_urls]
        results = await asyncio.gather(*tasks)
        bad_urls = [url for url in results if url is not None]

    return bad_urls