# util_funcs.py
import requests
import streamlit as st
import json
from bs4 import BeautifulSoup
import urllib.parse

# Configuration
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

def advanced_website_scraper(url):
    """Advanced website scraping with better content extraction"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Extract structured data
        extracted_data = {
            'title': '',
            'meta_description': '',
            'headings': [],
            'key_phrases': [],
            'content_blocks': []
        }
        
        # Title
        title = soup.find('title')
        if title:
            extracted_data['title'] = title.get_text().strip()
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            extracted_data['meta_description'] = meta_desc.get('content', '').strip()
        
        # Headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            text = heading.get_text().strip()
            if text and len(text) > 5:
                extracted_data['headings'].append(text)
        
        # Key content blocks (paragraphs with substantial text)
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 100:  # Only substantial paragraphs
                extracted_data['content_blocks'].append(text)
        
        # Extract potential technology keywords
        technology_keywords = extract_technology_keywords(str(soup))
        extracted_data['key_phrases'] = technology_keywords
        
        return extracted_data
        
    except Exception as e:
        return {'error': str(e)}

def extract_technology_keywords(html_content):
    """Extract technology-related keywords from content"""
    tech_keywords = [
        'AI', 'artificial intelligence', 'machine learning', 'blockchain', 'cloud',
        'IoT', 'internet of things', 'API', 'algorithm', 'software', 'hardware',
        'data', 'analytics', 'platform', 'system', 'method', 'process', 'device',
        'application', 'framework', 'architecture', 'protocol', 'interface'
    ]
    
    found_keywords = []
    content_lower = html_content.lower()
    
    for keyword in tech_keywords:
        if keyword.lower() in content_lower:
            found_keywords.append(keyword)
    
    return list(set(found_keywords))  # Remove duplicates

def generate_competitive_analysis(website_data, competitor_urls=None):
    """Generate competitive analysis from multiple websites"""
    analysis = {
        'primary_analysis': website_data,
        'technology_landscape': [],
        'innovation_gaps': [],
        'market_positioning': ''
    }
    
    # Analyze technology landscape from keywords
    if 'key_phrases' in website_data:
        analysis['technology_landscape'] = website_data['key_phrases']
    
    # Identify potential innovation gaps
    if 'content_blocks' in website_data:
        content_text = ' '.join(website_data['content_blocks'][:5])
        analysis['innovation_gaps'] = identify_innovation_gaps(content_text)
    
    return analysis

def identify_innovation_gaps(content):
    """Identify potential innovation gaps from content analysis"""
    # This would typically use more advanced NLP
    # For now, return placeholder analysis
    return [
        "Potential integration opportunities",
        "Scalability improvements needed",
        "User experience enhancements",
        "Technical performance optimizations"
    ]

def save_website_analysis(analysis_data, filename=None):
    """Save website analysis to JSON file"""
    if not filename:
        from datetime import datetime
        filename = f"website_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    return filename

def validate_website_content(extracted_data):
    """Validate that sufficient content was extracted"""
    validation = {
        'has_title': bool(extracted_data.get('title')),
        'has_content': len(extracted_data.get('content_blocks', [])) > 0,
        'content_quality': len(extracted_data.get('content_blocks', [])) >= 3,
        'word_count': sum(len(block.split()) for block in extracted_data.get('content_blocks', []))
    }
    
    return validation

def get_analysis_statistics():
    """Get statistics about website analyses"""
    try:
        with open("patent_usage_tracker.json", "r") as f:
            data = json.load(f)
        
        return {
            "total_analyses": data.get("total_patents_generated", 0),
            "current_hour_usage": data.get("usage_count", 0),
            "recent_websites": [entry.get('patent_title', 'Unknown') for entry in data.get('usage_history', [])[-5:]]
        }
    except FileNotFoundError:
        return {"total_analyses": 0, "current_hour_usage": 0, "recent_websites": []}