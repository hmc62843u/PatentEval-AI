# patent_infringement_scanner.py
import streamlit as st
import json
import requests
from pathlib import Path
from datetime import datetime
import re
from bs4 import BeautifulSoup
import urllib.parse
from ddgs import DDGS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import groq

# Configure Groq client
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = groq.Client(api_key=GROQ_API_KEY)
    GROQ_MODEL = "qwen/qwen3-32b"
except (KeyError, FileNotFoundError):
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets. Please configure it in the Secrets tab.")
    st.stop()

def groq_generate_content(prompt, model=GROQ_MODEL):
    """Generate content using Groq API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {str(e)}")
        return f"Error generating content: {str(e)}"

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file based on file type"""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            # Simple PDF text extraction - in production, use PyPDF2 or pdfplumber
            content = uploaded_file.getvalue()
            # Extract text between parentheses and brackets as fallback
            text = re.sub(r'[^\x00-\x7F]+', ' ', content.decode('latin-1'))
            return text[:5000]  # Limit length
        else:
            return f"File content: {uploaded_file.name}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def combine_patent_documents(specification_text, claims_text, drawings_text):
    """Combine all patent documents into a comprehensive analysis text"""
    combined_text = f"""
    PATENT SPECIFICATION:
    {specification_text[:3000]}
    
    PATENT CLAIMS:
    {claims_text[:2000]}
    
    DRAWINGS DESCRIPTION:
    {drawings_text[:1000]}
    """
    return combined_text

# Industry Detection Functions (from reference)
def keyword_industry_detection(patent_text):
    """Tier 1: Fast keyword-based industry detection"""
    combined_text = patent_text.lower()
    
    industry_taxonomy = {
        'fintech': ['financial', 'banking', 'payment', 'fintech', 'investment', 'lending', 'insurance', 'wealth', 'portfolio', 'transaction', 'digital wallet', 'cryptocurrency'],
        'healthtech': ['healthcare', 'medical', 'patient', 'clinical', 'telemedicine', 'biotech', 'pharmaceutical', 'diagnostic', 'health tech', 'medical device', 'hospital', 'treatment'],
        'edtech': ['education', 'learning', 'course', 'student', 'educational', 'online learning', 'edtech', 'curriculum', 'teaching', 'academic', 'school', 'university'],
        'saas': ['software', 'service', 'cloud', 'subscription', 'enterprise', 'platform', 'api', 'integration', 'dashboard', 'workflow', 'automation', 'business intelligence'],
        'iot': ['internet of things', 'iot', 'connected', 'sensor', 'smart device', 'embedded', 'wireless', 'smart home', 'industrial iot', 'sensor network'],
        'ai_ml': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'ai model', 'algorithm', 'predictive', 'natural language', 'computer vision', 'llm'],
        'ecommerce': ['ecommerce', 'e-commerce', 'online store', 'shopping cart', 'marketplace', 'retail', 'inventory', 'checkout', 'product catalog', 'digital storefront'],
        'cleantech': ['renewable', 'solar', 'wind', 'energy', 'sustainability', 'green tech', 'carbon', 'environmental', 'clean energy', 'climate', 'emissions']
    }
    
    industry_scores = {}
    for industry, keywords in industry_taxonomy.items():
        score = 0
        for keyword in keywords:
            if keyword in combined_text:
                score += len(keyword.split()) * 2
        if score > 0:
            industry_scores[industry] = score
    
    return dict(sorted(industry_scores.items(), key=lambda x: x[1], reverse=True)[:5])

def llm_industry_classifier(patent_text):
    """Tier 2: LLM-based industry classification using Groq"""
    prompt = f"""
    Analyze this patent document to determine the primary industry.
    
    PATENT DOCUMENT:
    {patent_text[:2000]}
    
    Classify into ONE primary industry from this list:
    - fintech (financial technology, banking, payments)
    - healthtech (healthcare, medical technology)
    - edtech (education technology)
    - saas (software as a service, enterprise software)
    - iot (internet of things, connected devices)
    - ai_ml (artificial intelligence, machine learning)
    - ecommerce (online retail, marketplaces)
    - cleantech (renewable energy, sustainability)
    - other (if none of the above fit well)
    
    Respond in this exact JSON format:
    {{
        "primary_industry": "industry_name",
        "confidence": "High/Medium/Low",
        "reasoning": "brief explanation",
        "alternative_industries": ["industry1", "industry2"]
    }}
    """
    
    try:
        response = groq_generate_content(prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return json.loads(response)
    except Exception as e:
        return {
            "primary_industry": "other",
            "confidence": "Low", 
            "reasoning": f"LLM analysis failed: {str(e)}",
            "alternative_industries": []
        }

def semantic_similarity_analysis(patent_text):
    """Tier 3: Semantic similarity analysis using TF-IDF"""
    try:
        # Industry domain descriptions
        industry_domains = {
            "fintech": "financial technology digital payments banking investment cryptocurrency blockchain fintech insurance wealth management",
            "healthtech": "healthcare medical technology patient care telemedicine biotech pharmaceuticals medical devices diagnostics treatment clinical",
            "edtech": "education learning educational technology online courses digital learning students teachers academic curriculum school university",
            "saas": "software as a service cloud computing enterprise business platform subscription api integration workflow automation",
            "iot": "internet of things connected devices sensors smart home industrial iot wireless embedded systems sensor networks",
            "ai_ml": "artificial intelligence machine learning neural networks deep learning algorithms predictive analytics natural language processing computer vision",
            "ecommerce": "ecommerce online shopping retail marketplace digital storefront inventory management checkout payment gateway",
            "cleantech": "renewable energy sustainability green technology environmental solar wind power carbon emissions clean energy climate"
        }
        
        # Calculate similarities using TF-IDF
        similarities = {}
        vectorizer = TfidfVectorizer()
        
        # Create document corpus
        documents = [patent_text[:2000]] + list(industry_domains.values())
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarities between patent text and each industry
        patent_vector = tfidf_matrix[0]
        for i, (industry, description) in enumerate(industry_domains.items(), 1):
            industry_vector = tfidf_matrix[i]
            similarity = cosine_similarity(patent_vector, industry_vector)[0][0]
            similarities[industry] = similarity
        
        return dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5])
        
    except Exception as e:
        st.warning(f"Semantic analysis failed: {str(e)}")
        return {}

def consolidate_industries(keyword_results, llm_results, semantic_results):
    """Consolidate results from all three tiers with weighted scoring"""
    consolidated = {}
    
    weights = {
        'llm': 0.5,
        'semantic': 0.3, 
        'keyword': 0.2
    }
    
    # Process LLM results
    if llm_results.get('primary_industry') and llm_results.get('primary_industry') != 'other':
        industry = llm_results['primary_industry']
        confidence_weight = {'High': 1.0, 'Medium': 0.7, 'Low': 0.4}.get(llm_results.get('confidence', 'Medium'), 0.5)
        consolidated[industry] = weights['llm'] * confidence_weight
    
    # Add LLM alternative industries
    for alt_industry in llm_results.get('alternative_industries', [])[:2]:
        if alt_industry in consolidated:
            consolidated[alt_industry] += weights['llm'] * 0.3
        else:
            consolidated[alt_industry] = weights['llm'] * 0.3
    
    # Process semantic results
    for industry, score in semantic_results.items():
        if industry in consolidated:
            consolidated[industry] += weights['semantic'] * score
        else:
            consolidated[industry] = weights['semantic'] * score
    
    # Process keyword results
    max_keyword_score = max(keyword_results.values()) if keyword_results else 1
    for industry, score in keyword_results.items():
        normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
        if industry in consolidated:
            consolidated[industry] += weights['keyword'] * normalized_score
        else:
            consolidated[industry] = weights['keyword'] * normalized_score
    
    # Filter and return top industries
    final_industries = dict(sorted(consolidated.items(), key=lambda x: x[1], reverse=True)[:3])
    
    # Convert to percentage confidence
    total = sum(final_industries.values()) if final_industries else 1
    return {industry: round(score/total * 100, 1) for industry, score in final_industries.items()}

def robust_industry_detection(patent_text):
    """Main tiered industry detection function for patent documents"""
    st.info("🔍 Analyzing patent industry context...")
    
    # Tier 1: Fast keyword analysis
    with st.spinner("Tier 1: Keyword analysis..."):
        keyword_industries = keyword_industry_detection(patent_text)
    
    # Tier 2: LLM classification
    with st.spinner("Tier 2: AI classification..."):
        llm_industries = llm_industry_classifier(patent_text)
    
    # Tier 3: Semantic similarity
    with st.spinner("Tier 3: Semantic analysis..."):
        semantic_industries = semantic_similarity_analysis(patent_text)
    
    # Consolidate results
    final_industries = consolidate_industries(keyword_industries, llm_industries, semantic_industries)
    
    # Display analysis results
    with st.expander("📊 Industry Analysis Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔤 Keyword Analysis**")
            for industry, score in keyword_industries.items():
                st.write(f"- {industry}: {score}")
        
        with col2:
            st.markdown("**🤖 LLM Classification**")
            st.write(f"Primary: {llm_industries.get('primary_industry', 'N/A')}")
            st.write(f"Confidence: {llm_industries.get('confidence', 'N/A')}")
            st.write(f"Alternatives: {', '.join(llm_industries.get('alternative_industries', []))}")
        
        with col3:
            st.markdown("**🎯 Semantic Similarity**")
            for industry, score in semantic_industries.items():
                st.write(f"- {industry}: {score:.3f}")
    
    return final_industries

# Competitor Search and Infringement Analysis
def scrape_website_content(url):
    """Scrape and extract meaningful content from a website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract meaningful content
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        content_parts = []
        
        # Try to get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            content_parts.append(f"Meta Description: {meta_desc.get('content', '').strip()}")
        
        # Get headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = heading.get_text().strip()
            if heading_text and len(heading_text) > 10:
                content_parts.append(f"Heading: {heading_text}")
        
        # Get paragraph content
        paragraphs = soup.find_all('p')
        for p in paragraphs[:10]:
            text = p.get_text().strip()
            if len(text) > 50:
                content_parts.append(text)
        
        # Combine all content
        full_content = f"Website Title: {title_text}\n\n" + "\n\n".join(content_parts[:15])
        return full_content[:4000]
    
    except Exception as e:
        return f"Error scraping website: {str(e)}"

def generate_competitor_search_query(patent_text):
    """Use Groq to generate specific competitor search queries based on patent"""
    prompt = f"""
    Based on this patent document, generate 3 specific search queries to find competitor companies that might be working on similar technology.
    
    PATENT DOCUMENT:
    {patent_text[:2000]}
    
    Generate 3 search queries that are:
    1. **Specific to the technology** (not just generic industry terms)
    2. **Include company/product names** if mentioned
    3. **Focus on commercial competitors** (not academic/research)
    4. **Use natural language** that would work in search engines
    
    Return in this exact JSON format:
    {{
        "search_queries": [
            "query 1",
            "query 2", 
            "query 3"
        ],
        "reasoning": "Why these queries are likely to find relevant competitors"
    }}
    """
    
    try:
        response = groq_generate_content(prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {
                "search_queries": [
                    "technology companies competitors",
                    "industry leading companies",
                    "innovative tech startups"
                ],
                "reasoning": "Fallback queries generated"
            }
    except Exception as e:
        return {
            "search_queries": ["technology companies competitors"],
            "reasoning": "Error fallback"
        }

def find_relevant_competitors(patent_text, max_competitors=8):
    """Find competitor websites using AI-generated search queries"""
    try:
        # Generate smart search queries
        search_plan = generate_competitor_search_query(patent_text)
        
        st.info(f"🔍 Using smart search: {search_plan.get('reasoning', 'Finding relevant competitors')}")
        
        competitors = []
        seen_urls = set()
        
        with DDGS() as ddgs:
            # Try each search query
            for query in search_plan.get("search_queries", [])[:2]:
                if len(competitors) >= max_competitors:
                    break
                    
                st.write(f"  - Searching: '{query}'")
                search_results = list(ddgs.text(query, max_results=max_competitors))
                
                for result in search_results:
                    if len(competitors) >= max_competitors:
                        break
                        
                    url = result.get("href", "")
                    title = result.get("title", "")
                    
                    # Filter out non-company websites
                    if (url and 
                        any(domain in url for domain in ['.com', '.io', '.co', '.tech', '.ai']) and
                        url not in seen_urls and
                        not any(exclude in url for exclude in ['wikipedia', 'academic', 'research', 'github']) and
                        not any(exclude in title.lower() for exclude in ['wikipedia', 'academic paper', 'research paper'])):
                        
                        competitors.append({
                            "name": title if title else "Unknown Company",
                            "url": url,
                            "description": result.get("body", "")[:200] + "...",
                            "search_query": query
                        })
                        seen_urls.add(url)
        
        return competitors
        
    except Exception as e:
        st.error(f"Error finding competitors: {str(e)}")
        return []

def infringement_risk_analysis(patent_claims, competitor_url):
    """Analyze a specific competitor website for infringement risks using Groq"""
    try:
        # Scrape the competitor site
        site_content = scrape_website_content(competitor_url)
        
        if site_content.startswith("Error"):
            return {
                "competitor_url": competitor_url,
                "risk_level": "Unknown",
                "error": site_content,
                "overlapping_features": [],
                "recommendations": []
            }
        
        # Analyze for infringement risks using Groq
        analysis_prompt = f"""
        PATENT CLAIMS TO ANALYZE:
        {patent_claims}
        
        COMPETITOR WEBSITE CONTENT:
        {site_content}
        
        Analyze for potential patent infringement risks. Focus on:
        
        1. **DIRECT PRODUCT MATCH**: Does the website show products/services that directly implement the claimed invention?
        2. **FEATURE OVERLAP**: Which specific claim elements appear to be implemented in their products?
        3. **TECHNICAL SIMILARITY**: How similar are the technical approaches and implementations?
        4. **COMMERCIAL USE EVIDENCE**: Is there evidence of actual commercial use of similar technology?
        
        Provide your analysis in this exact JSON format:
        {{
            "risk_level": "Low/Medium/High",
            "overlapping_features": ["list of specific overlapping features"],
            "infringement_confidence": "Low/Medium/High",
            "key_findings": "Detailed analysis of potential infringement",
            "recommendations": ["list of recommendations for further action"]
        }}
        
        Be thorough and evidence-based in your assessment.
        """
        
        response = groq_generate_content(analysis_prompt)
        
        # Parse the JSON response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
            else:
                analysis_result = json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from text
            analysis_result = {
                "risk_level": "Medium",
                "overlapping_features": ["Unable to parse detailed features"],
                "infringement_confidence": "Medium", 
                "key_findings": response[:500] + "...",
                "recommendations": ["Conduct manual review of this competitor"]
            }
        
        # Add competitor info to result
        analysis_result["competitor_url"] = competitor_url
        analysis_result["scraped_content_preview"] = site_content[:300] + "..."
        
        return analysis_result
        
    except Exception as e:
        return {
            "competitor_url": competitor_url,
            "risk_level": "Error",
            "error": str(e),
            "overlapping_features": [],
            "recommendations": ["Analysis failed - manual review required"]
        }

def industry_wide_infringement_scan(patent_text, patent_claims, max_competitors=5):
    """Main function to scan industry for infringement risks"""
    
    # Use robust industry detection
    detected_industries = robust_industry_detection(patent_text)
    
    if not detected_industries:
        st.error("❌ Could not detect relevant industries for scanning")
        return {
            "error": "Industry detection failed",
            "scanned_competitors": 0,
            "high_risk_findings": 0,
            "detailed_results": []
        }
    
    # Show detected industries
    st.success(f"🎯 Detected Industries: {', '.join(detected_industries.keys())}")
    
    # Use the highest confidence industry as context
    primary_industry = next(iter(detected_industries))
    industry_confidence = detected_industries[primary_industry]
    
    st.info(f"🔍 Scanning {primary_industry} industry (confidence: {industry_confidence}%)...")
    
    # Find competitors using AI-generated search queries
    competitors = find_relevant_competitors(patent_text, max_competitors)
    
    if not competitors:
        st.warning("No relevant competitors found with AI search.")
        return {
            "error": f"No competitors found in {primary_industry} industry",
            "scanned_competitors": 0,
            "high_risk_findings": 0,
            "detailed_results": []
        }
    
    infringement_findings = []
    high_risk_count = 0
    
    # Analyze each competitor
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, competitor in enumerate(competitors):
        status_text.text(f"Analyzing {competitor['name']}... ({i+1}/{len(competitors)})")
        
        risk_analysis = infringement_risk_analysis(patent_claims, competitor["url"])
        infringement_findings.append({
            "competitor_info": competitor,
            "risk_analysis": risk_analysis
        })
        
        if risk_analysis.get("risk_level") == "High":
            high_risk_count += 1
        
        progress_bar.progress((i + 1) / len(competitors))
    
    status_text.text("Analysis complete!")
    
    return {
        "scanned_competitors": len(competitors),
        "high_risk_findings": high_risk_count,
        "detected_industries": detected_industries,
        "primary_industry": primary_industry,
        "detailed_results": infringement_findings,
        "search_method": "AI-Generated Queries",
        "overall_risk_level": "High" if high_risk_count > 0 else "Medium" if len(competitors) > 0 else "Low"
    }

def display_infringement_results(infringement_results, patent_claims):
    """Display infringement analysis results in Streamlit"""
    st.subheader("⚖️ Infringement Risk Analysis Results")
    
    if infringement_results.get('error'):
        st.error(f"Infringement analysis failed: {infringement_results['error']}")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Competitors Scanned", infringement_results['scanned_competitors'])
    with col2:
        st.metric("High Risk Findings", infringement_results['high_risk_findings'])
    with col3:
        st.metric("Overall Risk", infringement_results['overall_risk_level'])
    
    # Industry analysis
    st.markdown("### 🎯 Detected Industries")
    for industry, confidence in infringement_results.get('detected_industries', {}).items():
        st.write(f"- **{industry}**: {confidence}% confidence")
    
    # Detailed results
    st.markdown("### 📊 Detailed Competitor Analysis")
    
    for i, result in enumerate(infringement_results['detailed_results']):
        competitor = result['competitor_info']
        analysis = result['risk_analysis']
        
        # Color code based on risk level
        risk_color = {
            "High": "red",
            "Medium": "orange", 
            "Low": "green",
            "Error": "gray"
        }.get(analysis.get('risk_level', 'Unknown'), 'gray')
        
        with st.expander(f"🎯 {competitor['name']} - Risk: :{risk_color}[{analysis.get('risk_level', 'Unknown')}]", expanded=i < 2):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Website:** {competitor['url']}")
                st.markdown(f"**Description:** {competitor.get('description', 'N/A')}")
                
                if analysis.get('key_findings'):
                    st.markdown("**Key Findings:**")
                    st.info(analysis['key_findings'])
                
                if analysis.get('overlapping_features'):
                    st.markdown("**Overlapping Features:**")
                    for feature in analysis['overlapping_features']:
                        st.write(f"• {feature}")
            
            with col2:
                st.markdown(f"**Confidence:** {analysis.get('infringement_confidence', 'N/A')}")
                
                if analysis.get('recommendations'):
                    st.markdown("**Recommendations:**")
                    for rec in analysis['recommendations'][:2]:
                        st.write(f"📌 {rec}")
            
            # Visit competitor button
            st.markdown(f"[🌐 Visit Competitor Website]({competitor['url']})")
    
    # Download report
    infringement_report = {
        "patent_claims": patent_claims,
        "analysis_summary": {
            "scanned_competitors": infringement_results['scanned_competitors'],
            "high_risk_findings": infringement_results['high_risk_findings'],
            "overall_risk_level": infringement_results['overall_risk_level'],
            "detected_industries": infringement_results.get('detected_industries', {})
        },
        "detailed_results": infringement_results['detailed_results'],
        "generated_at": datetime.now().isoformat()
    }
    
    st.download_button(
        label="📥 Download Infringement Analysis Report",
        data=json.dumps(infringement_report, indent=2),
        file_name=f"patent_infringement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def main():
    st.set_page_config(
        page_title="Patent Infringement Scanner", 
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Patent Infringement Scanner")
    st.markdown("""
    Upload your patent documents to scan for potential infringement risks across the industry.
    This tool uses AI to detect relevant industries and analyze competitor websites for potential patent violations.
    """)
    
    # File upload section
    st.markdown("## 📄 Upload Patent Documents")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Specification")
        specification_file = st.file_uploader(
            "Upload Patent Specification",
            type=["txt", "pdf"],
            key="specification"
        )
        if specification_file:
            st.success(f"✅ {specification_file.name} uploaded")
    
    with col2:
        st.markdown("### Claims")
        claims_file = st.file_uploader(
            "Upload Patent Claims", 
            type=["txt", "pdf"],
            key="claims"
        )
        if claims_file:
            st.success(f"✅ {claims_file.name} uploaded")
    
    with col3:
        st.markdown("### Drawings/Description")
        drawings_file = st.file_uploader(
            "Upload Patent Drawings Description",
            type=["txt", "pdf"], 
            key="drawings"
        )
        if drawings_file:
            st.success(f"✅ {drawings_file.name} uploaded")
    
    # Analysis options
    st.markdown("## ⚙️ Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_competitors = st.slider(
            "Maximum Competitors to Analyze",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of competitor websites to scan for infringement risks"
        )
    
    with col2:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Standard", "Comprehensive", "Quick Scan"],
            help="Depth of infringement analysis"
        )
    
    # Start analysis button
    analyze_button = st.button(
        "🔬 Start Infringement Analysis",
        type="primary",
        disabled=not (specification_file and claims_file)
    )
    
    if analyze_button:
        if not (specification_file and claims_file):
            st.error("Please upload both Specification and Claims documents to proceed.")
            return
        
        # Extract text from files
        with st.spinner("📖 Reading patent documents..."):
            specification_text = extract_text_from_file(specification_file)
            claims_text = extract_text_from_file(claims_file)
            drawings_text = extract_text_from_file(drawings_file) if drawings_file else "No drawings description provided"
        
        # Combine documents for comprehensive analysis
        patent_text = combine_patent_documents(specification_text, claims_text, drawings_text)
        
        # Show document preview
        with st.expander("📄 Patent Documents Preview", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Specification", "Claims", "Drawings"])
            
            with tab1:
                st.text_area("Specification Content", specification_text[:2000], height=200)
            with tab2:
                st.text_area("Claims Content", claims_text[:1500], height=200)
            with tab3:
                st.text_area("Drawings Description", drawings_text[:1000], height=200)
        
        # Perform infringement analysis
        infringement_results = industry_wide_infringement_scan(
            patent_text, 
            claims_text,
            max_competitors
        )
        
        # Display results
        display_infringement_results(infringement_results, claims_text)
    
    # Information section
    st.markdown("---")
    st.markdown("""
    ## 💡 How It Works
    
    1. **Document Analysis**: AI analyzes your patent documents to understand the technology and claims
    2. **Industry Detection**: Identifies relevant industries using keyword, semantic, and LLM analysis
    3. **Competitor Discovery**: Uses AI-generated search queries to find relevant companies
    4. **Infringement Assessment**: Analyzes competitor websites for potential patent violations
    5. **Risk Reporting**: Provides detailed risk assessment and recommendations
    
    ## 🛡️ Features
    
    - **Multi-tier Industry Detection**: Combines keyword, semantic, and AI analysis
    - **Smart Competitor Discovery**: AI-generated search queries for relevant companies
    - **Comprehensive Risk Analysis**: Detailed infringement risk assessment
    - **Actionable Insights**: Specific recommendations for each finding
    - **Exportable Reports**: Download comprehensive analysis reports
    """)

if __name__ == '__main__':
    main()