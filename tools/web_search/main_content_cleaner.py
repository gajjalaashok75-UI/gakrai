#!/usr/bin/env python3
"""
Enhanced main-content cleaning and structuring tool with filtering and advanced cleaning.
Reads search results, filters by extraction_status == "success",
cleans content, and writes a structured JSON.

Usage (CLI):
  python main_content_cleaner.py enterprise_search_20260207_152026.json --out struct_format_results.json

Usage (Programmatic):
  from main_content_cleaner import process_results
  structured, stats = process_results(results_list)
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import html


def advanced_clean_text(text: str, url: Optional[str] = None) -> str:
    """
    Advanced cleaning focused on NOISE removal, not content alteration.
    Preserves all function words, natural syntax, and content structure.
    
    Args:
        text: Raw text to clean
        url: Optional URL for domain-specific cleaning rules
    
    Returns:
        Cleaned text with noise removed but all content preserved
    """
    if not text:
        return ''
    
    # 1. DECODE & NORMALIZE (content-preserving)
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    
    # 2. REMOVE VISUAL/LAYOUT ARTIFACTS (not content)
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        
        # Skip visual separators (lines of symbols)
        if re.match(r'^[=*\-_~]{3,}$', line):
            continue
            
        # Skip page numbers/headers
        if re.match(r'^\s*\d+\s*$|^Page\s+\d+', line):
            continue
            
        # Skip breadcrumbs (navigation paths)
        if re.match(r'^[A-Za-z]+\s*>\s*[A-Za-z]+(?:\s*>\s*[A-Za-z]+)*$', line):
            continue
            
        lines.append(line)
    
    text = '\n'.join(lines)
    
    # 3. REMOVE REPETITIVE NAVIGATION (not content)
    nav_phrases = [
        r'home\s*\|.*',
        r'about\s*us.*',
        r'contact\s*us.*',
        r'sitemap.*',
        r'privacy\s*policy.*',
        r'terms\s*of\s*use.*',
        r'cookie\s*policy.*',
        r'accessibility\s*statement.*',
        r'legal\s*notice.*',
        r'copyright\s*©.*',
        r'all\s*rights\s*reserved.*',
    ]
    
    for pattern in nav_phrases:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 4. REMOVE SOCIAL/ADVERTISEMENT NOISE
    social_patterns = [
        r'follow\s*us\s*on.*',
        r'like\s*us\s*on.*',
        r'share\s*this\s*page.*',
        r'tweet\s*this.*',
        r'sponsored\s*content.*',
        r'advertisement.*',
        r'promoted\s*by.*',
        r'partner\s*content.*',
    ]
    
    for pattern in social_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 5. SMART DEDUPLICATION (preserve meaningful repetition)
    # Only deduplicate very short lines that look like headers
    lines = text.split('\n')
    if len(lines) > 10:
        short_lines = [line for line in lines if len(line) < 80]
        line_counts = Counter(short_lines)
        
        # Remove only if appears > 3 times AND is very short
        repeated_headers = {
            line for line, count in line_counts.items() 
            if count > 3 and len(line.split()) < 5
        }
        
        if repeated_headers:
            seen = set()
            deduped = []
            for line in lines:
                if line in repeated_headers and line in seen:
                    continue
                seen.add(line)
                deduped.append(line)
            text = '\n'.join(deduped)
    
    # 6. FINAL CLEANUP (preserve whitespace for readability)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Keep paragraph breaks
    text = re.sub(r'[ \t]{2,}', ' ', text)  # Normalize spaces
    text = text.strip()
    
    return text


def clean_text(text: str) -> str:
    """Legacy function for backward compatibility"""
    return advanced_clean_text(text)


def extract_content_quality_signals(text: str) -> Dict[str, any]:
    """
    Extract signals about content quality WITHOUT altering content.
    Used for filtering and sorting, not for content modification.
    
    Args:
        text: Cleaned text to analyze
        
    Returns:
        Dictionary with quality signals
    """
    signals = {
        'has_natural_language': False,
        'has_technical_terms': False,
        'has_code_examples': False,
        'has_references': False,
        'has_structure': False,
        'content_density': 0.0,
    }
    
    if not text:
        return signals
    
    # 1. Check for natural language patterns
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    if len(words) > 100:
        signals['has_natural_language'] = True
    
    # 2. Check for technical content (programming terms, etc.)
    technical_terms = {
        'function', 'class', 'method', 'variable', 'import', 
        'def', 'return', 'if', 'else', 'for', 'while', 'try',
        'except', 'database', 'api', 'server', 'client', 'json',
        'xml', 'html', 'css', 'javascript', 'python', 'java',
    }
    found_tech = sum(1 for word in words if word in technical_terms)
    signals['has_technical_terms'] = found_tech > 5
    
    # 3. Check for code examples
    code_patterns = [
        r'def\s+\w+\(.*\):',
        r'class\s+\w+:',
        r'import\s+\w+',
        r'print\(.*\)',
        r'console\.log\(.*\)',
        r'System\.out\.println\(.*\)',
        r'<[a-zA-Z][^>]*>',
    ]
    for pattern in code_patterns:
        if re.search(pattern, text):
            signals['has_code_examples'] = True
            break
    
    # 4. Check for references/citations
    if re.search(r'\[\d+\]|\(\d{4}\)|et al\.|pp\.', text):
        signals['has_references'] = True
    
    # 5. Check for structure (headings, lists)
    heading_count = len(re.findall(r'^\s*(#+|\*+|\-+|\d+\.)\s+', text, re.MULTILINE))
    signals['has_structure'] = heading_count > 2
    
    # 6. Calculate content density (non-noise ratio)
    total_chars = len(text)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if total_chars > 0:
        signals['content_density'] = alpha_chars / total_chars
    
    return signals


def extract_content_sections(text: str) -> Dict[str, List[str]]:
    """
    Extract structured sections from content based on headings.
    
    Args:
        text: Cleaned text content
        
    Returns:
        Dictionary with section titles as keys and paragraph lists as values
    """
    sections = {}
    current_section = "Main Content"
    sections[current_section] = []
    
    lines = text.split('\n')
    
    # Common heading patterns
    heading_patterns = [
        r'^#{1,6}\s+(.*)$',  # Markdown headings
        r'^([A-Z][A-Za-z\s]{3,50}):?$',  # Title case lines
        r'^(Introduction|Overview|Background|Methods|Results|Discussion|Conclusion|References):?$',
        r'^[A-Z\s]{5,30}$',  # ALL CAPS headings
    ]
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Check if line is a heading
        is_heading = False
        for pattern in heading_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                # Additional check: heading shouldn't be too long
                if len(line_stripped.split()) < 10:
                    is_heading = True
                    section_title = match.group(1) if match.groups() else line_stripped
                    section_title = section_title.strip(':').strip()
                    
                    # Clean section title
                    section_title = re.sub(r'[#*_`]', '', section_title)
                    
                    if section_title and section_title != current_section:
                        current_section = section_title
                        sections[current_section] = []
                    break
        
        if not is_heading and len(line_stripped) > 10:  # Skip very short lines
            sections[current_section].append(line_stripped)
    
    # Filter out empty sections
    sections = {k: v for k, v in sections.items() if v}
    
    return sections


def paragraphs(text: str) -> List[str]:
    """Extract meaningful paragraphs from text"""
    if not text:
        return []
    
    # Split on double newlines
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # Filter paragraphs by quality
    quality_paras = []
    for para in raw_paras:
        # Skip very short paragraphs
        if len(para) < 40:
            continue
            
        # Skip paragraphs that are mostly numbers/symbols
        alpha_ratio = sum(1 for c in para if c.isalpha()) / len(para) if para else 0
        if alpha_ratio < 0.4:
            continue
            
        quality_paras.append(para)
    
    return quality_paras


def sentences(text: str) -> List[str]:
    """Extract sentences from text with improved detection"""
    if not text:
        return []
    
    # Better sentence splitting that handles abbreviations
    # Split on sentence endings, but be careful with abbreviations
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    
    # Replace common abbreviations to avoid false splits
    replacements = [
        (r'Dr\.', 'Dr#'),
        (r'Mr\.', 'Mr#'),
        (r'Mrs\.', 'Mrs#'),
        (r'Ms\.', 'Ms#'),
        (r'Prof\.', 'Prof#'),
        (r'etc\.', 'etc#'),
        (r'e\.g\.', 'e#g#'),
        (r'i\.e\.', 'i#e#'),
        (r'vs\.', 'vs#'),
        (r'U\.S\.', 'U#S#'),
        (r'U\.K\.', 'U#K#'),
        (r'Jan\.', 'Jan#'),
        (r'Feb\.', 'Feb#'),
        (r'Aug\.', 'Aug#'),
        (r'Sept\.', 'Sept#'),
        (r'Oct\.', 'Oct#'),
        (r'Nov\.', 'Nov#'),
        (r'Dec\.', 'Dec#'),
    ]
    
    text_clean = text
    for pattern, replacement in replacements:
        text_clean = re.sub(pattern, replacement, text_clean)
    
    # Split sentences
    raw_sentences = re.split(sentence_endings, text_clean)
    
    # Restore abbreviations
    restored_sentences = []
    for sent in raw_sentences:
        sent_restored = sent
        for pattern, replacement in replacements:
            sent_restored = sent_restored.replace(replacement, pattern.replace(r'\.', '.'))
        restored_sentences.append(sent_restored.strip())
    
    # Filter out empty sentences and very short ones
    return [s for s in restored_sentences if len(s) > 15 and not s.isspace()]


def top_keywords(text: str, n: int = 15, min_word_length: int = 4) -> List[str]:
    """
    Extract top keywords with improved filtering.
    
    Args:
        text: Text to analyze
        n: Number of keywords to return
        min_word_length: Minimum word length to consider
        
    Returns:
        List of top keywords
    """
    if not text:
        return []
    
    # Convert to lowercase
    text_lower = text.lower()
    
    # Extract words (allow hyphens and apostrophes in words)
    words = re.findall(r"\b[a-zA-Z'][a-zA-Z'-]{%d,}\b" % (min_word_length-1), text_lower)
    
    # Filter by word length only (no stopwords removal)
    filtered_words = [w for w in words if len(w) >= min_word_length]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    # Return top N keywords
    return [word for word, count in word_counts.most_common(n)]


def calculate_readability_metrics(text: str) -> Dict[str, float]:
    """
    Calculate basic readability metrics for the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with readability metrics
    """
    if not text:
        return {}
    
    # Count sentences
    sents = sentences(text)
    num_sentences = len(sents)
    
    # Count words
    words = re.findall(r'\b\w+\b', text)
    num_words = len(words)
    
    # Count syllables (approximate)
    vowels = 'aeiouy'
    num_syllables = 0
    for word in words:
        word_lower = word.lower()
        if len(word_lower) <= 3:
            num_syllables += 1
            continue
            
        # Count vowel groups
        num_syllables += len(re.findall(r'[aeiouy]+', word_lower))
    
    # Calculate metrics
    if num_sentences > 0 and num_words > 0:
        avg_sentence_length = num_words / num_sentences
        avg_word_length = sum(len(w) for w in words) / num_words
        
        # Flesch Reading Ease (approximate)
        flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        
        return {
            'word_count': num_words,
            'sentence_count': num_sentences,
            'avg_sentence_length': round(avg_sentence_length, 1),
            'avg_word_length': round(avg_word_length, 1),
            'flesch_reading_ease': round(flesch_score, 1) if flesch_score > 0 else 0,
            'syllable_count': num_syllables,
        }
    
    return {}


def process_record(rec: dict) -> dict:
    """
    Process a single record with advanced cleaning.
    Preserves all content while removing only noise.
    
    Args:
        rec: Input record dictionary
        
    Returns:
        Processed record dictionary with natural language preserved
    """
    raw = rec.get('main_content') or ''
    url = rec.get('url') or rec.get('final_url')
    
    # Clean ONLY noise, preserve all content words
    cleaned = advanced_clean_text(raw, url)
    
    # Extract quality signals (for filtering/sorting, not content modification)
    quality_signals = extract_content_quality_signals(cleaned)
    
    # Extract paragraphs with natural language preserved
    paras = paragraphs(cleaned)
    first_para = paras[0] if paras else ''
    
    # Extract sentences with natural syntax
    sens = sentences(cleaned)
    
    # Extract keywords (stopwords removed only here)
    kw = top_keywords(cleaned, n=15)
    
    # Extract content sections
    sections = extract_content_sections(cleaned)
    
    # Calculate readability metrics
    readability = calculate_readability_metrics(cleaned)
    
    # Get content type based on URL and content
    content_type = "unknown"
    if url:
        url_lower = url.lower()
        if any(domain in url_lower for domain in ['wikipedia.org', 'wiki.']):
            content_type = "encyclopedia"
        elif any(domain in url_lower for domain in ['docs.python.org', 'documentation']):
            content_type = "official_docs"
        elif any(domain in url_lower for domain in ['tutorial', 'course', 'learn']):
            content_type = "tutorial"
        elif any(domain in url_lower for domain in ['blog', 'medium.com', 'dev.to']):
            content_type = "blog"
        elif any(domain in url_lower for domain in ['stackoverflow.com', 'stackexchange.com']):
            content_type = "qna"
        elif any(domain in url_lower for domain in ['github.com', 'gitlab.com']):
            content_type = "code_repository"
        elif any(domain in url_lower for domain in ['research', 'arxiv.org', 'academic']):
            content_type = "research_paper"
        elif any(domain in url_lower for domain in ['news', 'article', 'press']):
            content_type = "news_article"
    
    # Create structured result - full natural language preserved
    result = {
        'position': rec.get('position'),
        'title': rec.get('title'),
        'url': rec.get('url'),
        'final_url': rec.get('final_url'),
        'publish_date': rec.get('publish_date'),
        'author': rec.get('author'),
        'fetch_time': rec.get('fetch_time'),
        'extraction_status': rec.get('extraction_status'),
        'confidence_score': rec.get('confidence_score'),
        'content_word_count': rec.get('content_word_count'),
        'content_type': content_type,
        'cleaned_content': cleaned,  # ✅ Full natural language preserved
        'first_paragraph': first_para,
        'paragraphs': paras,  # ✅ Natural paragraphs with all words
        'content_sections': sections,
        'sentences_count': len(sens),
        'sample_sentences': sens[:5],  # ✅ Natural sentences with proper syntax
        'top_keywords': kw,  # ✅ Keywords extracted separately (stopwords removed only here)
        'readability_metrics': readability,
        'quality_signals': quality_signals,  # ✅ Quality metrics without content alteration
    }
    
    # Add content quality score
    quality_score = 0.0
    if cleaned:
        # Base score on extraction confidence
        quality_score += rec.get('confidence_score', 0.0) * 0.3
        
        # Score based on content length
        word_count = len(cleaned.split())
        if word_count > 1000:
            quality_score += 0.4
        elif word_count > 500:
            quality_score += 0.3
        elif word_count > 200:
            quality_score += 0.2
        elif word_count > 50:
            quality_score += 0.1
        
        # Score based on structure
        if len(sections) > 1:
            quality_score += 0.2
        if len(paras) > 5:
            quality_score += 0.1
        
        # Bonus for natural language signals
        if quality_signals.get('has_natural_language'):
            quality_score += 0.1
        
        # Cap at 1.0
        quality_score = min(quality_score, 1.0)
    
    result['content_quality_score'] = round(quality_score, 2)
    
    return result


def process_results(results: list) -> tuple:
    """
    Process search results: filter by success, clean content.
    
    Args:
        results: List of result dicts (from quick_scrape.py)
    
    Returns:
        (structured_results, stats) where stats has filtering info
    """
    # Filter: only "success" extraction status
    successful_results = [r for r in results if r.get('extraction_status') == 'success']
    
    structured_results = []
    
    for rec in successful_results:
        # Process and clean
        structured_rec = process_record(rec)
        structured_results.append(structured_rec)
    
    # Sort by quality score (descending) then position (ascending)
    structured_results.sort(key=lambda x: (-x.get('content_quality_score', 0), x.get('position', 0)))
    
    stats = {
        'total_input': len(results),
        'successful': len(successful_results),
        'failed': len(results) - len(successful_results),
        'processed': len(structured_results)
    }
    print(f"📈 total chars in the structured_results (from main_content extract {len(structured_results)}")
    return structured_results, stats


def main():
    parser = argparse.ArgumentParser(
        description='Advanced clean and structure main_content from enterprise JSON'
    )
    parser.add_argument('input', help='Input enterprise JSON file')
    parser.add_argument('--out', '-o', default='struct_format_results.json', 
                       help='Output structured JSON path')
    parser.add_argument('--min-quality', type=float, default=0.3,
                       help='Minimum content quality score to include (0.0-1.0)')
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f'Input not found: {in_path}')
        return

    data = json.loads(in_path.read_text(encoding='utf-8'))
    results = data.get('results', [])

    # Process results: filter, clean
    structured_results, stats = process_results(results)
    
    # Filter by quality score if specified
    if args.min_quality > 0:
        original_count = len(structured_results)
        structured_results = [
            r for r in structured_results 
            if r.get('content_quality_score', 0) >= args.min_quality
        ]
        stats['quality_filtered'] = original_count - len(structured_results)

    # Build output structure
    structured = {
        'metadata': data.get('metadata', {}),
        'processing_stats': stats,
        'processing_notes': {
            'cleaning_method': 'advanced_clean_text',
            'min_quality_score': args.min_quality,
            'stopwords_filtering': False,
            'version': '2.0'
        },
        'structured_results': structured_results
    }

    # Save structured JSON
    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False), 
        encoding='utf-8'
    )
    
    print(f'\n✅ Advanced processing complete!')
    print(f'   📄 Structured JSON: {out_path}')
    print(f'   📊 Total input: {stats["total_input"]}')
    print(f'   ✅ Successful: {stats["successful"]}')
    print(f'   ❌ Failed (ignored): {stats["failed"]}')
    if 'quality_filtered' in stats:
        print(f'   ⚠️  Quality filtered: {stats["quality_filtered"]}')
    
    # Show top results summary
    if structured_results:
        print(f'\n🏆 Top {min(3, len(structured_results))} results by quality:')
        for i, result in enumerate(structured_results[:3], 1):
            print(f'   {i}. {result.get("title", "No title")[:60]}...')
            print(f'      URL: {result.get("url", "No URL")[:80]}...')
            print(f'      Quality: {result.get("content_quality_score", 0):.2f}')
            print(f'      Words: {len(result.get("cleaned_content", "").split())}')
            print(f'      Type: {result.get("content_type", "unknown")}')
            print()


if __name__ == '__main__':
    main()