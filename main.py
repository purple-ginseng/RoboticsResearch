#!/usr/bin/env python3
"""
Multi-Source Research Paper Crawler & HTML Report Generator

Reads configuration from markdown knowledge base files and crawls
multiple academic sources: arXiv, Semantic Scholar, Google Scholar, etc.
"""

import re
import json
import html
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Optional imports (graceful degradation if not installed)
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False


# ============ CONFIGURATION ============
KNOWLEDGE_BASE_DIR = Path(__file__).parent  # Directory with .md files
OUTPUT_FILE = "research_report.html"
DEFAULT_LIMIT = 150
REQUEST_DELAY = 1.0  # Seconds between API requests (be nice to servers)

# Scopus/Elsevier API Configuration
SCOPUS_API_KEY = "0abb37e13520d982834dcaa108059fc2"
SCOPUS_API_URL = "https://api.elsevier.com/content/search/scopus"
# =======================================


@dataclass
class Paper:
    """Unified paper representation across all sources."""
    title: str
    authors: list[str]
    year: int
    abstract: str = ""
    source: str = ""
    url: str = ""
    pdf_url: str = ""
    doi: str = ""
    venue: str = ""
    citations: int = 0
    categories: list[str] = field(default_factory=list)
    # Extended fields for detailed analysis
    keywords: list[str] = field(default_factory=list)
    core_contribution: str = ""
    problems_addressed: str = ""
    methodology: str = ""
    core_technology: str = ""
    # Enhanced fields from summary.md
    summary_cn: str = ""           # 一句话总结
    contribution_cn: str = ""      # 核心贡献
    method_keywords: list[str] = field(default_factory=list)  # 方法关键词
    pros: str = ""                 # 最大优点
    cons: str = ""                 # 最大不足
    read_rating: str = ""          # 是否值得精读 (必读/可选/可跳过)
    research_themes: list[str] = field(default_factory=list)  # L/I/M/E/R/S
    research_gaps: list[str] = field(default_factory=list)    # Gap identifiers


@dataclass
class KnowledgeBase:
    """Parsed configuration from a markdown knowledge base file."""
    name: str
    source_type: str  # arxiv, semantic, googlescholar, scopus
    query: str
    year_min: int = 2020
    year_max: int = 2026
    limit: int = 150
    fields: list[str] = field(default_factory=list)
    sort_by: str = "relevance"
    llm_instructions: str = ""
    raw_content: str = ""


class KnowledgeBaseParser:
    """Parses markdown knowledge base files into configuration objects."""

    @staticmethod
    def detect_source_type(filename: str, content: str) -> str:
        """Detect the source type from filename or content."""
        filename_lower = filename.lower()
        content_lower = content.lower()

        if "semantic" in filename_lower or "semantic scholar" in content_lower:
            return "semantic"
        elif "google" in filename_lower or "scholar.google" in content_lower:
            return "googlescholar"
        elif "scopus" in filename_lower or "wos" in filename_lower or "web of science" in content_lower:
            return "scopus"
        elif "arxiv" in filename_lower or "arxiv" in content_lower:
            return "arxiv"
        elif "pubmed" in filename_lower or "pubmed" in content_lower:
            return "pubmed"
        else:
            return "generic"

    @staticmethod
    def extract_query(content: str) -> str:
        """Extract the search query from content."""
        patterns = [
            r'query:\s*(.+?)(?:\n|$)',
            r'[Ss]earch query:\s*\n?"?([^"\n]+(?:\n[^"\n]+)?)"?',
            r'TITLE-ABS-KEY\(([\s\S]+?)\)(?:\s*AND\s*TITLE-ABS-KEY\(([\s\S]+?)\))?',
            r'"([^"]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                if match.lastindex and match.lastindex > 1:
                    # Multiple capture groups (Scopus format)
                    parts = [g.strip() for g in match.groups() if g]
                    return " AND ".join(parts)
                return match.group(1).strip().replace('\n', ' ')

        # Fallback: extract any quoted text or first substantial line
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('-')]
        if lines:
            return lines[0][:200]

        return ""

    @staticmethod
    def extract_year_range(content: str) -> tuple[int, int]:
        """Extract year range from content."""
        year_max = datetime.now().year + 1

        # Pattern: year >= 2023 or PUBYEAR > 2022
        match = re.search(r'(?:year|PUBYEAR)\s*(?:>=|>)\s*(\d{4})', content, re.IGNORECASE)
        if match:
            year_min = int(match.group(1))
            if '>' in content and '>=' not in content:
                year_min += 1
            return year_min, year_max

        # Pattern: 2023-2026 or 2023–2026
        match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', content)
        if match:
            return int(match.group(1)), int(match.group(2))

        return 2020, year_max

    @staticmethod
    def extract_limit(content: str) -> int:
        """Extract result limit from content."""
        patterns = [
            r'limit:\s*(\d+)',
            r'~?(\d+)\s*results',
            r'up to\s*(\d+)',
            r'[Rr]etrieve\s*~?(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return DEFAULT_LIMIT

    @staticmethod
    def extract_llm_instructions(content: str) -> str:
        """Extract LLM processing instructions."""
        # Find instruction sections
        patterns = [
            r'instructions_to_LLM:\s*([\s\S]+?)(?:output_format|sort_by|$)',
            r'For each (?:paper|result)[^:]*:\s*([\s\S]+?)(?:\n\n|Return|$)',
            r'then:\s*pass to LLM[^:]*:\s*([\s\S]+?)$',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @classmethod
    def parse_file(cls, filepath: Path) -> KnowledgeBase:
        """Parse a markdown file into a KnowledgeBase object."""
        content = filepath.read_text(encoding='utf-8')
        name = filepath.stem

        source_type = cls.detect_source_type(name, content)
        query = cls.extract_query(content)
        year_min, year_max = cls.extract_year_range(content)
        limit = cls.extract_limit(content)
        llm_instructions = cls.extract_llm_instructions(content)

        return KnowledgeBase(
            name=name,
            source_type=source_type,
            query=query,
            year_min=year_min,
            year_max=year_max,
            limit=limit,
            llm_instructions=llm_instructions,
            raw_content=content
        )


class SummaryParser:
    """Parses summary.md to extract paper metadata and research themes."""

    # Research theme mappings from gaps.md
    RESEARCH_THEMES = {
        "L": {"name": "Locomotion", "keywords": ["locomotion", "walking", "bipedal", "gait", "balance", "mpc", "whole-body"]},
        "I": {"name": "Imitation", "keywords": ["imitation", "amp", "motion", "tracking", "mocap", "demonstration", "shadowing"]},
        "M": {"name": "Manipulation", "keywords": ["manipulation", "grasping", "dexterous", "object", "loco-manipulation"]},
        "E": {"name": "Embodied AI", "keywords": ["llm", "vlm", "language", "foundation", "reasoning", "vision-language", "chain-of"]},
        "R": {"name": "Multi-Robot", "keywords": ["multi-robot", "marl", "coordination", "swarm", "multi-agent", "heterogeneous"]},
        "S": {"name": "Safety/Deploy", "keywords": ["safety", "security", "deployment", "real-world", "cybersecurity", "fault"]}
    }

    # Research gaps from gaps.md
    RESEARCH_GAPS = {
        "Gap-L1": "Unified framework for non-periodic, contact-rich transitions",
        "Gap-I1": "Demonstration generalization and morphology mismatch",
        "Gap-M1": "Cross-task generalization and reusable motion representations",
        "Gap-E1": "Bridge high-level reasoning and low-level physical feasibility",
        "Gap-R1": "Multi-humanoid coordination with whole-body dynamics",
        "Gap-S1": "Safety-aware, scalable real-world learning frameworks"
    }

    @classmethod
    def parse_summary_file(cls, filepath: Path) -> dict:
        """Parse summary.md and return a dict mapping paper titles to metadata."""
        if not filepath.exists():
            print(f"   ⚠️  Summary file not found: {filepath}")
            return {}

        content = filepath.read_text(encoding='utf-8')
        papers_meta = {}

        # Split by paper entries (## 📄 or numbered entries)
        paper_blocks = re.split(r'\n---\n|\n##\s*📄', content)

        for block in paper_blocks:
            if not block.strip():
                continue

            # Extract title - look for patterns like "1. *Title*" or just "*Title*"
            title_match = re.search(r'\*([^*]+)\*', block)
            if not title_match:
                # Try to find title from citation line
                cite_match = re.search(r'"([^"]+),"', block)
                if cite_match:
                    title = cite_match.group(1).strip()
                else:
                    continue
            else:
                title = title_match.group(1).strip()

            # Normalize title for matching
            title_normalized = cls._normalize_title(title)

            meta = {
                "title": title,
                "summary_cn": "",
                "contribution_cn": "",
                "method_keywords": [],
                "pros": "",
                "cons": "",
                "read_rating": ""
            }

            # Extract summary (一句话总结)
            summary_match = re.search(r'一句话总结[：:]\s*(.+?)(?:\n|$)', block)
            if summary_match:
                meta["summary_cn"] = summary_match.group(1).strip()

            # Extract contribution (核心贡献)
            contrib_match = re.search(r'核心贡献[：:]\s*(.+?)(?:\n|$)', block)
            if contrib_match:
                meta["contribution_cn"] = contrib_match.group(1).strip()

            # Extract method keywords (方法关键词)
            keywords_match = re.search(r'方法关键词[：:]\s*(.+?)(?:\n|$)', block)
            if keywords_match:
                keywords_str = keywords_match.group(1)
                # Remove emojis and split
                keywords_str = re.sub(r'[^\w\s,，/\-]', '', keywords_str)
                keywords = [k.strip() for k in re.split(r'[,，]', keywords_str) if k.strip()]
                meta["method_keywords"] = keywords

            # Extract pros (最大优点)
            pros_match = re.search(r'最大优点[：:]\s*(.+?)(?:\n|$)', block)
            if pros_match:
                meta["pros"] = pros_match.group(1).strip()

            # Extract cons (最大不足)
            cons_match = re.search(r'最大不足[：:]\s*(.+?)(?:\n|$)', block)
            if cons_match:
                meta["cons"] = cons_match.group(1).strip()

            # Extract read rating (是否值得精读 or 值不值得精读)
            rating_match = re.search(r'(?:是否|值不值得)?值?得?精读[：:\*]*\s*(.+?)(?:\n|$)', block)
            if rating_match:
                rating_text = rating_match.group(1).strip()
                # Handle star ratings and keywords
                if "必读" in rating_text or "⭐" in rating_text:
                    meta["read_rating"] = "must_read"
                elif "可选" in rating_text or "⚠️" in rating_text:
                    meta["read_rating"] = "optional"
                elif "可跳过" in rating_text or "❌" in rating_text or "否" in rating_text:
                    meta["read_rating"] = "skip"
                elif "**是**" in rating_text or "是**" in rating_text or rating_text.startswith("**是"):
                    meta["read_rating"] = "must_read"
                elif "**否**" in rating_text or "否**" in rating_text:
                    meta["read_rating"] = "skip"
                else:
                    meta["read_rating"] = "optional"

            papers_meta[title_normalized] = meta

        print(f"   📚 Parsed {len(papers_meta)} paper summaries from summary.md")
        return papers_meta

    @classmethod
    def _normalize_title(cls, title: str) -> str:
        """Normalize title for fuzzy matching."""
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    @classmethod
    def match_paper_to_summary(cls, paper: Paper, summaries: dict) -> dict | None:
        """Find matching summary for a paper."""
        paper_title_norm = cls._normalize_title(paper.title)

        # Exact match
        if paper_title_norm in summaries:
            return summaries[paper_title_norm]

        # Fuzzy match - check if summary title is contained in paper title or vice versa
        for summary_title, meta in summaries.items():
            if summary_title in paper_title_norm or paper_title_norm in summary_title:
                return meta
            # Check significant word overlap
            paper_words = set(paper_title_norm.split())
            summary_words = set(summary_title.split())
            overlap = len(paper_words & summary_words)
            if overlap >= 3 and overlap / min(len(paper_words), len(summary_words)) > 0.5:
                return meta

        return None

    @classmethod
    def detect_research_themes(cls, paper: Paper) -> list[str]:
        """Detect research themes based on paper content."""
        themes = []
        text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()

        for theme_code, theme_info in cls.RESEARCH_THEMES.items():
            for keyword in theme_info["keywords"]:
                if keyword in text:
                    themes.append(theme_code)
                    break

        return list(set(themes)) if themes else ["L"]  # Default to Locomotion

    @classmethod
    def detect_research_gaps(cls, paper: Paper) -> list[str]:
        """Detect relevant research gaps based on paper themes."""
        gaps = []
        themes = paper.research_themes or cls.detect_research_themes(paper)

        theme_to_gap = {"L": "Gap-L1", "I": "Gap-I1", "M": "Gap-M1",
                        "E": "Gap-E1", "R": "Gap-R1", "S": "Gap-S1"}

        for theme in themes:
            if theme in theme_to_gap:
                gaps.append(theme_to_gap[theme])

        return gaps


class BaseCrawler(ABC):
    """Abstract base class for all source crawlers."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.papers: list[Paper] = []

    @abstractmethod
    def crawl(self) -> list[Paper]:
        """Execute the crawl and return papers."""
        pass

    def _normalize_query(self, query: str) -> str:
        """Normalize query for API use."""
        # Remove Scopus-style syntax
        query = re.sub(r'TITLE-ABS-KEY\s*\(', '', query)
        # Balance parentheses by removing all of them
        query = re.sub(r'[()]', '', query)
        # Clean up whitespace
        query = re.sub(r'\s+', ' ', query)
        return query.strip()

    def _extract_keywords(self, query: str) -> str:
        """Extract simple keywords from complex query for arXiv."""
        # Remove boolean operators and quotes for simpler query
        query = re.sub(r'\b(AND|OR)\b', ' ', query, flags=re.IGNORECASE)
        query = re.sub(r'[()"]', '', query)
        query = re.sub(r'\s+', ' ', query)
        # Take first few meaningful terms
        words = [w for w in query.split() if len(w) > 2][:8]
        return ' '.join(words)


class ArxivCrawler(BaseCrawler):
    """Crawler for arXiv using the official API."""

    def crawl(self) -> list[Paper]:
        if not ARXIV_AVAILABLE:
            print(f"   ⚠️  arxiv library not installed. Run: pip install arxiv")
            return []

        # Use simplified keywords for arXiv (it's picky about query syntax)
        query = self._extract_keywords(self.kb.query)
        print(f"   🔍 Searching arXiv for: '{query[:60]}...'")

        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=self.kb.limit + 50,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in client.results(search):
                year = result.published.year
                if self.kb.year_min <= year <= self.kb.year_max:
                    papers.append(Paper(
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        year=year,
                        abstract=result.summary,
                        source="arXiv",
                        url=result.entry_id,
                        pdf_url=result.pdf_url or "",
                        categories=list(result.categories),
                    ))

                if len(papers) >= self.kb.limit:
                    break

            print(f"   ✅ Found {len(papers)} papers from arXiv")
            return papers

        except Exception as e:
            print(f"   ⚠️  arXiv error: {e}")
            return []


class SemanticScholarCrawler(BaseCrawler):
    """Crawler for Semantic Scholar using the public API."""

    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    MAX_RETRIES = 3
    BATCH_SIZE = 50  # Smaller batches to avoid rate limits

    def _make_request(self, url: str, retry: int = 0) -> dict | None:
        """Make API request with retry logic for rate limiting."""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "ResearchCrawler/1.0 (Academic Research Tool)"
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())

        except urllib.error.HTTPError as e:
            if e.code == 429 and retry < self.MAX_RETRIES:
                # Rate limited - exponential backoff
                wait_time = (2 ** retry) * 5  # 5s, 10s, 20s
                print(f"   ⏳ Rate limited. Waiting {wait_time}s before retry {retry + 1}/{self.MAX_RETRIES}...")
                time.sleep(wait_time)
                return self._make_request(url, retry + 1)
            else:
                print(f"   ⚠️  HTTP Error {e.code}: {e.reason}")
                return None

        except Exception as e:
            print(f"   ⚠️  Request error: {e}")
            return None

    def crawl(self) -> list[Paper]:
        query = self._normalize_query(self.kb.query)
        # Simplify query for better API results
        query = re.sub(r'\s+AND\s+', ' ', query)
        query = re.sub(r'[()]', '', query)
        query = query[:200]  # API has query length limits

        print(f"   🔍 Searching Semantic Scholar for: '{query[:60]}...'")

        papers = []
        offset = 0

        while len(papers) < self.kb.limit:
            params = {
                "query": query,
                "offset": offset,
                "limit": self.BATCH_SIZE,
                "fields": "title,authors,year,abstract,venue,citationCount,externalIds,url",
                "year": f"{self.kb.year_min}-{self.kb.year_max}"
            }

            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            data = self._make_request(url)

            if not data or "data" not in data or not data["data"]:
                break

            for item in data["data"]:
                if not item.get("year"):
                    continue

                year = item["year"]
                if self.kb.year_min <= year <= self.kb.year_max:
                    doi = ""
                    if item.get("externalIds"):
                        doi = item["externalIds"].get("DOI", "")

                    papers.append(Paper(
                        title=item.get("title", ""),
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        year=year,
                        abstract=item.get("abstract") or "",
                        source="Semantic Scholar",
                        url=f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                        doi=doi,
                        venue=item.get("venue") or "",
                        citations=item.get("citationCount", 0),
                    ))

                if len(papers) >= self.kb.limit:
                    break

            print(f"   📥 Retrieved {len(papers)}/{self.kb.limit} papers...")
            offset += self.BATCH_SIZE
            time.sleep(3)  # Longer delay between requests

            if offset >= data.get("total", 0):
                break

        print(f"   ✅ Found {len(papers)} papers from Semantic Scholar")
        return papers


class GoogleScholarCrawler(BaseCrawler):
    """Crawler for Google Scholar using scholarly library."""

    def crawl(self) -> list[Paper]:
        if not SCHOLARLY_AVAILABLE:
            print(f"   ⚠️  scholarly library not installed. Run: pip install scholarly")
            return []

        query = self._normalize_query(self.kb.query)
        print(f"   🔍 Searching Google Scholar for: '{query[:60]}...'")

        papers = []
        try:
            search_query = scholarly.search_pubs(query)

            for i, result in enumerate(search_query):
                if i >= self.kb.limit + 50:  # Fetch extra for filtering
                    break

                try:
                    year = int(result.get("bib", {}).get("pub_year", 0))
                except (ValueError, TypeError):
                    year = 0

                if year and self.kb.year_min <= year <= self.kb.year_max:
                    bib = result.get("bib", {})
                    papers.append(Paper(
                        title=bib.get("title", ""),
                        authors=bib.get("author", "").split(" and ") if isinstance(bib.get("author"), str) else bib.get("author", []),
                        year=year,
                        abstract=bib.get("abstract", ""),
                        source="Google Scholar",
                        url=result.get("pub_url", ""),
                        venue=bib.get("venue", ""),
                        citations=result.get("num_citations", 0),
                    ))

                if len(papers) >= self.kb.limit:
                    break

                time.sleep(REQUEST_DELAY * 2)  # Be extra nice to Google

        except Exception as e:
            print(f"   ⚠️  Google Scholar error: {e}")

        print(f"   ✅ Found {len(papers)} papers from Google Scholar")
        return papers


class ScopusCrawler(BaseCrawler):
    """Crawler for Scopus using Elsevier API."""

    def crawl(self) -> list[Paper]:
        if not SCOPUS_API_KEY:
            print(f"   ⚠️  Scopus API key not configured.")
            print(f"   💡 Tip: Set SCOPUS_API_KEY in main.py or export CSV from scopus.com")
            # Fallback to CSV
            csv_files = list(KNOWLEDGE_BASE_DIR.glob("scopus*.csv"))
            if csv_files:
                return self._parse_csv(csv_files[0])
            return []

        print(f"   🔍 Searching Scopus API...")
        papers = []

        try:
            # Build Scopus query
            query = self._build_scopus_query()
            print(f"   📝 Query: {query[:80]}...")

            # Paginate through results
            start = 0
            count = 25  # Scopus returns max 25 per request

            while len(papers) < self.kb.limit:
                params = {
                    "query": query,
                    "start": start,
                    "count": count,
                    "date": f"{self.kb.year_min}-{self.kb.year_max}",
                    "sort": "-citedby-count",  # Sort by citations
                }

                url = f"{SCOPUS_API_URL}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url)
                req.add_header("X-ELS-APIKey", SCOPUS_API_KEY)
                req.add_header("Accept", "application/json")

                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode())

                results = data.get("search-results", {})
                entries = results.get("entry", [])

                if not entries or (len(entries) == 1 and "error" in entries[0]):
                    break

                for entry in entries:
                    if len(papers) >= self.kb.limit:
                        break

                    # Skip error entries
                    if "error" in entry:
                        continue

                    # Extract year from cover date
                    cover_date = entry.get("prism:coverDate", "")
                    try:
                        year = int(cover_date[:4]) if cover_date else 0
                    except ValueError:
                        year = 0

                    # Extract authors
                    authors = []
                    if "author" in entry:
                        authors = [a.get("authname", "") for a in entry.get("author", [])]
                    elif "dc:creator" in entry:
                        authors = [entry.get("dc:creator", "")]

                    # Build paper URL
                    scopus_id = entry.get("dc:identifier", "").replace("SCOPUS_ID:", "")
                    paper_url = entry.get("prism:url", "")
                    if scopus_id:
                        paper_url = f"https://www.scopus.com/record/display.uri?eid={entry.get('eid', '')}&origin=resultslist"

                    # Get DOI link if available
                    doi = entry.get("prism:doi", "")
                    if doi and not paper_url:
                        paper_url = f"https://doi.org/{doi}"

                    papers.append(Paper(
                        title=entry.get("dc:title", ""),
                        authors=authors,
                        year=year,
                        abstract=entry.get("dc:description", "") or entry.get("prism:teaser", ""),
                        source="Scopus",
                        doi=doi,
                        url=paper_url,
                        venue=entry.get("prism:publicationName", ""),
                        citations=int(entry.get("citedby-count", 0) or 0),
                    ))

                # Check if more results available
                total_results = int(results.get("opensearch:totalResults", 0))
                start += count

                if start >= total_results:
                    break

                time.sleep(REQUEST_DELAY)

            print(f"   ✅ Found {len(papers)} papers from Scopus")

        except urllib.error.HTTPError as e:
            print(f"   ❌ Scopus API error: {e.code} - {e.reason}")
            if e.code == 401:
                print(f"   💡 Check your API key is valid")
            elif e.code == 429:
                print(f"   💡 Rate limit exceeded, try again later")
        except Exception as e:
            print(f"   ❌ Scopus error: {e}")

        return papers

    def _build_scopus_query(self) -> str:
        """Build Scopus search query from knowledge base."""
        # Try to read full Scopus query from KB file
        kb_file = KNOWLEDGE_BASE_DIR / f"{self.kb.name}.md"
        if kb_file.exists():
            content = kb_file.read_text(encoding='utf-8')
            # Extract full TITLE-ABS-KEY query blocks
            scopus_match = re.search(
                r'(TITLE-ABS-KEY\s*\([^)]+\)(?:\s*AND\s*TITLE-ABS-KEY\s*\([^)]+\))*)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if scopus_match:
                query = scopus_match.group(1)
                # Clean up whitespace
                query = re.sub(r'\s+', ' ', query).strip()
                return query

        # Fallback to KB query
        query = self.kb.query

        # If already in Scopus format, use as-is
        if "TITLE-ABS-KEY" in query.upper():
            return query

        # Convert simple query to Scopus format
        # Clean up the query
        query = re.sub(r'[()]', '', query)
        terms = [t.strip() for t in re.split(r'\s+AND\s+|\s+OR\s+', query, flags=re.IGNORECASE) if t.strip()]

        if terms:
            # Build TITLE-ABS-KEY query
            scopus_terms = [f'TITLE-ABS-KEY({t})' for t in terms[:5]]  # Limit to 5 terms
            return " AND ".join(scopus_terms)

        return f'TITLE-ABS-KEY({query})'

    def _parse_csv(self, filepath: Path) -> list[Paper]:
        """Parse exported Scopus CSV as fallback."""
        import csv
        papers = []

        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        year = int(row.get("Year", 0))
                    except ValueError:
                        year = 0

                    if year and self.kb.year_min <= year <= self.kb.year_max:
                        papers.append(Paper(
                            title=row.get("Title", ""),
                            authors=row.get("Authors", "").split(", "),
                            year=year,
                            abstract=row.get("Abstract", ""),
                            source="Scopus",
                            doi=row.get("DOI", ""),
                            venue=row.get("Source title", ""),
                            citations=int(row.get("Cited by", 0) or 0),
                        ))

                    if len(papers) >= self.kb.limit:
                        break

            print(f"   ✅ Loaded {len(papers)} papers from Scopus CSV")

        except Exception as e:
            print(f"   ⚠️  Error parsing Scopus CSV: {e}")

        return papers


class GenericCrawler(BaseCrawler):
    """Fallback crawler that uses arXiv as default."""

    def crawl(self) -> list[Paper]:
        print(f"   ℹ️  Using arXiv as fallback for '{self.kb.name}'")
        arxiv_crawler = ArxivCrawler(self.kb)
        papers = arxiv_crawler.crawl()
        for p in papers:
            p.source = f"arXiv (via {self.kb.name})"
        return papers


class MultiSourceCrawler:
    """Orchestrates crawling from multiple sources based on knowledge bases."""

    CRAWLERS = {
        "arxiv": ArxivCrawler,
        "semantic": SemanticScholarCrawler,
        "googlescholar": GoogleScholarCrawler,
        "scopus": ScopusCrawler,
        "generic": GenericCrawler,
    }

    def __init__(self, kb_dir: Path):
        self.kb_dir = kb_dir
        self.knowledge_bases: list[KnowledgeBase] = []
        self.all_papers: list[Paper] = []

    def load_knowledge_bases(self) -> list[KnowledgeBase]:
        """Load all markdown knowledge base files."""
        md_files = list(self.kb_dir.glob("*.md"))

        if not md_files:
            print("⚠️  No markdown knowledge base files found!")
            print(f"   Looking in: {self.kb_dir}")
            return []

        print(f"📚 Found {len(md_files)} knowledge base file(s):\n")

        for md_file in md_files:
            kb = KnowledgeBaseParser.parse_file(md_file)
            self.knowledge_bases.append(kb)
            print(f"   • {kb.name}.md → {kb.source_type.upper()}")
            print(f"     Query: {kb.query[:50]}...")
            print(f"     Years: {kb.year_min}-{kb.year_max}, Limit: {kb.limit}")
            print()

        return self.knowledge_bases

    def crawl_all(self) -> list[Paper]:
        """Crawl all configured sources."""
        print("\n" + "=" * 60)
        print("   Starting Multi-Source Crawl")
        print("=" * 60 + "\n")

        for kb in self.knowledge_bases:
            print(f"📡 Crawling: {kb.name} ({kb.source_type})")

            crawler_class = self.CRAWLERS.get(kb.source_type, GenericCrawler)
            crawler = crawler_class(kb)

            papers = crawler.crawl()
            self.all_papers.extend(papers)

            print()
            time.sleep(REQUEST_DELAY)

        # Deduplicate by title similarity
        self._deduplicate()

        print(f"\n📊 Total unique papers collected: {len(self.all_papers)}")
        return self.all_papers

    def _deduplicate(self):
        """Remove duplicate papers based on title similarity."""
        seen_titles = set()
        unique_papers = []

        for paper in self.all_papers:
            # Normalize title for comparison
            normalized = re.sub(r'[^a-z0-9]', '', paper.title.lower())
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_papers.append(paper)

        removed = len(self.all_papers) - len(unique_papers)
        if removed > 0:
            print(f"   🔄 Removed {removed} duplicate papers")

        self.all_papers = unique_papers


def analyze_paper(paper: Paper) -> Paper:
    """Extract keywords, contributions, and methodology from paper abstract."""
    abstract = paper.abstract.lower()

    # Extract keywords from categories and abstract
    tech_keywords = [
        "deep learning", "reinforcement learning", "neural network", "transformer",
        "cnn", "rnn", "lstm", "gnn", "attention", "bert", "gpt", "diffusion",
        "robot", "humanoid", "bipedal", "locomotion", "manipulation", "navigation",
        "swarm", "multi-agent", "multi-robot", "cooperative", "coordination",
        "slam", "perception", "planning", "control", "optimization", "simulation",
        "ros", "gazebo", "mujoco", "isaac", "pytorch", "tensorflow",
        "lidar", "camera", "imu", "sensor fusion", "computer vision",
        "nlp", "language model", "embodied ai", "imitation learning"
    ]

    found_keywords = []
    for kw in tech_keywords:
        if kw in abstract:
            found_keywords.append(kw.title())

    # Add categories as keywords
    found_keywords.extend([cat.upper() for cat in paper.categories[:3]])
    paper.keywords = list(dict.fromkeys(found_keywords))[:10]  # Dedupe, limit to 10

    # Extract core contribution (look for key phrases)
    contribution_markers = [
        "we propose", "we present", "we introduce", "we develop",
        "this paper presents", "this work introduces", "our contribution",
        "we design", "we demonstrate", "our approach", "our method"
    ]

    sentences = paper.abstract.split('. ')
    contributions = []
    problems = []
    methods = []

    for sent in sentences:
        sent_lower = sent.lower()

        # Find contribution sentences
        for marker in contribution_markers:
            if marker in sent_lower:
                contributions.append(sent.strip())
                break

        # Find problem/challenge sentences
        if any(w in sent_lower for w in ["challenge", "problem", "difficult", "limitation", "issue", "gap"]):
            problems.append(sent.strip())

        # Find methodology sentences
        if any(w in sent_lower for w in ["method", "algorithm", "approach", "technique", "framework", "architecture", "model"]):
            methods.append(sent.strip())

    paper.core_contribution = contributions[0] if contributions else ""
    paper.problems_addressed = ". ".join(problems[:2]) if problems else ""
    paper.methodology = methods[0] if methods else ""

    # Identify core technology
    tech_patterns = {
        "Deep Learning": ["deep learning", "neural network", "cnn", "rnn", "transformer"],
        "Reinforcement Learning": ["reinforcement learning", "rl", "policy", "reward", "q-learning", "ppo", "sac"],
        "Computer Vision": ["vision", "image", "visual", "camera", "detection", "segmentation"],
        "Motion Planning": ["planning", "trajectory", "path", "motion planning", "navigation"],
        "Control Systems": ["control", "pid", "mpc", "feedback", "stability"],
        "Multi-Agent Systems": ["multi-agent", "multi-robot", "swarm", "cooperative", "distributed"],
        "Simulation": ["simulation", "simulator", "gazebo", "mujoco", "isaac"],
        "Natural Language": ["language", "nlp", "text", "instruction", "command"],
        "Imitation Learning": ["imitation", "demonstration", "learning from", "behavior cloning"],
    }

    detected_tech = []
    for tech_name, keywords in tech_patterns.items():
        if any(kw in abstract for kw in keywords):
            detected_tech.append(tech_name)

    paper.core_technology = ", ".join(detected_tech[:3]) if detected_tech else "Not specified"

    return paper


def analyze_all_papers(papers: list[Paper]) -> list[Paper]:
    """Analyze all papers to extract detailed information."""
    print("🔬 Analyzing papers for detailed extraction...")
    analyzed = [analyze_paper(p) for p in papers]
    print(f"   ✅ Analyzed {len(analyzed)} papers")
    return analyzed


def enrich_papers_with_summary(papers: list[Paper], summary_file: Path) -> list[Paper]:
    """Enrich papers with metadata from summary.md."""
    print("📖 Enriching papers with summary.md data...")

    summaries = SummaryParser.parse_summary_file(summary_file)
    if not summaries:
        print("   ⚠️  No summaries loaded, skipping enrichment")
        return papers

    matched = 0
    for paper in papers:
        meta = SummaryParser.match_paper_to_summary(paper, summaries)
        if meta:
            paper.summary_cn = meta.get("summary_cn", "")
            paper.contribution_cn = meta.get("contribution_cn", "")
            paper.method_keywords = meta.get("method_keywords", [])
            paper.pros = meta.get("pros", "")
            paper.cons = meta.get("cons", "")
            paper.read_rating = meta.get("read_rating", "")
            matched += 1

        # Detect research themes
        paper.research_themes = SummaryParser.detect_research_themes(paper)
        paper.research_gaps = SummaryParser.detect_research_gaps(paper)

    print(f"   ✅ Matched {matched}/{len(papers)} papers with summaries")
    print(f"   🏷️  Research themes assigned to all papers")
    return papers


def sanitize_filename(title: str, max_length: int = 80) -> str:
    """Sanitize paper title for use as filename."""
    # Remove/replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '', title)
    filename = re.sub(r'[\s]+', '_', filename)
    filename = re.sub(r'[^\w\-_.]', '', filename)
    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length].rstrip('_')
    return filename or "untitled"


def download_papers(papers: list[Paper], base_dir: str = "papers") -> dict:
    """
    Download all papers and organize by year and category.

    Directory structure: papers/{year}/{category}/paper_title.pdf

    Returns dict with download statistics.
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    # Category name mapping
    CATEGORY_NAMES = {
        "cs.ro": "Robotics",
        "cs.ai": "Artificial_Intelligence",
        "cs.lg": "Machine_Learning",
        "cs.cv": "Computer_Vision",
        "cs.ma": "Multi_Agent_Systems",
        "cs.sy": "Systems_Control",
        "cs.hc": "Human_Computer_Interaction",
        "cs.cl": "Computational_Linguistics",
        "cs.ne": "Neural_Evolutionary",
        "eess.sy": "Signal_Processing",
        "stat.ml": "Statistics_ML",
    }

    def get_category_folder(paper: Paper) -> str:
        if paper.categories:
            cat = paper.categories[0].lower()
            return CATEGORY_NAMES.get(cat, cat.replace(".", "_").upper())
        return "Other"

    stats = {
        "total": len(papers),
        "downloaded": 0,
        "skipped_no_pdf": 0,
        "skipped_exists": 0,
        "failed": 0,
        "errors": []
    }

    print(f"\n📥 Downloading {len(papers)} papers to '{base_dir}/'...")
    print("=" * 60)

    for i, paper in enumerate(papers, 1):
        # Skip papers without PDF URL
        if not paper.pdf_url:
            stats["skipped_no_pdf"] += 1
            continue

        # Create directory structure: papers/year/category/
        year_dir = base_path / str(paper.year)
        category_name = get_category_folder(paper)
        category_dir = year_dir / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        filename = sanitize_filename(paper.title) + ".pdf"
        filepath = category_dir / filename

        # Skip if already downloaded
        if filepath.exists():
            stats["skipped_exists"] += 1
            print(f"   [{i}/{len(papers)}] ⏭️  Already exists: {paper.title[:50]}...")
            continue

        # Download PDF
        try:
            print(f"   [{i}/{len(papers)}] ⬇️  Downloading: {paper.title[:50]}...")

            req = urllib.request.Request(
                paper.pdf_url,
                headers={"User-Agent": "ResearchCrawler/1.0 (Academic Research Tool)"}
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read()

                # Verify it's a PDF (check magic bytes)
                if content[:4] != b'%PDF':
                    raise ValueError("Downloaded file is not a valid PDF")

                filepath.write_bytes(content)
                stats["downloaded"] += 1
                print(f"       ✅ Saved: {filepath.relative_to(base_path)}")

            # Be nice to servers
            time.sleep(0.5)

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"{paper.title[:40]}: {str(e)}"
            stats["errors"].append(error_msg)
            print(f"       ❌ Failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    print(f"   ✅ Downloaded: {stats['downloaded']}")
    print(f"   ⏭️  Already existed: {stats['skipped_exists']}")
    print(f"   ⚠️  No PDF URL: {stats['skipped_no_pdf']}")
    print(f"   ❌ Failed: {stats['failed']}")

    if stats["errors"]:
        print(f"\n⚠️  Errors:")
        for err in stats["errors"][:5]:  # Show first 5 errors
            print(f"   • {err}")
        if len(stats["errors"]) > 5:
            print(f"   ... and {len(stats['errors']) - 5} more")

    print(f"\n📁 Papers saved to: {base_path.absolute()}")

    return stats


def generate_html_report(papers: list[Paper], knowledge_bases: list[KnowledgeBase], output_path: str) -> str:
    """Generate a comprehensive HTML report with PSI Lab-inspired design."""

    def esc(text):
        return html.escape(str(text)) if text else ""

    def format_ieee_author(name: str) -> str:
        """Format author name in IEEE style: F. M. LastName"""
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0][0]}. {parts[1]}"
        else:
            # First and middle initials + last name
            initials = ". ".join(p[0] for p in parts[:-1])
            return f"{initials}. {parts[-1]}"

    def generate_ieee_citation(paper: Paper) -> str:
        """Generate IEEE format citation for a paper."""
        # Format authors
        if len(paper.authors) > 6:
            authors_str = format_ieee_author(paper.authors[0]) + " et al."
        elif len(paper.authors) > 1:
            formatted = [format_ieee_author(a) for a in paper.authors[:-1]]
            authors_str = ", ".join(formatted) + ", and " + format_ieee_author(paper.authors[-1])
        elif paper.authors:
            authors_str = format_ieee_author(paper.authors[0])
        else:
            authors_str = "Unknown Author"

        # Build citation
        citation = f'{authors_str}, "{paper.title},"'

        # Add venue/source
        if paper.venue:
            citation += f" {paper.venue},"
        elif paper.source:
            citation += f" {paper.source},"

        # Add year
        citation += f" {paper.year}."

        # Add DOI (required per user request)
        if paper.doi:
            citation += f" doi: {paper.doi}."

        # Add URL
        if paper.url:
            citation += f" [Online]. Available: {paper.url}"

        return citation

    # Category name mapping for display
    CATEGORY_NAMES = {
        "cs.ro": "Robotics",
        "cs.ai": "Artificial Intelligence",
        "cs.lg": "Machine Learning",
        "cs.cv": "Computer Vision",
        "cs.ma": "Multi-Agent Systems",
        "cs.sy": "Systems & Control",
        "cs.hc": "Human-Computer Interaction",
        "cs.cl": "Computational Linguistics",
        "cs.ne": "Neural & Evolutionary",
        "eess.sy": "Signal Processing",
        "stat.ml": "Statistics ML",
    }

    def get_category_name(cat: str) -> str:
        cat_lower = cat.lower()
        return CATEGORY_NAMES.get(cat_lower, cat.upper())

    def get_primary_category(paper) -> str:
        if paper.categories:
            return paper.categories[0].lower()
        return "other"

    # Group papers by source, year, and category
    by_source = {}
    by_year = {}
    by_category = {}
    for p in papers:
        by_source.setdefault(p.source, []).append(p)
        by_year.setdefault(p.year, []).append(p)
        primary_cat = get_primary_category(p)
        by_category.setdefault(primary_cat, []).append(p)

    # Sort years descending, categories by count
    sorted_years = sorted(by_year.keys(), reverse=True)
    sorted_categories = sorted(by_category.keys(), key=lambda c: len(by_category[c]), reverse=True)

    # Build filter buttons HTML
    filter_buttons = ''.join(
        f'<button class="filter-btn source-btn" onclick="filterBySource(\'{esc(src)}\')">{esc(src)}</button>'
        for src in by_source.keys()
    )
    year_buttons = ''.join(
        f'<button class="filter-btn year-btn" onclick="filterByYear({y})">{y}</button>'
        for y in sorted_years
    )
    category_buttons = ''.join(
        f'<button class="filter-btn cat-btn" onclick="filterByCategory(\'{esc(cat)}\')">{get_category_name(cat)} ({len(by_category[cat])})</button>'
        for cat in sorted_categories[:10]  # Top 10 categories
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report - Multi-Source Literature Review</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        :root {{
            --accent: #00d4ff;
            --accent-secondary: #667eea;
            --bg-dark: #0a0a1a;
            --bg-card: #12122a;
            --bg-hover: #1a1a3a;
            --text-primary: #e8e8f0;
            --text-secondary: #888899;
            --border: #2a2a4a;
            --success: #10b981;
            --warning: #f59e0b;
            --highlight: #ec4899;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 30px; }}

        /* Header */
        header {{
            text-align: center;
            padding: 60px 40px;
            margin-bottom: 40px;
            background: linear-gradient(180deg, #1a1a3a 0%, var(--bg-dark) 100%);
            border-bottom: 1px solid var(--border);
        }}
        h1 {{
            color: var(--text-primary);
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-bottom: 15px;
        }}
        .timeline-link {{
            display: inline-block;
            padding: 12px 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            text-decoration: none;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 20px;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        .timeline-link:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 30px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--accent);
            display: block;
        }}
        .stat-label {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Filters */
        .filters {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px 25px;
            margin-bottom: 30px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }}
        .filter-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 500;
            margin-right: 5px;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-dark);
        }}
        .search-box {{
            padding: 10px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-dark);
            color: var(--text-primary);
            font-size: 0.9rem;
            width: 280px;
            margin-left: auto;
        }}
        .search-box:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        .filter-divider {{
            width: 1px;
            height: 30px;
            background: var(--border);
            margin: 0 10px;
        }}

        /* Year sections */
        .year-section {{
            margin-bottom: 50px;
        }}
        .year-header {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--accent);
            display: inline-block;
        }}

        /* Paper grid - PSI Lab style */
        .papers-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 25px;
        }}

        /* Paper card */
        .paper-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
        }}
        .paper-card:hover {{
            transform: translateY(-4px);
            border-color: var(--accent);
            box-shadow: 0 20px 40px rgba(0, 212, 255, 0.1);
        }}

        /* Card thumbnail area */
        .card-visual {{
            height: 180px;
            background: linear-gradient(135deg, #1a1a3a 0%, #2a2a5a 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }}
        .card-visual-icon {{
            font-size: 4rem;
            opacity: 0.3;
        }}
        .card-badges {{
            position: absolute;
            top: 12px;
            right: 12px;
            display: flex;
            gap: 6px;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-citations {{
            background: var(--warning);
            color: #000;
        }}
        .badge-new {{
            background: var(--success);
            color: #000;
        }}
        .badge-highlight {{
            background: var(--highlight);
            color: #fff;
        }}
        /* Research theme badges */
        .theme-badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .theme-L {{ background: #3b82f6; color: #fff; }}  /* Locomotion - Blue */
        .theme-I {{ background: #8b5cf6; color: #fff; }}  /* Imitation - Purple */
        .theme-M {{ background: #f59e0b; color: #000; }}  /* Manipulation - Orange */
        .theme-E {{ background: #10b981; color: #000; }}  /* Embodied AI - Green */
        .theme-R {{ background: #ef4444; color: #fff; }}  /* Multi-Robot - Red */
        .theme-S {{ background: #6366f1; color: #fff; }}  /* Safety - Indigo */
        /* Read rating badges */
        .rating-must_read {{ background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #000; }}
        .rating-optional {{ background: rgba(255,255,255,0.2); color: var(--text-secondary); border: 1px solid var(--border); }}
        .rating-skip {{ background: rgba(100,100,100,0.3); color: var(--text-secondary); }}
        /* Summary info block */
        .summary-block {{
            background: rgba(0, 212, 255, 0.05);
            border-left: 3px solid var(--accent);
            padding: 12px 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
        .summary-label {{
            font-size: 0.7rem;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .summary-text {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}
        .pros-cons {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }}
        .pros {{ border-left-color: var(--success); }}
        .cons {{ border-left-color: var(--highlight); }}
        .theme-tags {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            margin-top: 8px;
        }}
        .card-source {{
            position: absolute;
            bottom: 12px;
            left: 12px;
            padding: 4px 10px;
            background: rgba(0,0,0,0.6);
            border-radius: 6px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .card-tech {{
            position: absolute;
            bottom: 12px;
            right: 12px;
            display: flex;
            gap: 4px;
        }}
        .tech-chip {{
            padding: 3px 8px;
            background: rgba(102, 126, 234, 0.3);
            border-radius: 4px;
            font-size: 0.65rem;
            color: var(--accent-secondary);
        }}

        /* Card content */
        .card-content {{
            padding: 20px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        .card-title {{
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 10px;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .card-title a {{
            color: inherit;
            text-decoration: none;
        }}
        .card-title a:hover {{
            color: var(--accent);
        }}
        .card-authors {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 12px;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .card-keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 15px;
        }}
        .keyword-tag {{
            padding: 3px 10px;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            font-size: 0.7rem;
            color: var(--accent);
        }}

        /* Card details (expandable) */
        .card-details {{
            display: none;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid var(--border);
        }}
        .card-details.show {{
            display: block;
        }}
        .detail-block {{
            margin-bottom: 15px;
        }}
        .detail-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--accent);
            margin-bottom: 6px;
            font-weight: 600;
        }}
        .detail-text {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        .abstract-text {{
            max-height: 150px;
            overflow-y: auto;
        }}

        /* Card footer */
        .card-footer {{
            padding: 15px 20px;
            background: rgba(0,0,0,0.2);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .card-link {{
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.8rem;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .link-paper {{
            background: var(--accent);
            color: var(--bg-dark);
        }}
        .link-pdf {{
            background: var(--success);
            color: #000;
        }}
        .link-doi {{
            background: var(--warning);
            color: #000;
        }}
        .card-link:hover {{
            transform: scale(1.05);
        }}
        .expand-toggle {{
            margin-left: auto;
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }}
        .expand-toggle:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .cite-btn {{
            background: linear-gradient(135deg, #764ba2, #667eea);
            border: none;
            color: #fff;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .cite-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        .cite-btn.copied {{
            background: var(--success);
        }}
        .cite-tooltip {{
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--success);
            color: #000;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }}
        .cite-tooltip.show {{
            opacity: 1;
        }}

        /* View Toggle */
        .view-toggle {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
        }}
        .toggle-btn {{
            padding: 12px 24px;
            border-radius: 8px;
            border: 2px solid var(--border);
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.95rem;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .toggle-btn:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .toggle-btn.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-dark);
        }}

        /* Section headers */
        .section-header {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--accent);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-count {{
            font-size: 1rem;
            font-weight: 400;
            color: var(--text-secondary);
        }}
        .group-section {{
            margin-bottom: 50px;
        }}

        /* Utility */
        .hidden {{ display: none !important; }}

        /* Responsive */
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            h1 {{ font-size: 1.8rem; }}
            .papers-grid {{ grid-template-columns: 1fr; }}
            .stats {{ flex-direction: column; gap: 20px; }}
            .search-box {{ width: 100%; margin: 10px 0; }}
            .filters {{ flex-direction: column; align-items: stretch; }}
            .view-toggle {{ flex-wrap: wrap; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Research Literature Review</h1>
        <p class="subtitle">Multi-Source Academic Paper Analysis | {', '.join(kb.name for kb in knowledge_bases)}</p>
        <a href="research_timeline.html" class="timeline-link">🗺️ Interactive Timeline & Mindmap →</a>
        <div class="stats">
            <div class="stat">
                <span class="stat-value">{len(papers)}</span>
                <span class="stat-label">Papers</span>
            </div>
            <div class="stat">
                <span class="stat-value">{len(by_source)}</span>
                <span class="stat-label">Sources</span>
            </div>
            <div class="stat">
                <span class="stat-value">{len(sorted_categories)}</span>
                <span class="stat-label">Categories</span>
            </div>
            <div class="stat">
                <span class="stat-value">{len(sorted_years)}</span>
                <span class="stat-label">Years</span>
            </div>
        </div>
    </header>

    <div class="container">
        <!-- View Toggle -->
        <div class="view-toggle">
            <span class="filter-label">Group by:</span>
            <button class="toggle-btn active" onclick="setView('year')" id="btn-year">📅 Year</button>
            <button class="toggle-btn" onclick="setView('category')" id="btn-category">🏷️ Category</button>
        </div>

        <div class="filters">
            <span class="filter-label">Source:</span>
            <button class="filter-btn source-btn active" onclick="filterBySource('all')">All</button>
            {filter_buttons}
            <div class="filter-divider"></div>
            <span class="filter-label">Year:</span>
            <button class="filter-btn year-btn active" onclick="filterByYear('all')">All</button>
            {year_buttons}
            <input type="text" class="search-box" placeholder="Search papers & keywords..." oninput="searchPapers(this.value)">
        </div>

        <div class="filters" id="category-filters">
            <span class="filter-label">Category:</span>
            <button class="filter-btn cat-btn active" onclick="filterByCategory('all')">All</button>
            {category_buttons}
        </div>

        <!-- Year View -->
        <div id="year-view">
"""

    # Generate papers grouped by year
    for year in sorted_years:
        year_papers = by_year[year]
        html_content += f"""
            <div class="group-section year-section" data-year="{year}">
                <h2 class="section-header">📅 {year} <span class="section-count">({len(year_papers)} papers)</span></h2>
                <div class="papers-grid">
"""
        for paper in year_papers:
            authors_short = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_short += f" et al."

            # Keywords HTML
            keywords_html = "".join(
                f'<span class="keyword-tag">{esc(kw)}</span>'
                for kw in paper.keywords[:4]
            )

            # Tech chips
            tech_list = paper.core_technology.split(", ")[:2] if paper.core_technology != "Not specified" else []
            tech_html = "".join(f'<span class="tech-chip">{esc(t)}</span>' for t in tech_list)

            # Badges
            badges_html = ""
            if paper.citations and paper.citations > 50:
                badges_html += '<span class="badge badge-citations">High Impact</span>'
            elif paper.citations and paper.citations > 20:
                badges_html += '<span class="badge badge-citations">Cited</span>'
            if paper.year >= datetime.now().year:
                badges_html += '<span class="badge badge-new">New</span>'

            # Links
            links_html = ""
            if paper.url:
                links_html += f'<a href="{esc(paper.url)}" class="card-link link-paper" target="_blank">📄 Paper</a>'
            if paper.pdf_url:
                links_html += f'<a href="{esc(paper.pdf_url)}" class="card-link link-pdf" target="_blank">⬇ PDF</a>'
            if paper.doi:
                links_html += f'<a href="https://doi.org/{esc(paper.doi)}" class="card-link link-doi" target="_blank">🔗 DOI</a>'

            # IEEE Citation
            ieee_citation = generate_ieee_citation(paper)
            citation_escaped = esc(ieee_citation).replace("'", "\\'").replace('"', '&quot;')

            # Research theme badges
            theme_names = {"L": "Locomotion", "I": "Imitation", "M": "Manipulation",
                          "E": "Embodied AI", "R": "Multi-Robot", "S": "Safety"}
            themes_html = "".join(
                f'<span class="theme-badge theme-{t}">{theme_names.get(t, t)}</span>'
                for t in (paper.research_themes or [])[:3]
            )

            # Read rating badge
            rating_labels = {"must_read": "⭐ Must Read", "optional": "📖 Optional", "skip": "⏭️ Skip"}
            rating_html = f'<span class="badge rating-{paper.read_rating}">{rating_labels.get(paper.read_rating, "")}</span>' if paper.read_rating else ""

            # Visual icon based on category
            icon = "🤖"
            if paper.categories:
                cat = paper.categories[0].lower()
                if "cs.cv" in cat or "vision" in cat:
                    icon = "👁"
                elif "cs.lg" in cat or "learning" in cat:
                    icon = "🧠"
                elif "cs.ro" in cat or "robot" in cat:
                    icon = "🤖"
                elif "cs.ai" in cat:
                    icon = "🔮"

            # Summary info HTML
            summary_html = ""
            if paper.summary_cn:
                summary_html += f'<div class="summary-block"><div class="summary-label">📝 Summary</div><p class="summary-text">{esc(paper.summary_cn)}</p></div>'
            if paper.contribution_cn:
                summary_html += f'<div class="summary-block"><div class="summary-label">💡 Contribution</div><p class="summary-text">{esc(paper.contribution_cn)}</p></div>'
            if paper.pros or paper.cons:
                summary_html += '<div class="pros-cons">'
                if paper.pros:
                    summary_html += f'<div class="summary-block pros"><div class="summary-label">✅ Pros</div><p class="summary-text">{esc(paper.pros)}</p></div>'
                if paper.cons:
                    summary_html += f'<div class="summary-block cons"><div class="summary-label">⚠️ Cons</div><p class="summary-text">{esc(paper.cons)}</p></div>'
                summary_html += '</div>'

            html_content += f"""
                    <div class="paper-card" data-source="{esc(paper.source)}" data-year="{paper.year}" data-category="{esc(get_primary_category(paper))}" data-title="{esc(paper.title.lower())}" data-keywords="{esc(' '.join(paper.keywords).lower())}" data-themes="{' '.join(paper.research_themes or [])}">
                        <div class="card-visual">
                            <span class="card-visual-icon">{icon}</span>
                            <div class="card-badges">{badges_html}{rating_html}</div>
                            <span class="card-source">{esc(paper.source)} · {get_category_name(get_primary_category(paper))}</span>
                            <div class="card-tech">{tech_html}</div>
                        </div>
                        <div class="card-content">
                            <h3 class="card-title">
                                <a href="{esc(paper.url or '#')}" target="_blank">{esc(paper.title)}</a>
                            </h3>
                            <p class="card-authors">{esc(authors_short)}</p>
                            <div class="theme-tags">{themes_html}</div>
                            <div class="card-keywords">{keywords_html}</div>

                            <div class="card-details" id="details-{id(paper)}">
                                {summary_html}
                                <div class="detail-block">
                                    <div class="detail-label">Abstract</div>
                                    <p class="detail-text abstract-text">{esc(paper.abstract[:600])}{'...' if len(paper.abstract) > 600 else ''}</p>
                                </div>
                                {f'<div class="detail-block"><div class="detail-label">Core Contribution</div><p class="detail-text">{esc(paper.core_contribution)}</p></div>' if paper.core_contribution else ''}
                                {f'<div class="detail-block"><div class="detail-label">Problems Addressed</div><p class="detail-text">{esc(paper.problems_addressed)}</p></div>' if paper.problems_addressed else ''}
                                {f'<div class="detail-block"><div class="detail-label">Methodology</div><p class="detail-text">{esc(paper.methodology)}</p></div>' if paper.methodology else ''}
                            </div>
                        </div>
                        <div class="card-footer">
                            {links_html}
                            <button class="cite-btn" onclick="copyCitation(this, '{citation_escaped}')">📋 Cite</button>
                            <button class="expand-toggle" onclick="toggleCard(this, 'details-{id(paper)}')">Show More</button>
                        </div>
                    </div>
"""
        html_content += """
                </div>
            </div>
"""

    # Close year view
    html_content += """
        </div>

        <!-- Category View -->
        <div id="category-view" class="hidden">
"""

    # Generate papers grouped by category
    for cat in sorted_categories:
        cat_papers = by_category[cat]
        cat_display = get_category_name(cat)
        html_content += f"""
            <div class="group-section cat-section" data-category="{esc(cat)}">
                <h2 class="section-header">🏷️ {cat_display} <span class="section-count">({len(cat_papers)} papers)</span></h2>
                <div class="papers-grid">
"""
        for paper in cat_papers:
            authors_short = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_short += " et al."

            keywords_html = "".join(
                f'<span class="keyword-tag">{esc(kw)}</span>'
                for kw in paper.keywords[:4]
            )

            tech_list = paper.core_technology.split(", ")[:2] if paper.core_technology != "Not specified" else []
            tech_html = "".join(f'<span class="tech-chip">{esc(t)}</span>' for t in tech_list)

            badges_html = ""
            if paper.citations and paper.citations > 50:
                badges_html += '<span class="badge badge-citations">High Impact</span>'
            elif paper.citations and paper.citations > 20:
                badges_html += '<span class="badge badge-citations">Cited</span>'
            if paper.year >= datetime.now().year:
                badges_html += '<span class="badge badge-new">New</span>'

            links_html = ""
            if paper.url:
                links_html += f'<a href="{esc(paper.url)}" class="card-link link-paper" target="_blank">📄 Paper</a>'
            if paper.pdf_url:
                links_html += f'<a href="{esc(paper.pdf_url)}" class="card-link link-pdf" target="_blank">⬇ PDF</a>'
            if paper.doi:
                links_html += f'<a href="https://doi.org/{esc(paper.doi)}" class="card-link link-doi" target="_blank">🔗 DOI</a>'

            # IEEE Citation for category view
            ieee_citation_cat = generate_ieee_citation(paper)
            citation_escaped_cat = esc(ieee_citation_cat).replace("'", "\\'").replace('"', '&quot;')

            # Research theme badges (category view)
            theme_names = {"L": "Locomotion", "I": "Imitation", "M": "Manipulation",
                          "E": "Embodied AI", "R": "Multi-Robot", "S": "Safety"}
            themes_html_cat = "".join(
                f'<span class="theme-badge theme-{t}">{theme_names.get(t, t)}</span>'
                for t in (paper.research_themes or [])[:3]
            )
            rating_labels = {"must_read": "⭐ Must Read", "optional": "📖 Optional", "skip": "⏭️ Skip"}
            rating_html_cat = f'<span class="badge rating-{paper.read_rating}">{rating_labels.get(paper.read_rating, "")}</span>' if paper.read_rating else ""

            icon = "🤖"
            if paper.categories:
                c = paper.categories[0].lower()
                if "cs.cv" in c:
                    icon = "👁"
                elif "cs.lg" in c:
                    icon = "🧠"
                elif "cs.ro" in c:
                    icon = "🤖"
                elif "cs.ai" in c:
                    icon = "🔮"
                elif "cs.ma" in c:
                    icon = "👥"

            # Summary info HTML (category view)
            summary_html_cat = ""
            if paper.summary_cn:
                summary_html_cat += f'<div class="summary-block"><div class="summary-label">📝 Summary</div><p class="summary-text">{esc(paper.summary_cn)}</p></div>'
            if paper.contribution_cn:
                summary_html_cat += f'<div class="summary-block"><div class="summary-label">💡 Contribution</div><p class="summary-text">{esc(paper.contribution_cn)}</p></div>'

            html_content += f"""
                    <div class="paper-card" data-source="{esc(paper.source)}" data-year="{paper.year}" data-category="{esc(get_primary_category(paper))}" data-title="{esc(paper.title.lower())}" data-keywords="{esc(' '.join(paper.keywords).lower())}" data-themes="{' '.join(paper.research_themes or [])}">
                        <div class="card-visual">
                            <span class="card-visual-icon">{icon}</span>
                            <div class="card-badges">{badges_html}{rating_html_cat}</div>
                            <span class="card-source">{esc(paper.source)} · {paper.year}</span>
                            <div class="card-tech">{tech_html}</div>
                        </div>
                        <div class="card-content">
                            <h3 class="card-title">
                                <a href="{esc(paper.url or '#')}" target="_blank">{esc(paper.title)}</a>
                            </h3>
                            <p class="card-authors">{esc(authors_short)}</p>
                            <div class="theme-tags">{themes_html_cat}</div>
                            <div class="card-keywords">{keywords_html}</div>
                            <div class="card-details" id="cat-details-{id(paper)}">
                                {summary_html_cat}
                                <div class="detail-block">
                                    <div class="detail-label">Abstract</div>
                                    <p class="detail-text abstract-text">{esc(paper.abstract[:600])}{'...' if len(paper.abstract) > 600 else ''}</p>
                                </div>
                                {f'<div class="detail-block"><div class="detail-label">Core Contribution</div><p class="detail-text">{esc(paper.core_contribution)}</p></div>' if paper.core_contribution else ''}
                            </div>
                        </div>
                        <div class="card-footer">
                            {links_html}
                            <button class="cite-btn" onclick="copyCitation(this, '{citation_escaped_cat}')">📋 Cite</button>
                            <button class="expand-toggle" onclick="toggleCard(this, 'cat-details-{id(paper)}')">Show More</button>
                        </div>
                    </div>
"""
        html_content += """
                </div>
            </div>
"""

    # Close category view
    html_content += """
        </div>
    </div>

    <!-- Citation Tooltip -->
    <div class="cite-tooltip" id="cite-tooltip">✓ IEEE Citation Copied!</div>

    <script>
        let currentSource = 'all';
        let currentYear = 'all';
        let currentCategory = 'all';
        let currentSearch = '';
        let currentView = 'year';

        function setView(view) {
            currentView = view;
            document.getElementById('btn-year').classList.toggle('active', view === 'year');
            document.getElementById('btn-category').classList.toggle('active', view === 'category');
            document.getElementById('year-view').classList.toggle('hidden', view !== 'year');
            document.getElementById('category-view').classList.toggle('hidden', view !== 'category');

            // Reset filters when switching views
            currentYear = 'all';
            currentCategory = 'all';
            document.querySelectorAll('.year-btn, .cat-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.year-btn').classList.add('active');
            document.querySelector('.cat-btn').classList.add('active');

            applyFilters();
        }

        function applyFilters() {
            const activeView = currentView === 'year' ? '#year-view' : '#category-view';

            document.querySelectorAll(activeView + ' .paper-card').forEach(card => {
                const matchSource = currentSource === 'all' || card.dataset.source === currentSource;
                const matchYear = currentYear === 'all' || card.dataset.year === String(currentYear);
                const matchCategory = currentCategory === 'all' || card.dataset.category === currentCategory;
                const matchSearch = currentSearch === '' ||
                    card.dataset.title.includes(currentSearch) ||
                    card.dataset.keywords.includes(currentSearch);

                if (matchSource && matchYear && matchCategory && matchSearch) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });

            // Hide empty sections
            document.querySelectorAll(activeView + ' .group-section').forEach(section => {
                const visibleCards = section.querySelectorAll('.paper-card:not(.hidden)');
                section.style.display = visibleCards.length === 0 ? 'none' : 'block';
            });
        }

        function filterBySource(source) {
            currentSource = source;
            document.querySelectorAll('.source-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }

        function filterByYear(year) {
            currentYear = year;
            document.querySelectorAll('.year-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }

        function filterByCategory(category) {
            currentCategory = category;
            document.querySelectorAll('.cat-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }

        function searchPapers(query) {
            currentSearch = query.toLowerCase();
            applyFilters();
        }

        function toggleCard(btn, detailsId) {
            const details = document.getElementById(detailsId);
            details.classList.toggle('show');
            btn.textContent = details.classList.contains('show') ? 'Show Less' : 'Show More';
        }

        function copyCitation(btn, citation) {
            // Decode HTML entities
            const textarea = document.createElement('textarea');
            textarea.innerHTML = citation;
            const decodedCitation = textarea.value;

            // Copy to clipboard
            navigator.clipboard.writeText(decodedCitation).then(() => {
                // Visual feedback on button
                const originalText = btn.innerHTML;
                btn.innerHTML = '✓ Copied!';
                btn.classList.add('copied');

                // Show tooltip
                const tooltip = document.getElementById('cite-tooltip');
                tooltip.classList.add('show');

                // Reset after 2 seconds
                setTimeout(() => {
                    btn.innerHTML = originalText;
                    btn.classList.remove('copied');
                    tooltip.classList.remove('show');
                }, 2000);
            }).catch(err => {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = decodedCitation;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                document.body.appendChild(textArea);
                textArea.select();
                try {
                    document.execCommand('copy');
                    btn.innerHTML = '✓ Copied!';
                    btn.classList.add('copied');
                    setTimeout(() => {
                        btn.innerHTML = '📋 Cite';
                        btn.classList.remove('copied');
                    }, 2000);
                } catch (e) {
                    alert('Failed to copy citation. Please copy manually:\\n\\n' + decodedCitation);
                }
                document.body.removeChild(textArea);
            });
        }
    </script>
</body>
</html>
"""

    output = Path(output_path)
    output.write_text(html_content, encoding='utf-8')
    print(f"\n✅ HTML report saved to: {output.absolute()}")

    return str(output.absolute())


def generate_timeline_mindmap(papers: list[Paper], output_path: str = "research_timeline.html") -> str:
    """Generate an interactive timeline and mindmap visualization."""

    def esc(text):
        return html.escape(str(text)) if text else ""

    # Group papers by theme and year
    by_theme = {}
    by_year = {}
    theme_names = {
        "L": "Locomotion", "I": "Imitation", "M": "Manipulation",
        "E": "Embodied AI", "R": "Multi-Robot", "S": "Safety/Deploy"
    }

    for p in papers:
        for theme in (p.research_themes or ["L"]):
            by_theme.setdefault(theme, []).append(p)
        by_year.setdefault(p.year, []).append(p)

    sorted_years = sorted(by_year.keys())

    # Research gaps from gaps.md
    gaps_data = {
        "Gap-L1": {"theme": "L", "desc": "Unified framework for non-periodic, contact-rich transitions"},
        "Gap-I1": {"theme": "I", "desc": "Demonstration generalization and morphology mismatch"},
        "Gap-M1": {"theme": "M", "desc": "Cross-task generalization and reusable motion representations"},
        "Gap-E1": {"theme": "E", "desc": "Bridge high-level reasoning and low-level physical feasibility"},
        "Gap-R1": {"theme": "R", "desc": "Multi-humanoid coordination with whole-body dynamics"},
        "Gap-S1": {"theme": "S", "desc": "Safety-aware, scalable real-world learning frameworks"}
    }

    # Build papers JSON for JavaScript
    papers_json = []
    for p in papers:
        papers_json.append({
            "title": p.title[:80],
            "year": p.year,
            "themes": p.research_themes or ["L"],
            "rating": p.read_rating or "optional",
            "summary": p.summary_cn or p.abstract[:200],
            "url": p.url
        })

    # Generate theme stats
    theme_stats = []
    for theme_code, theme_name in theme_names.items():
        count = len(by_theme.get(theme_code, []))
        theme_stats.append({"code": theme_code, "name": theme_name, "count": count})

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Timeline & Mindmap - Humanoid Robotics</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        :root {{
            --accent: #00d4ff;
            --bg-dark: #0a0a1a;
            --bg-card: #12122a;
            --text-primary: #e8e8f0;
            --text-secondary: #888899;
            --border: #2a2a4a;
            --theme-L: #3b82f6;
            --theme-I: #8b5cf6;
            --theme-M: #f59e0b;
            --theme-E: #10b981;
            --theme-R: #ef4444;
            --theme-S: #6366f1;
        }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}

        /* Header */
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(180deg, #1a1a3a 0%, var(--bg-dark) 100%);
            border-bottom: 1px solid var(--border);
        }}
        h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .subtitle {{ color: var(--text-secondary); }}

        /* Tab Navigation */
        .tabs {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 30px 0;
        }}
        .tab-btn {{
            padding: 12px 30px;
            border: 2px solid var(--border);
            background: transparent;
            color: var(--text-secondary);
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
        .tab-btn.active {{ background: var(--accent); border-color: var(--accent); color: #000; }}

        /* Views */
        .view {{ display: none; }}
        .view.active {{ display: block; }}

        /* Timeline */
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 4px;
            background: linear-gradient(180deg, var(--theme-L), var(--theme-E), var(--theme-R));
            transform: translateX(-50%);
        }}
        .timeline-year {{
            position: relative;
            margin: 40px 0;
        }}
        .year-marker {{
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            background: var(--accent);
            color: #000;
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.2rem;
            z-index: 2;
        }}
        .timeline-papers {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 60px;
            padding: 0 20px;
        }}
        .timeline-papers > div:nth-child(odd) {{ text-align: right; padding-right: 40px; }}
        .timeline-papers > div:nth-child(even) {{ text-align: left; padding-left: 40px; }}

        .paper-node {{
            display: inline-block;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            max-width: 350px;
            text-align: left;
            transition: all 0.3s;
            cursor: pointer;
        }}
        .paper-node:hover {{
            border-color: var(--accent);
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0,212,255,0.15);
        }}
        .paper-node.must_read {{ border-left: 4px solid #fbbf24; }}
        .paper-node .title {{ font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; line-height: 1.3; }}
        .paper-node .themes {{ display: flex; gap: 4px; flex-wrap: wrap; }}
        .theme-dot {{
            width: 8px; height: 8px; border-radius: 50%;
        }}
        .theme-dot.L {{ background: var(--theme-L); }}
        .theme-dot.I {{ background: var(--theme-I); }}
        .theme-dot.M {{ background: var(--theme-M); }}
        .theme-dot.E {{ background: var(--theme-E); }}
        .theme-dot.R {{ background: var(--theme-R); }}
        .theme-dot.S {{ background: var(--theme-S); }}

        /* Mindmap */
        .mindmap-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 80vh;
            position: relative;
        }}
        .mindmap {{
            position: relative;
            width: 900px;
            height: 900px;
        }}
        .central-node {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 180px;
            height: 180px;
            background: linear-gradient(135deg, var(--accent), #667eea);
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #000;
            font-weight: 700;
            font-size: 1.1rem;
            text-align: center;
            z-index: 10;
            box-shadow: 0 0 60px rgba(0,212,255,0.4);
        }}
        .central-node .count {{ font-size: 2.5rem; }}
        .theme-node {{
            position: absolute;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }}
        .theme-node:hover {{
            transform: scale(1.15);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .theme-node .count {{ font-size: 2rem; }}
        .theme-node .name {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }}
        .theme-node.L {{ background: var(--theme-L); top: 5%; left: 50%; transform: translateX(-50%); }}
        .theme-node.I {{ background: var(--theme-I); top: 20%; right: 10%; }}
        .theme-node.M {{ background: var(--theme-M); bottom: 20%; right: 10%; }}
        .theme-node.E {{ background: var(--theme-E); bottom: 5%; left: 50%; transform: translateX(-50%); }}
        .theme-node.R {{ background: var(--theme-R); bottom: 20%; left: 10%; }}
        .theme-node.S {{ background: var(--theme-S); top: 20%; left: 10%; }}

        .connector {{
            position: absolute;
            background: linear-gradient(90deg, transparent, var(--border), transparent);
            height: 2px;
            z-index: 1;
        }}

        /* Gaps View */
        .gaps-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            padding: 20px;
        }}
        .gap-card {{
            background: var(--bg-card);
            border: 2px solid var(--border);
            border-radius: 12px;
            padding: 25px;
            transition: all 0.3s;
        }}
        .gap-card:hover {{ border-color: var(--accent); transform: translateY(-5px); }}
        .gap-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 15px;
        }}
        .gap-icon {{
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 700;
            font-size: 1.2rem;
        }}
        .gap-title {{ font-size: 1.3rem; font-weight: 700; }}
        .gap-desc {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 15px;
            padding: 15px;
            background: rgba(0,212,255,0.05);
            border-left: 3px solid var(--accent);
            border-radius: 0 8px 8px 0;
        }}
        .gap-papers {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }}
        .gap-paper-list {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
        }}
        .gap-paper-item {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 8px 10px;
            border-radius: 6px;
            margin-bottom: 6px;
            background: rgba(255,255,255,0.03);
            transition: all 0.2s;
        }}
        .gap-paper-item:hover {{
            background: rgba(0,212,255,0.1);
        }}
        .gap-paper-item:last-child {{ margin-bottom: 0; }}
        .gap-paper-item .rating-icon {{
            flex-shrink: 0;
            font-size: 0.9rem;
        }}
        .gap-paper-item a {{
            color: var(--text-primary);
            text-decoration: none;
            font-size: 0.85rem;
            line-height: 1.4;
            flex: 1;
        }}
        .gap-paper-item a:hover {{
            color: var(--accent);
            text-decoration: underline;
        }}
        .gap-paper-item.must-read {{
            border-left: 3px solid #fbbf24;
            background: rgba(251, 191, 36, 0.1);
        }}
        .gap-paper-item.optional {{
            border-left: 3px solid #60a5fa;
        }}
        .gap-paper-item.skip {{
            border-left: 3px solid #6b7280;
            opacity: 0.7;
        }}

        /* Stats */
        .stats-row {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        .stat-box {{
            text-align: center;
            padding: 20px 30px;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .stat-value {{ font-size: 2.5rem; font-weight: 700; color: var(--accent); }}
        .stat-label {{ font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; }}

        /* Back link */
        .back-link {{
            display: inline-block;
            margin: 20px;
            padding: 10px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--accent);
            text-decoration: none;
            transition: all 0.2s;
        }}
        .back-link:hover {{ border-color: var(--accent); background: rgba(0,212,255,0.1); }}

        @media (max-width: 900px) {{
            .timeline::before {{ left: 20px; }}
            .timeline-papers {{ grid-template-columns: 1fr; padding-left: 50px; }}
            .timeline-papers > div {{ text-align: left !important; padding: 0 !important; }}
            .year-marker {{ left: 20px; transform: none; }}
            .mindmap {{ transform: scale(0.6); }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>🤖 Humanoid Robotics Research</h1>
        <p class="subtitle">Interactive Timeline & Research Landscape</p>
        <a href="research_report.html" class="back-link">← Back to Full Report</a>
    </header>

    <div class="stats-row">
        <div class="stat-box">
            <div class="stat-value">{len(papers)}</div>
            <div class="stat-label">Papers</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(sorted_years)}</div>
            <div class="stat-label">Years</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len([p for p in papers if p.read_rating == 'must_read'])}</div>
            <div class="stat-label">Must Read</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">6</div>
            <div class="stat-label">Research Themes</div>
        </div>
    </div>

    <div class="tabs">
        <button class="tab-btn active" onclick="showView('timeline')">📅 Timeline</button>
        <button class="tab-btn" onclick="showView('mindmap')">🧠 Research Map</button>
        <button class="tab-btn" onclick="showView('gaps')">🔬 Research Gaps</button>
    </div>

    <div class="container">
        <!-- Timeline View -->
        <div id="timeline-view" class="view active">
            <div class="timeline">
"""

    # Generate timeline content
    for year in sorted(sorted_years, reverse=True):
        year_papers = by_year[year]
        html_content += f'''
                <div class="timeline-year">
                    <div class="year-marker">{year}</div>
                    <div class="timeline-papers">
'''
        for p in year_papers[:12]:  # Limit to 12 per year for readability
            rating_class = "must_read" if p.read_rating == "must_read" else ""
            themes_dots = "".join(f'<span class="theme-dot {t}"></span>' for t in (p.research_themes or ["L"])[:3])
            html_content += f'''
                        <div>
                            <a href="{esc(p.url)}" target="_blank" style="text-decoration:none;">
                                <div class="paper-node {rating_class}">
                                    <div class="title">{esc(p.title[:70])}{'...' if len(p.title) > 70 else ''}</div>
                                    <div class="themes">{themes_dots}</div>
                                </div>
                            </a>
                        </div>
'''
        if len(year_papers) > 12:
            html_content += f'<div style="text-align:center;grid-column:1/-1;color:var(--text-secondary);">+ {len(year_papers)-12} more papers</div>'
        html_content += '''
                    </div>
                </div>
'''

    html_content += """
            </div>
        </div>

        <!-- Mindmap View -->
        <div id="mindmap-view" class="view">
            <div class="mindmap-container">
                <div class="mindmap">
                    <div class="central-node">
                        <div class="count">""" + str(len(papers)) + """</div>
                        <div>Papers</div>
                    </div>
"""

    # Add theme nodes
    for stat in theme_stats:
        html_content += f'''
                    <div class="theme-node {stat['code']}" onclick="filterByTheme('{stat['code']}')">
                        <div class="count">{stat['count']}</div>
                        <div class="name">{stat['name']}</div>
                    </div>
'''

    html_content += """
                </div>
            </div>
            <div style="text-align:center;margin-top:20px;color:var(--text-secondary);">
                Click a theme node to filter papers in the main report
            </div>
        </div>

        <!-- Research Gaps View -->
        <div id="gaps-view" class="view">
            <div class="gaps-grid">
"""

    # Add gap cards with sorted paper lists
    rating_order = {"must_read": 0, "optional": 1, "skip": 2, "": 3}
    rating_icons = {"must_read": "⭐", "optional": "📖", "skip": "⏭️", "": "📄"}

    for gap_id, gap_info in gaps_data.items():
        theme_code = gap_info["theme"]
        theme_color = f"var(--theme-{theme_code})"
        theme_name = theme_names.get(theme_code, theme_code)
        related_papers = by_theme.get(theme_code, [])

        # Sort papers: must_read first, then optional, then skip
        sorted_papers = sorted(related_papers, key=lambda p: (rating_order.get(p.read_rating, 3), p.title))

        # Generate paper list HTML
        paper_list_html = ""
        for p in sorted_papers:
            rating = p.read_rating or ""
            rating_icon = rating_icons.get(rating, "📄")
            rating_class = "must-read" if rating == "must_read" else rating if rating else "optional"
            paper_url = p.url or "#"
            paper_title = p.title[:80] + "..." if len(p.title) > 80 else p.title
            paper_list_html += f'''
                        <div class="gap-paper-item {rating_class}">
                            <span class="rating-icon">{rating_icon}</span>
                            <a href="{paper_url}" target="_blank" title="{p.title}">{paper_title}</a>
                        </div>'''

        html_content += f'''
                <div class="gap-card">
                    <div class="gap-header">
                        <div class="gap-icon" style="background:{theme_color}">{theme_code}</div>
                        <div class="gap-title">{gap_id}: {theme_name}</div>
                    </div>
                    <div class="gap-desc">"{gap_info['desc']}"</div>
                    <div class="gap-papers">📚 {len(sorted_papers)} related papers in this theme</div>
                    <div class="gap-paper-list">{paper_list_html}
                    </div>
                </div>
'''

    html_content += """
            </div>
        </div>
    </div>

    <script>
        function showView(viewId) {
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(viewId + '-view').classList.add('active');
            event.target.classList.add('active');
        }

        function filterByTheme(theme) {
            // Navigate to main report with theme filter
            window.location.href = 'research_report.html#theme-' + theme;
        }
    </script>
</body>
</html>
"""

    output = Path(output_path)
    output.write_text(html_content, encoding='utf-8')
    print(f"✅ Timeline/Mindmap saved to: {output.absolute()}")
    return str(output.absolute())


def main():
    print("=" * 60)
    print("   Multi-Source Research Paper Crawler")
    print("   Using Markdown Knowledge Bases")
    print("=" * 60)
    print()

    # Initialize crawler
    crawler = MultiSourceCrawler(KNOWLEDGE_BASE_DIR)

    # Load knowledge bases
    kbs = crawler.load_knowledge_bases()

    if not kbs:
        print("\n💡 Create .md files in this directory to define search sources.")
        print("   See README for examples.")
        return

    # Crawl all sources
    papers = crawler.crawl_all()

    if not papers:
        print("\n❌ No papers found. Check your search queries and try again.")
        return

    # Analyze papers for detailed extraction
    papers = analyze_all_papers(papers)

    # Enrich with summary.md data
    summary_file = KNOWLEDGE_BASE_DIR / "summary.md"
    if summary_file.exists():
        papers = enrich_papers_with_summary(papers, summary_file)

    # Generate report
    print(f"\n📝 Generating HTML report with {len(papers)} papers...")
    generate_html_report(papers, kbs, OUTPUT_FILE)

    # Generate interactive timeline/mindmap
    print(f"\n🗺️  Generating interactive timeline and mindmap...")
    generate_timeline_mindmap(papers, "research_timeline.html")

    # Download papers
    print("\n" + "=" * 60)
    print("   📥 Starting Paper Downloads")
    print("=" * 60)
    download_papers(papers, "papers")

    print("\n" + "=" * 60)
    print("   🎉 Done! Open research_report.html in your browser.")
    print("   🗺️  Interactive timeline: research_timeline.html")
    print("   📁 Papers downloaded to papers/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
