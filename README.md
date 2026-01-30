# Robotics Research Literature Review

Interactive HTML reports for multi-source academic paper analysis on humanoid robotics and multi-robot systems.

## Files

- **research_report.html** - Main literature review report with 264 papers
- **research_timeline.html** - Interactive timeline, research map, and research gaps visualization

## Features

### Main Report (`research_report.html`)
- 264 papers from multiple sources (arXiv, Google Scholar, Scopus/WOS)
- IEEE citation copy button for each paper
- Research theme badges (L/I/M/E/R/S)
- Read rating indicators (⭐ Must Read, 📖 Optional, ⏭️ Skip)
- Paper summaries in Chinese with contributions and limitations
- PDF download links

### Interactive Timeline (`research_timeline.html`)
- **Timeline View**: Papers organized by publication year
- **Research Map**: Mindmap visualization of 6 research themes
- **Research Gaps**: Detailed gap analysis with sorted paper lists

## Research Themes

| Code | Theme | Description |
|------|-------|-------------|
| **L** | Locomotion | Walking / Whole-body motion / Control |
| **I** | Imitation | Imitation learning |
| **M** | Manipulation | Object manipulation |
| **E** | Embodied AI | LLM / VLM / Reasoning |
| **R** | Multi-Robot | Multi-robot coordination |
| **S** | Safety | Real-world deployment / Safety |

## Research Gaps Identified

- **Gap-L1**: Unified framework for non-periodic, contact-rich transitions
- **Gap-I1**: Demonstration generalization and morphology mismatch
- **Gap-M1**: Cross-task generalization and reusable motion representations
- **Gap-E1**: Bridge high-level reasoning and low-level physical feasibility
- **Gap-R1**: Multi-humanoid coordination with whole-body dynamics
- **Gap-S1**: Safety-aware, scalable real-world learning frameworks

## Usage

Open `research_report.html` in any modern web browser. Click the "Interactive Timeline & Mindmap" button to access the visualization page.

## Data Sources

- arXiv (cs.RO, cs.AI)
- Google Scholar
- Scopus / Web of Science
- Manual curated summaries (summary.md)
