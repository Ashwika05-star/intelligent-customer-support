import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import POLICIES_DIR, MANUALS_DIR, RETRIEVAL_TOP_K
from src.state import RetrievedDocument

class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        title: str,
        section: str,
        category: str,
        source_file: str,
        content: str
    ):
        self.chunk_id = chunk_id
        self.title = title
        self.section = section
        self.category = category  # "policy" or "manual"
        self.source_file = source_file
        self.content = content.strip()

    def to_dict(self, score: float = 0.0) -> RetrievedDocument:
        return {
            "id": self.chunk_id,
            "title": self.title,
            "section": self.section,
            "category": self.category,
            "source_file": self.source_file,
            "content": self.content,
            "score": round(score, 4),
        }

class KnowledgeBaseRetriever:
    """Hybrid Retriever supporting section-based chunking and similarity search."""
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _parse_markdown_sections(self, file_path: Path, category: str) -> List[DocumentChunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        lines = text.split("\n")
        doc_title = file_path.stem.replace("_", " ").title()
        
        # Check for first H1 header
        for line in lines:
            if line.startswith("# "):
                doc_title = line.replace("# ", "").strip()
                break

        chunks = []
        current_section = "Overview"
        current_lines = []
        chunk_idx = 1

        for line in lines:
            if line.startswith("## "):
                # Save existing section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if len(content) > 30:
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{file_path.stem}_sec_{chunk_idx}",
                                title=doc_title,
                                section=current_section,
                                category=category,
                                source_file=file_path.name,
                                content=content
                            )
                        )
                        chunk_idx += 1
                current_section = line.replace("## ", "").strip()
                current_lines = [f"{doc_title} - {current_section}"]
            else:
                current_lines.append(line)

        # Flush final section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if len(content) > 30:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{file_path.stem}_sec_{chunk_idx}",
                        title=doc_title,
                        section=current_section,
                        category=category,
                        source_file=file_path.name,
                        content=content
                    )
                )

        return chunks

    def _load_and_index(self):
        self.chunks = []
        # Index Policies
        if POLICIES_DIR.exists():
            for policy_file in POLICIES_DIR.glob("*.md"):
                self.chunks.extend(self._parse_markdown_sections(policy_file, category="policy"))

        # Index Manuals
        if MANUALS_DIR.exists():
            for manual_file in MANUALS_DIR.glob("*.md"):
                self.chunks.extend(self._parse_markdown_sections(manual_file, category="manual"))

        if not self.chunks:
            return

        corpus = [f"{c.title} {c.section}\n{c.content}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = RETRIEVAL_TOP_K
    ) -> List[RetrievedDocument]:
        """Search chunks with optional category filtering and keyword boosting."""
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Add explicit keyword boost for exact product models or policy keywords
        query_lower = query.lower()
        scored_results = []

        for idx, chunk in enumerate(self.chunks):
            # Category filter if provided
            if category and chunk.category != category:
                continue

            base_score = float(similarities[idx])
            
            # Boost score if query terms match title or section heading exactly
            chunk_text_lower = f"{chunk.title} {chunk.section} {chunk.content}".lower()
            keyword_boost = 0.0
            
            for word in query_lower.split():
                if len(word) > 3 and word in chunk_text_lower:
                    keyword_boost += 0.05
                    if word in chunk.title.lower() or word in chunk.section.lower():
                        keyword_boost += 0.10

            final_score = min(1.0, base_score + keyword_boost)
            scored_results.append((final_score, chunk))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_results[:top_k]

        return [chunk.to_dict(score=score) for score, chunk in top_results]

# Global singleton retriever
_retriever_instance: Optional[KnowledgeBaseRetriever] = None

def get_retriever() -> KnowledgeBaseRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KnowledgeBaseRetriever()
    return _retriever_instance
