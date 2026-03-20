from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .text_safety import sanitize_provider_text

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings


class GuidelineRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._vectorstore: FAISS | None = None
        self._embedding_model = None

    def available_guidelines(self) -> list[Path]:
        return [path for path in self.settings.guideline_paths() if path.exists()]

    def _source_label(self, path: Path) -> str:
        name = path.name.lower()
        if "guideline-170" in name or "msf" in name:
            return "MSF"
        if "who" in name or "respiratory guidelines" in name:
            return "WHO"
        return path.stem

    def _get_embeddings(self):
        if self._embedding_model is None:
            from langchain_openai import OpenAIEmbeddings

            kwargs = {"openai_api_key": self.settings.openai_api_key}
            if self.settings.embedding_model:
                kwargs["model"] = self.settings.embedding_model
            self._embedding_model = OpenAIEmbeddings(**kwargs)
        return self._embedding_model

    def _load_or_build(self) -> FAISS:
        if self._vectorstore is not None:
            return self._vectorstore

        embeddings = self._get_embeddings()
        index_dir = self.settings.index_dir
        faiss_index = index_dir / "index.faiss"
        faiss_meta = index_dir / "index.pkl"

        if faiss_index.exists() and faiss_meta.exists():
            self._vectorstore = FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return self._vectorstore

        guideline_paths = self.available_guidelines()
        if not guideline_paths:
            raise FileNotFoundError(
                "No guideline PDFs were found in data/guidelines/. "
                "Add the WHO and MSF respiratory PDFs or set SMO_GUIDELINE_PATHS."
            )

        documents = []
        for path in guideline_paths:
            loader = PyPDFLoader(str(path))
            loaded = loader.load()
            source = self._source_label(path)
            for doc in loaded:
                doc.page_content = sanitize_provider_text(doc.page_content)
                doc.metadata["source"] = source
                doc.metadata["source_file"] = path.name
            documents.extend(loaded)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise RuntimeError("Guideline parsing succeeded, but no chunks were produced.")

        index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = FAISS.from_documents(chunks, embeddings)
        self._vectorstore.save_local(str(index_dir))
        return self._vectorstore

    def retrieve_context(self, query: str, k: int | None = None) -> str:
        vectorstore = self._load_or_build()
        safe_query = sanitize_provider_text(query)
        try:
            results = vectorstore.similarity_search(safe_query, k=k or self.settings.top_k)
        except Exception as exc:
            raise RuntimeError(f"Guideline retrieval failed during embedding lookup: {exc}") from exc

        seen: set[str] = set()
        filtered: list[str] = []
        for doc in results:
            chunk = sanitize_provider_text(doc.page_content)
            if chunk and chunk not in seen:
                filtered.append(chunk)
                seen.add(chunk)

        return "\n\n".join(filtered[: self.settings.top_k])
