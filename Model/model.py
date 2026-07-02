import os
import re
import pickle
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import faiss
from bs4 import BeautifulSoup
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Suppress warnings to keep CLI output clean
warnings.filterwarnings('ignore')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('so_search')

# Valid Tags List
VALID_TAGS = [
    'assembly', 'c#', 'c++', 'dart', 'go', 'haskell', 'java',
    'javascript', 'kotlin', 'lua', 'objective-c', 'perl', 'php',
    'python', 'r', 'ruby', 'rust', 'scala', 'swift', 'typescript'
]

class StackOverflowSearchEngine:
    """
    Stack Overflow Hybrid Q&A Search Engine
    Combines TF-IDF (lexical) and Sentence-BERT + FAISS (semantic) with answer scores.
    """
    def __init__(self, project_root=None):
        if project_root is None:
            # Resolve project root relative to this file (which resides in Model/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(script_dir)
        else:
            self.project_root = os.path.abspath(project_root)
            
        self.raw_dataset_path = os.path.join(self.project_root, 'Dataset', 'dataset.csv')
        self.clean_dataset_path = os.path.join(self.project_root, 'Dataset', 'dataset_clean.csv')
        self.tfidf_path = os.path.join(self.project_root, 'Model', 'tfidf_data.pkl')
        self.global_faiss_path = os.path.join(self.project_root, 'Model', 'global_faiss.index')
        self.local_faiss_path = os.path.join(self.project_root, 'Model', 'tag_local_indices.pkl')
        
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.embedding_model = None
        self.index = None
        self.tag_local_indices = {}
        self.translator = None

    def clean_html(self, text):
        if pd.isna(text):
            return ''
        # Strip HTML tags
        text = BeautifulSoup(str(text), 'html.parser').get_text(separator=' ')
        # Remove redundant whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_primary_tag(self, tag_str):
        if pd.isna(tag_str):
            return 'other'
        parts = str(tag_str).strip().lower().split()
        for part in parts:
            if part in VALID_TAGS:
                return part
        return 'other'

    def train_and_serialize(self, limit=None):
        """
        Cleans the dataset, fits TF-IDF, computes SBERT embeddings, builds FAISS indices,
        and serializes all models to disk to enable instant reloading.
        """
        if not os.path.exists(self.raw_dataset_path):
            raise FileNotFoundError(
                f"Dataset mentah tidak ditemukan di: {self.raw_dataset_path}\n"
                "Harap unduh dataset dan simpan sebagai 'Dataset/dataset.csv'."
            )

        logger.info("--- MEMULAI PROSES TRAINING DAN SERIALISASI ---")
        
        # 1. Load and Preprocess Dataset
        logger.info("Memuat dan membersihkan dataset...")
        df = pd.read_csv(self.raw_dataset_path)
        logger.info(f"Dataset awal dimuat: {df.shape}")
        
        # Apply preprocessing
        df['Title']       = df['Title'].fillna('').apply(self.clean_html)
        df['Body']        = df['Body'].fillna('').apply(self.clean_html)
        df['AnswerBody']  = df['AnswerBody'].fillna('').apply(self.clean_html)
        df['AnswerScore'] = pd.to_numeric(df['AnswerScore'], errors='coerce').fillna(0)
        df['Tags']        = df['Tags'].apply(self.extract_primary_tag)
        
        # Filter tags
        before = len(df)
        df = df[df['Tags'] != 'other'].reset_index(drop=True)
        logger.info(f"Baris dihapus (tag tidak valid): {before - len(df)}")
        
        df['combined_text'] = (
            df['Title'].astype(str) + ' ' +
            df['Title'].astype(str) + ' ' +
            df['Title'].astype(str) + ' ' +
            df['Body'].astype(str)
        )
        
        df = df.reset_index(drop=True)
        
        # Apply limit if specified (useful for quick verification)
        if limit is not None:
            df = df.head(limit).reset_index(drop=True)
            logger.info(f"Dataset dibatasi menjadi {limit} baris untuk pelatihan cepat.")
            
        logger.info(f"Total baris setelah dibersihkan: {len(df)}")
        
        # Save cleaned dataset
        os.makedirs(os.path.dirname(self.clean_dataset_path), exist_ok=True)
        df.to_csv(self.clean_dataset_path, index=False)
        logger.info(f"Dataset bersih berhasil disimpan ke: {self.clean_dataset_path}")

        # 2. Build and save TF-IDF Model
        logger.info("Membangun model TF-IDF...")
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])
        logger.info(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
        
        os.makedirs(os.path.dirname(self.tfidf_path), exist_ok=True)
        with open(self.tfidf_path, 'wb') as f:
            pickle.dump({'vectorizer': tfidf, 'matrix': tfidf_matrix}, f)
        logger.info(f"TF-IDF model disimpan ke: {self.tfidf_path}")

        # 3. Build Sentence-BERT Embeddings
        logger.info("Mengunduh/memuat model Sentence-BERT (all-MiniLM-L6-v2)...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Menghitung embedding SBERT untuk seluruh dataset (proses ini memerlukan waktu)...")
        
        embeddings = embedding_model.encode(
            df['combined_text'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64
        )
        logger.info(f"Embedding selesai. Matrix shape: {embeddings.shape}")

        # 4. Build FAISS Indices
        logger.info("Membangun Global FAISS Index...")
        dimension = embeddings.shape[1]
        embeddings_norm = embeddings.copy().astype('float32')
        faiss.normalize_L2(embeddings_norm)
        
        global_index = faiss.IndexFlatIP(dimension)
        global_index.add(embeddings_norm)
        
        os.makedirs(os.path.dirname(self.global_faiss_path), exist_ok=True)
        faiss.write_index(global_index, self.global_faiss_path)
        logger.info(f"Global FAISS Index disimpan ke: {self.global_faiss_path} ({global_index.ntotal} vektor)")

        # Local Indices for Small Tags
        logger.info("Membangun Local Per-Tag FAISS Index...")
        SMALL_TAG_THRESHOLD = 500
        serialized_local_indices = {}
        
        small_tags = [t for t in df['Tags'].unique() if len(df[df['Tags'] == t]) < SMALL_TAG_THRESHOLD]
        logger.info(f"Kategori tag kecil (jumlah data < {SMALL_TAG_THRESHOLD}): {small_tags}")
        
        for tag in small_tags:
            tag_mask = df['Tags'] == tag
            tag_global_idx = df[tag_mask].index.tolist()
            sub_emb = embeddings_norm[tag_global_idx].copy()
            
            local_idx = faiss.IndexFlatIP(dimension)
            local_idx.add(sub_emb)
            
            # Serialize index to numpy array to enable pickle storage
            serialized_idx = faiss.serialize_index(local_idx)
            serialized_local_indices[tag] = (serialized_idx, tag_global_idx)
            logger.info(f"  Indeks lokal dibuat untuk '{tag}' dengan {local_idx.ntotal} vektor.")
            
        with open(self.local_faiss_path, 'wb') as f:
            pickle.dump(serialized_local_indices, f)
        logger.info(f"Local Per-Tag FAISS Indices berhasil disimpan ke: {self.local_faiss_path}")
        logger.info("--- PROSES TRAINING DAN SERIALISASI SELESAI ---")

    def load_resources(self):
        """
        Fast loader for pre-built serialized files.
        """
        required_files = [
            self.clean_dataset_path,
            self.tfidf_path,
            self.global_faiss_path,
            self.local_faiss_path
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(
                f"File model berikut tidak ditemukan: {missing_files}\n"
                "Harap jalankan proses training terlebih dahulu: python Model/model.py train"
            )

        logger.info("Memuat sumber daya model dari disk...")
        
        # 1. Load clean dataset
        self.df = pd.read_csv(self.clean_dataset_path)
        
        # 2. Load TF-IDF vectorizer and matrix
        with open(self.tfidf_path, 'rb') as f:
            tfidf_data = pickle.load(f)
            self.tfidf = tfidf_data['vectorizer']
            self.tfidf_matrix = tfidf_data['matrix']
            
        # 3. Load global FAISS index
        self.index = faiss.read_index(self.global_faiss_path)
        
        # 4. Load local per-tag FAISS indices
        with open(self.local_faiss_path, 'rb') as f:
            serialized_local_indices = pickle.load(f)
            
        self.tag_local_indices = {}
        for tag, (serialized_idx, tag_global_idx) in serialized_local_indices.items():
            self.tag_local_indices[tag] = (faiss.deserialize_index(serialized_idx), tag_global_idx)
            
        # 5. Initialize SBERT & Translator
        self.translator = GoogleTranslator(source='auto', target='en')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Semua sumber daya model berhasil dimuat.")

    def translate_if_needed(self, text: str) -> str:
        try:
            result = self.translator.translate(text)
            return result if result else text
        except Exception as e:
            logger.warning(f'Translasi gagal: {e}. Menggunakan query asli.')
            return text

    def check_query_length(self, text: str, max_tokens: int = 256) -> str:
        words = text.split()
        estimated_tokens = int(len(words) * 1.4)
        if estimated_tokens > max_tokens:
            logger.warning(
                f'Query terlalu panjang (~{estimated_tokens} token estimasi, '
                f'batas SBERT={max_tokens}). '
                f'Query dipotong pada {max_tokens} token pertama.'
            )
            safe_words = max_tokens // 2
            text = ' '.join(words[:safe_words])
        return text

    def normalize_score(self, scores_array: np.ndarray) -> np.ndarray:
        mn, mx = scores_array.min(), scores_array.max()
        if mx == mn:
            return np.ones_like(scores_array)
        return (scores_array - mn) / (mx - mn)

    @lru_cache(maxsize=256)
    def _encode_query(self, query_text: str) -> np.ndarray:
        emb = self.embedding_model.encode([query_text], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(emb)
        return emb.copy()

    @lru_cache(maxsize=256)
    def _tfidf_transform(self, query_text: str):
        return self.tfidf.transform([query_text])

    def search_hybrid(
        self,
        query: str,
        tag_filter: str = None,
        top_k: int = 5,
        alpha: float = 0.4,
        beta: float = 0.5,
        gamma: float = 0.1
    ) -> list[dict]:
        """
        Executes hybrid lexical-semantic search with social quality prioritization.
        """
        if tag_filter:
            tag_filter_clean = tag_filter.strip().lower()

            if tag_filter_clean not in VALID_TAGS:
                logger.warning(
                    f'Tag "{tag_filter_clean}" tidak ditemukan di VALID_TAGS.\n'
                    f'Tag yang tersedia: {VALID_TAGS}'
                )
                return []

            mask = self.df['Tags'] == tag_filter_clean
            subset_df = self.df[mask].copy()
            subset_indices = self.df[mask].index.tolist()

            if len(subset_df) == 0:
                logger.warning(f'Tag "{tag_filter_clean}" terdaftar tapi tidak ada data yang cocok.')
                return []
        else:
            tag_filter_clean = None
            subset_df = self.df
            subset_indices = list(range(len(self.df)))

        # Translate Indonesian queries
        translated_query = self.translate_if_needed(query)
        translated_query = self.check_query_length(translated_query)

        logger.info(f'Query  (asli)      : {query[:80]}')
        logger.info(f'Query  (translated): {translated_query[:80]}')
        logger.info(f'Scope              : {len(subset_df)} rows (tag="{tag_filter_clean}")')

        # 1. Lexical Similarity (TF-IDF)
        query_vec = self._tfidf_transform(translated_query)
        tfidf_scores_all = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        tfidf_scores = tfidf_scores_all[subset_indices]

        # 2. Semantic Similarity (Sentence-BERT via FAISS)
        if tag_filter_clean and tag_filter_clean in self.tag_local_indices:
            # Query on small-tag local index
            local_faiss_idx, local_global_indices = self.tag_local_indices[tag_filter_clean]
            query_emb = self._encode_query(translated_query)
            k_local = min(local_faiss_idx.ntotal, top_k * 5)
            scores_local, indices_local = local_faiss_idx.search(query_emb, k_local)

            sbert_scores = np.zeros(len(subset_indices))
            for sc, loc_idx in zip(scores_local[0], indices_local[0]):
                if loc_idx >= 0:
                    global_idx = local_global_indices[loc_idx]
                    if global_idx in subset_indices:
                        pos = subset_indices.index(global_idx)
                        sbert_scores[pos] = float(sc)
        else:
            # Query on global index
            query_emb = self._encode_query(translated_query)
            k_search = min(len(self.df), top_k * 20)
            sbert_scores_all = np.zeros(len(self.df))
            scores_faiss, indices_faiss = self.index.search(query_emb, k_search)
            for sc, idx in zip(scores_faiss[0], indices_faiss[0]):
                if idx >= 0:
                    sbert_scores_all[idx] = float(sc)
            sbert_scores = sbert_scores_all[subset_indices]

        # 3. Community Score Prioritization (AnswerScore)
        answer_scores = subset_df['AnswerScore'].values.astype(float)
        answer_scores = np.log1p(np.clip(answer_scores, 0, None))

        # Normalization
        tfidf_norm  = self.normalize_score(tfidf_scores)
        sbert_norm  = self.normalize_score(sbert_scores)
        answer_norm = self.normalize_score(answer_scores)

        # Fusion Calculation
        fusion = alpha * tfidf_norm + beta * sbert_norm + gamma * answer_norm
        top_local_indices = fusion.argsort()[-top_k:][::-1]

        results = []
        for local_idx in top_local_indices:
            global_idx = subset_indices[local_idx]
            row = self.df.iloc[global_idx]
            results.append({
                'fusion_score':    float(fusion[local_idx]),
                'tfidf_score':     float(tfidf_norm[local_idx]),
                'sbert_score':     float(sbert_norm[local_idx]),
                'answer_score_raw': int(row['AnswerScore']),
                'title':           row['Title'],
                'tags':            row['Tags'],
                'answer':          row['AnswerBody'],
            })

        return results

    def evaluate_model(self, top_k: int = 5) -> dict:
        """
        Runs standardized test suite evaluation.
        """
        eval_queries = [
            {
                'query': 'how to connect python to mysql database',
                'tag': 'python',
                'expected_keywords': ['mysql', 'connect', 'database', 'pymysql', 'sqlalchemy', 'cursor']
            },
            {
                'query': 'difference between machine code and assembly language',
                'tag': 'assembly',
                'expected_keywords': ['assembly', 'machine', 'instruction', 'code', 'processor']
            },
            {
                'query': 'how to sort array in javascript',
                'tag': 'javascript',
                'expected_keywords': ['sort', 'array', 'javascript', 'function', 'method']
            },
            {
                'query': 'null pointer exception java',
                'tag': 'java',
                'expected_keywords': ['null', 'pointer', 'exception', 'object', 'reference']
            },
            {
                'query': 'how to read file in python',
                'tag': 'python',
                'expected_keywords': ['file', 'open', 'read', 'with', 'lines']
            },
        ]
        
        def compute_keyword_hit(answer_text: str, keywords: list) -> bool:
            answer_lower = answer_text.lower()
            return any(kw.lower() in answer_lower for kw in keywords)
            
        results_summary = []
        hits = 0

        print(f'\n{"="*70}')
        print(f'EVALUASI MODEL — {len(eval_queries)} query, top_k={top_k}')
        print(f'{"="*70}')

        for i, eq in enumerate(eval_queries, 1):
            results = self.search_hybrid(eq['query'], tag_filter=eq['tag'], top_k=top_k)

            if not results:
                hit = False
                best_score = 0.0
            else:
                combined_answers = ' '.join([r['answer'] + ' ' + r['title'] for r in results])
                hit = compute_keyword_hit(combined_answers, eq['expected_keywords'])
                best_score = results[0]['fusion_score'] if results else 0.0

            if hit:
                hits += 1

            status = 'HIT' if hit else 'MISS'
            print(f'\n[Q{i}] {status}')
            print(f'  Query   : {eq["query"]}')
            print(f'  Tag     : {eq["tag"]}')
            print(f'  Keywords: {eq["expected_keywords"]}')
            print(f'  Top hasil: {results[0]["title"] if results else "—"}')
            print(f'  Best fusion_score: {best_score:.4f}')

            results_summary.append({
                'query': eq['query'],
                'tag': eq['tag'],
                'hit': hit,
                'best_fusion_score': best_score,
                'num_results': len(results)
            })

        keyword_hit_rate = hits / len(eval_queries)
        avg_score = np.mean([r['best_fusion_score'] for r in results_summary])

        print(f'\n{"="*70}')
        print(f'RINGKASAN EVALUASI')
        print(f'{"="*70}')
        print(f'  Keyword Hit Rate @{top_k} : {keyword_hit_rate:.2%} ({hits}/{len(eval_queries)})')
        print(f'  Avg Fusion Score       : {avg_score:.4f}')
        print(f'{"="*70}\n')

        return {
            'keyword_hit_rate': keyword_hit_rate,
            'avg_fusion_score': avg_score,
            'top_k': top_k,
            'n_queries': len(eval_queries),
            'detail': results_summary
        }

def main():
    parser = argparse.ArgumentParser(description="Stack Overflow Q&A Search Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Perintah CLI")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Melatih model, membangun FAISS index, dan menyimpan file serialisasi")
    train_parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah baris dataset untuk latihan cepat/testing")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Mencari jawaban pemrograman secara hybrid")
    search_parser.add_argument("query", type=str, help="Kueri pencarian (bisa dalam Bahasa Indonesia atau Inggris)")
    search_parser.add_argument("--tag", type=str, default=None, help="Filter pencarian berdasarkan tag bahasa pemrograman tertentu")
    search_parser.add_argument("--top_k", type=int, default=5, help="Jumlah hasil pencarian yang dikembalikan (default: 5)")
    search_parser.add_argument("--alpha", type=float, default=0.4, help="Bobot skor TF-IDF (default: 0.4)")
    search_parser.add_argument("--beta", type=float, default=0.5, help="Bobot skor Sentence-BERT (default: 0.5)")
    search_parser.add_argument("--gamma", type=float, default=0.1, help="Bobot skor Answer Score (default: 0.1)")

    # Command: evaluate
    subparsers.add_parser("evaluate", help="Menjalankan pengujian evaluasi kuantitatif terhadap 5 query standar")

    args = parser.parse_args()

    engine = StackOverflowSearchEngine()

    if args.command == "train":
        engine.train_and_serialize(limit=args.limit)
    elif args.command == "search":
        try:
            engine.load_resources()
            results = engine.search_hybrid(
                query=args.query,
                tag_filter=args.tag,
                top_k=args.top_k,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma
            )
            
            if not results:
                print("\nTidak ada hasil yang cocok ditemukan.")
                return
                
            print('\n' + '='*80)
            print(f'HASIL PENCARIAN HYBRID  |  query="{args.query}"  |  tag="{args.tag or "Semua Tag"}"')
            print('='*80)
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['fusion_score']:.4f}] {r['title']}")
                print(f"   tfidf={r['tfidf_score']:.3f} | sbert={r['sbert_score']:.3f} | ans_score={r['answer_score_raw']}")
                print(f"   tags: {r['tags']}")
                print(f"   answer: {r['answer'][:350]}...")
                print('-'*80)
            print()
        except Exception as e:
            logger.error(f"Gagal melakukan pencarian: {e}")
            logger.error("Pastikan Anda sudah menjalankan training sebelumnya: python Model/model.py train")
    elif args.command == "evaluate":
        try:
            engine.load_resources()
            engine.evaluate_model()
        except Exception as e:
            logger.error(f"Gagal melakukan evaluasi: {e}")
            logger.error("Pastikan Anda sudah menjalankan training sebelumnya: python Model/model.py train")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
