import os
import warnings
import requests
import pandas as pd
from io import BytesIO, StringIO
from typing import List
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)


class PolicyDataLoader:
    def __init__(self, config):
        self.config = config
        self.pdf_path = config.POLICY_PDF_PATH
        self.excel_url = config.EXCEL_URL

        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.vectorstore = None
        self.retriever = None
        self.all_chunks = []
        self.chunk_embeddings = None

    def _load_pdf_documents(self) -> List[Document]:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        # Use different chunking strategies for different content types
        table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Larger chunks for tables
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Larger chunks for better context
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        for page in pages:
            # Check if page contains tables or structured data
            if "│" in page.page_content or "TABLE" in page.page_content.upper() or "Grade" in page.page_content:
                # Keep tables and grade information together
                chunks.append(page)
            else:
                # Split regular text with larger chunks
                chunks.extend(text_splitter.split_documents([page]))

        print(f"📄 PDF Chunks Created: {len(chunks)}")
        
        # Debug: Show some chunk previews
        print("\n🔍 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1}: {chunk.page_content[:200]}...")
        
        return chunks

    def _convert_google_sheets_url(self, url: str) -> str:
        """Convert Google Sheets sharing URL to CSV download URL (most reliable)"""
        try:
            if "docs.google.com/spreadsheets" in url:
                if "/d/" in url:
                    sheet_id = url.split("/d/")[1].split("/")[0]
                    # Use CSV format - more reliable than Excel
                    download_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                    print(f"🔄 Converted to CSV download format")
                    return download_url
            return url
        except Exception as e:
            print(f"⚠️ URL conversion failed: {str(e)}")
            return url

    def _detect_content_type(self, content: bytes) -> str:
        """Detect if content is CSV, Excel, or HTML"""
        try:
            # Try to decode as text first
            text_content = content.decode('utf-8', errors='ignore')
            
            # Check if it's HTML (error page)
            if '<html' in text_content.lower() or '<!doctype html' in text_content.lower():
                return 'html'
            
            # Check if it's CSV (contains commas and newlines)
            if ',' in text_content and '\n' in text_content:
                lines = text_content.split('\n')
                if len(lines) > 1:
                    # Check if first few lines have consistent comma count
                    comma_counts = [line.count(',') for line in lines[:3] if line.strip()]
                    if comma_counts and all(count == comma_counts[0] for count in comma_counts):
                        return 'csv'
            
            # Check for Excel magic bytes
            if content.startswith(b'PK'):  # ZIP-based formats like .xlsx
                return 'xlsx'
            elif content.startswith(b'\xd0\xcf\x11\xe0'):  # OLE2 format like .xls
                return 'xls'
                
            return 'unknown'
            
        except Exception as e:
            print(f"⚠️ Content type detection failed: {str(e)}")
            return 'unknown'

    def _load_excel_documents(self) -> List[Document]:
        print("📥 Downloading data file...")
        
        # Convert Google Sheets URL to CSV format (most reliable)
        download_url = self._convert_google_sheets_url(self.excel_url)
        print(f"🌐 Using URL: {download_url}")
        
        try:
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,text/csv,*/*'
            }
            
            # Download with timeout and error handling
            print("⏳ Downloading file...")
            response = requests.get(download_url, headers=headers, timeout=30)
            
            print(f"📊 Response status: {response.status_code}")
            print(f"📊 Content type: {response.headers.get('content-type', 'unknown')}")
            print(f"📊 Content length: {len(response.content)} bytes")
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.reason}")

            # Detect actual content type
            content_type = self._detect_content_type(response.content)
            print(f"🔍 Detected content type: {content_type}")
            
            # Handle different content types
            if content_type == 'html':
                # This is likely an error page or access denied
                text_content = response.content.decode('utf-8', errors='ignore')
                if 'access denied' in text_content.lower() or 'permission' in text_content.lower():
                    raise Exception("❌ Access denied. Please make the Google Sheet publicly accessible.")
                else:
                    raise Exception("❌ Received HTML instead of data file. Check the URL and permissions.")
            
            elif content_type == 'csv':
                print("📄 Processing as CSV file...")
                # Handle CSV content
                text_content = response.content.decode('utf-8-sig', errors='ignore')  # Handle BOM
                df = pd.read_csv(StringIO(text_content))
                dfs = {'Sheet1': df}
                
            elif content_type in ['xlsx', 'xls']:
                print(f"📊 Processing as {content_type.upper()} file...")
                # Handle Excel content
                excel_file = BytesIO(response.content)
                
                if content_type == 'xlsx':
                    dfs = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
                else:
                    dfs = pd.read_excel(excel_file, sheet_name=None, engine='xlrd')
                    
            else:
                # Unknown format - try multiple approaches
                print("❓ Unknown format, trying multiple parsers...")
                
                # Try CSV first (most common for Google Sheets)
                try:
                    text_content = response.content.decode('utf-8-sig', errors='ignore')
                    df = pd.read_csv(StringIO(text_content))
                    dfs = {'Sheet1': df}
                    print("✅ Successfully parsed as CSV")
                except Exception as csv_error:
                    print(f"⚠️ CSV parsing failed: {csv_error}")
                    
                    # Try Excel formats
                    excel_file = BytesIO(response.content)
                    try:
                        dfs = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
                        print("✅ Successfully parsed as XLSX")
                    except Exception as xlsx_error:
                        print(f"⚠️ XLSX parsing failed: {xlsx_error}")
                        try:
                            excel_file.seek(0)
                            dfs = pd.read_excel(excel_file, sheet_name=None, engine='xlrd')
                            print("✅ Successfully parsed as XLS")
                        except Exception as xls_error:
                            print(f"⚠️ XLS parsing failed: {xls_error}")
                            raise Exception(f"❌ Could not parse file in any format. CSV: {csv_error}, XLSX: {xlsx_error}, XLS: {xls_error}")

            # Process the dataframes
            documents = []
            for name, df in dfs.items():
                if not df.empty:
                    print(f"📊 Processing sheet: {name}")
                    print(f"   - Shape: {df.shape}")
                    print(f"   - Columns: {list(df.columns)}")
                    
                    # Clean the dataframe
                    df = df.dropna(how='all')  # Remove completely empty rows
                    df = df.fillna('')  # Fill NaN with empty strings
                    
                    # Show sample data for debugging
                    if len(df) > 0:
                        print(f"   - Sample data:")
                        print(df.head(2).to_string(index=False))
                    
                    # Convert to string format that preserves structure
                    content = df.to_string(index=False)
                    documents.append(Document(
                        page_content=content, 
                        metadata={"source": f"sheet:{name}", "rows": len(df), "columns": list(df.columns)}
                    ))

            if not documents:
                raise Exception("❌ No valid data sheets found")

            print(f"✅ Successfully loaded {len(documents)} data sheets")
            return documents

        except requests.exceptions.Timeout:
            raise Exception("❌ Download timeout. Please check your internet connection.")
        except requests.exceptions.ConnectionError:
            raise Exception("❌ Connection error. Please check your internet connection.")
        except Exception as e:
            error_msg = str(e)
            if "Access denied" in error_msg or "403" in error_msg:
                raise Exception("❌ Access denied. Please ensure the Google Sheet is publicly accessible:\n"
                              "1. Open your Google Sheet\n"
                              "2. Click 'Share' button\n"
                              "3. Change access to 'Anyone with the link can view'\n"
                              "4. Copy the sharing URL to your config")
            elif "engine manually" in error_msg:
                raise Exception("❌ Excel parsing failed. This might be due to:\n"
                              "1. Missing required packages (openpyxl, xlrd)\n"
                              "2. Corrupted download\n"
                              "3. Invalid file format\n"
                              "Try using CSV format instead.")
            else:
                raise Exception(f"❌ Data loading failed: {error_msg}")

    def _create_fallback_user_data(self) -> List[Document]:
        """Create fallback user data if download fails"""
        print("🔄 Creating fallback user database...")
        
        # Sample user data
        sample_data = """username,password,grade,gender,remaining_leaves,total_leaves
john_doe,password123,L3,male,15,30
jane_smith,password456,L4,female,20,35
admin_user,admin123,L5,male,25,40
test_user,test123,L2,female,12,25"""
        
        df = pd.read_csv(StringIO(sample_data))
        content = df.to_string(index=False)
        
        document = Document(
            page_content=content,
            metadata={"source": "sheet:fallback_users", "rows": len(df), "columns": list(df.columns)}
        )
        
        print("⚠️ Using fallback user data. Please fix data source URL for actual user database.")
        return [document]

    def load_all_documents(self):
        try:
            pdf_docs = self._load_pdf_documents()
            
            # Try to load data, use fallback if it fails
            try:
                excel_docs = self._load_excel_documents()
            except Exception as excel_error:
                print(f"⚠️ Data loading failed: {str(excel_error)}")
                print("🔄 Using fallback user data...")
                excel_docs = self._create_fallback_user_data()

            all_docs = pdf_docs + excel_docs
            self.all_chunks = all_docs

            total = len(all_docs)
            print(f"\n📦 Total Combined Chunks: {total}")

            # Preview chunks
            def preview_chunks(docs, label="Chunk"):
                indices = list(range(min(3, total))) + list(range(max(total - 3, 3), total))
                print("\n🔍 Preview of Document Chunks:")
                for i in indices:
                    print(f"\n{label} #{i + 1}/{total}:\n{'-' * 40}")
                    content = docs[i].page_content.strip()
                    source = docs[i].metadata.get('source', 'unknown')
                    print(f"Source: {source}")
                    print(content[:300] + ("..." if len(content) > 300 else ""))

            preview_chunks(all_docs)

            # Create embeddings
            texts = [doc.page_content for doc in all_docs]
            self.chunk_embeddings = self.embedder.encode(texts)
            print(f"\n🧠 Embedding Matrix Shape: {self.chunk_embeddings.shape}")

            # Create vectorstore
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            self.vectorstore = FAISS.from_documents(all_docs, embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

            print("✅ Vectorstore & retriever created.\n")

        except Exception as e:
            raise RuntimeError(f"❌ Document load failed: {str(e)}")

    def get_retriever(self):
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call load_all_documents() first.")
        return self.retriever

    def rerank_chunks(self, query: str, retrieved_chunks: List[Document], top_k: int = 3) -> List[Document]:
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks to rerank.")
            return []

        query_vec = self.embedder.encode([query])
        chunk_texts = [chunk.page_content for chunk in retrieved_chunks]
        chunk_vecs = self.embedder.encode(chunk_texts)
        scores = cosine_similarity(query_vec, chunk_vecs)[0]
        ranked = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)

        print(f"\n🔍 Top {top_k} Reranked Chunks for Query: '{query}'\n{'='*60}")
        for i, (score, doc) in enumerate(ranked[:top_k], 1):
            content_preview = doc.page_content.strip().replace("\n", " ")[:300]
            print(f"\n🔹 Rank #{i} | Similarity: {score:.4f}\n{'-'*40}\n{content_preview}...\n")

        return [doc for score, doc in ranked[:top_k]]

    def test_data_connection(self) -> bool:
        """Test if data URL is accessible"""
        print("🧪 Testing data connection...")
        try:
            download_url = self._convert_google_sheets_url(self.excel_url)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.head(download_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                print("✅ Data URL is accessible")
                return True
            else:
                print(f"❌ Data URL returned status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Data connection test failed: {str(e)}")
            return False