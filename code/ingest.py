import os
import json
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Ingestor")

def ingest_data(source_dir: str = "source_corpus", output_dir: str = "data", model_name: str = "all-MiniLM-L6-v2"):
    """
    Crawls the source_corpus, chunks text, generates embeddings, and saves the FAISS index.
    """
    logger.info(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    chunks = []
    embeddings = []

    logger.info(f"Crawling directory: {source_dir}...")
    
    # Supported companies based on folder names
    for company_folder in os.listdir(source_dir):
        company_path = os.path.join(source_dir, company_folder)
        if not os.path.isdir(company_path):
            continue
            
        company_name = company_folder.capitalize()
        logger.info(f"Processing documents for: {company_name}")

        for root, _, files in os.walk(company_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                        # Chunking Strategy: Sliding window
                        # 500 characters with 50 character overlap
                        chunk_size = 500
                        overlap = 50
                        
                        for i in range(0, len(content), chunk_size - overlap):
                            chunk_text = content[i : i + chunk_size].strip()
                            if len(chunk_text) < 50:  # Skip tiny fragments
                                continue
                                
                            chunks.append({
                                "company": company_name,
                                "text": chunk_text,
                                "source": file_path
                            })
                            
                            # Generate Embedding
                            embedding = model.encode(chunk_text)
                            embeddings.append(embedding)
                            
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")

    if not embeddings:
        logger.error("No data found to ingest!")
        return

    # Create FAISS Index
    logger.info(f"Creating FAISS index with {len(embeddings)} chunks...")
    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    faiss_path = os.path.join(output_dir, "index.faiss")
    chunks_path = os.path.join(output_dir, "chunks.json")
    
    faiss.write_index(index, faiss_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info(f"Successfully saved FAISS index to {faiss_path}")
    logger.info(f"Successfully saved chunks to {chunks_path}")
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
