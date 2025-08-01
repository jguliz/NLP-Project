import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from collections import defaultdict

class SemanticOpinionSearch:
    def __init__(self, preprocessed_data_path='preprocessed_data.pkl', model_name='all-MiniLM-L6-v2'):
        """Initialize semantic search with sentence embeddings"""
        print("Loading preprocessed data...")
        with open(preprocessed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.df = data['df']
        self.inverted_index = data['inverted_index']
        
        # Load sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Opinion lexicons
        self.positive_words = self.load_positive_lexicon()
        self.negative_words = self.load_negative_lexicon()
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        print("Semantic search engine initialized!")
    
    def load_positive_lexicon(self):
        """Load positive opinion words"""
        return set([
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'perfect', 'love', 'best', 'awesome', 'nice', 'super', 'brilliant',
            'outstanding', 'superior', 'exceptional', 'magnificent', 'terrific',
            'useful', 'helpful', 'strong', 'sharp', 'clear', 'fast', 'reliable',
            'solid', 'sturdy', 'efficient', 'effective', 'convenient', 'comfortable'
        ])
    
    def load_negative_lexicon(self):
        """Load negative opinion words"""
        return set([
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointing', 'useless', 'broken', 'defective', 'weak', 'slow',
            'unreliable', 'uncomfortable', 'inconvenient', 'difficult', 'hard',
            'problem', 'issue', 'fail', 'failed', 'failure', 'error', 'wrong'
        ])
    
    def parse_query(self, query):
        """Parse query in format 'aspect1 aspect2:opinion1 [opinion2]'"""
        parts = query.lower().split(':')
        if len(parts) != 2:
            raise ValueError("Query must be in format 'aspect:opinion'")
        
        aspect_terms = parts[0].strip().split()
        opinion_terms = parts[1].strip().split()
        
        return aspect_terms, opinion_terms
    
    def extract_sentences_with_aspect(self, text, aspect_terms):
        """Extract sentences containing aspect terms"""
        if pd.isna(text):
            return []
        
        sentences = re.split(r'[.!?]+', str(text).lower())
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(aspect in sentence.split() for aspect in aspect_terms):
                relevant_sentences.append(sentence)
        
        return relevant_sentences
    
    def get_sentence_embedding(self, sentence):
        """Get embedding for a sentence with caching"""
        if sentence in self.embedding_cache:
            return self.embedding_cache[sentence]
        
        embedding = self.model.encode([sentence])[0]
        self.embedding_cache[sentence] = embedding
        return embedding
    
    def compute_semantic_similarity(self, query_text, candidate_text):
        """Compute semantic similarity between query and candidate"""
        query_emb = self.get_sentence_embedding(query_text)
        candidate_emb = self.get_sentence_embedding(candidate_text)
        
        similarity = cosine_similarity([query_emb], [candidate_emb])[0][0]
        return similarity
    
    def analyze_opinion_connotation(self, sentence, opinion_terms):
        """Analyze if sentence expresses the intended opinion"""
        words = sentence.split()
        
        # Check for negation patterns
        negation_words = {'not', 'no', 'never', 'neither', 'none', 'nobody', 
                         'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'}
        
        # Find opinion word positions
        opinion_positions = []
        for i, word in enumerate(words):
            if word in opinion_terms or word in self.positive_words or word in self.negative_words:
                opinion_positions.append(i)
        
        # Check for negations near opinion words
        is_negated = False
        for pos in opinion_positions:
            # Check 3 words before the opinion word for negation
            for j in range(max(0, pos-3), pos):
                if words[j] in negation_words:
                    is_negated = True
                    break
        
        # Determine the intended polarity
        intended_positive = any(term in self.positive_words for term in opinion_terms)
        intended_negative = any(term in self.negative_words for term in opinion_terms)
        
        # Count positive and negative words in sentence
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        # Apply negation logic
        if is_negated:
            pos_count, neg_count = neg_count, pos_count
        
        # Check if sentence matches intended opinion
        if intended_positive and pos_count > neg_count:
            return True
        elif intended_negative and neg_count > pos_count:
            return True
        elif not intended_positive and not intended_negative:
            # Neutral query - accept any opinion
            return True
        
        return False
    
    def search_semantic_connotation(self, query, top_k=100):
        """Advanced semantic search with proper connotation handling"""
        aspect_terms, opinion_terms = self.parse_query(query)
        
        # Create query representation
        query_text = f"{' '.join(aspect_terms)} {' '.join(opinion_terms)}"
        
        # Get candidate reviews containing aspect terms
        aspect_reviews = set()
        for term in aspect_terms:
            if term in self.inverted_index:
                aspect_reviews.update(self.inverted_index[term])
        
        # Score each candidate review
        scored_reviews = []
        
        for review_id in aspect_reviews:
            review = self.df[self.df['review_id'] == review_id].iloc[0]
            review_text = review.get('review_text', '')
            
            # Extract sentences with aspects
            relevant_sentences = self.extract_sentences_with_aspect(review_text, aspect_terms)
            
            if not relevant_sentences:
                continue
            
            # Score each sentence
            best_score = 0
            best_sentence = ""
            
            for sentence in relevant_sentences:
                # Check if sentence has proper connotation
                if not self.analyze_opinion_connotation(sentence, opinion_terms):
                    continue
                
                # Compute semantic similarity
                similarity = self.compute_semantic_similarity(query_text, sentence)
                
                # Boost score if opinion terms are present
                opinion_boost = sum(0.2 for term in opinion_terms if term in sentence.split())
                
                # Proximity boost - aspect and opinion in same sentence
                proximity_boost = 0.3 if any(op in sentence for op in opinion_terms) else 0
                
                # Rating consistency boost
                rating = review['star_rating']
                intended_positive = any(term in self.positive_words for term in opinion_terms)
                intended_negative = any(term in self.negative_words for term in opinion_terms)
                
                rating_boost = 0
                if intended_positive and rating > 3:
                    rating_boost = 0.2
                elif intended_negative and rating <= 3:
                    rating_boost = 0.2
                
                total_score = similarity + opinion_boost + proximity_boost + rating_boost
                
                if total_score > best_score:
                    best_score = total_score
                    best_sentence = sentence
            
            if best_score > 0:
                scored_reviews.append((review_id, best_score, best_sentence))
        
        # Sort by score and return top results
        scored_reviews.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return [review_id for review_id, score, _ in scored_reviews[:top_k]]
    
    def search_test4(self, query):
        """Test 4: Proper connotation using semantic search"""
        return self.search_semantic_connotation(query, top_k=100)
    
    def save_results(self, results, output_file):
        """Save results to text file"""
        with open(output_file, 'w') as f:
            for review_id in results:
                f.write(f"{review_id}\n")
        print(f"Saved {len(results)} results to {output_file}")

# Usage
if __name__ == "__main__":
    searcher = SemanticOpinionSearch()
    
    test_queries = [
        "audio quality:poor",
        "wifi signal:strong", 
        "mouse button:click problem",
        "gps map:useful",
        "image quality:sharp"
    ]
    
    import os
    os.makedirs("outputs/advanced_models", exist_ok=True)
    
    for query in test_queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        
        # Run Method 2 search
        results = searcher.search_test4(query)
        searcher.save_results(results, f"outputs/advanced_models/{query_name}_test4.txt")
        
        print(f"Query: {query} - Found {len(results)} results")