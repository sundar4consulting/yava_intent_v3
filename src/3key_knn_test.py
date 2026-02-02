#!/usr/bin/env python3
"""
3-Key KNN Test Script
Tests the 3-key KNN search against training utterances from the intent list.
Uses local embeddings file as the lookup data source.

Data Sources:
- Training utterances: esSearchintentList_47_keyword.json
- Embeddings lookup: esSearchintentList_47_3key_embeddings.json
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import csv
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(__file__)
TRAINING_DATA_PATH = os.path.join(BASE_DIR, '../esSearchintentList_47 _keyword.json')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'esSearchintentList_47_3key_embeddings.json')
MODEL_PATH = os.path.join(BASE_DIR, '../../Transformer')
RESULTS_CSV_PATH = os.path.join(BASE_DIR, '3key_knn_test_results.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, '3key_knn_test_summary.txt')


class ThreeKeyKNNTester:
    """
    Tests 3-Key KNN search using local embeddings data.
    Evaluates accuracy by matching training utterances to their expected intents.
    """
    
    def __init__(self):
        print("🔧 Initializing 3-Key KNN Tester...")
        
        # Load model
        print(f"📦 Loading model from: {MODEL_PATH}")
        self.model = SentenceTransformer(MODEL_PATH)
        
        # Load training data (source of test utterances)
        print(f"📄 Loading training data from: {TRAINING_DATA_PATH}")
        with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            self.training_data = json.load(f)
        
        # Load embeddings data (lookup source)
        print(f"📄 Loading embeddings from: {EMBEDDINGS_PATH}")
        with open(EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
            self.embedding_data = json.load(f)
        
        # Prepare embedding matrices for efficient search
        self._prepare_embedding_matrices()
        
        print(f"✅ Loaded {len(self.training_data)} intents with training data")
        print(f"✅ Loaded {len(self.embedding_data)} intent embeddings for lookup")
        
    def _prepare_embedding_matrices(self):
        """Prepare numpy matrices for efficient cosine similarity search."""
        self.intent_embs = np.array([item['intent_embedding'] for item in self.embedding_data])
        self.desc_embs = np.array([item['description_embedding'] for item in self.embedding_data])
        self.example_embs = np.array([item['example_embedding'] for item in self.embedding_data])
        
        # Normalize for cosine similarity
        self.intent_embs_norm = self.intent_embs / np.linalg.norm(self.intent_embs, axis=1, keepdims=True)
        self.desc_embs_norm = self.desc_embs / np.linalg.norm(self.desc_embs, axis=1, keepdims=True)
        self.example_embs_norm = self.example_embs / np.linalg.norm(self.example_embs, axis=1, keepdims=True)
        
        # Store metadata for results
        self.metadata = [
            {
                'intent_name': item['intent_name'],
                'description_short': item['description_short'],
                'example_utterance': item['example_utterance'],
                'category': item.get('category', '')
            }
            for item in self.embedding_data
        ]
    
    def knn_search_3key(
        self, 
        query: str, 
        k: int = 1,
        return_all_matches: bool = False
    ) -> Dict:
        """
        Perform 3-key KNN search using cosine similarity.
        
        Args:
            query: The utterance to search for
            k: Number of top results to consider per key
            return_all_matches: If True, return matches from all 3 keys
            
        Returns:
            Best matching result with score and matched key
        """
        # Encode query
        query_vec = self.model.encode(query)
        query_norm = query_vec / np.linalg.norm(query_vec)
        
        # Search each embedding key
        results = []
        
        # Intent embedding search
        intent_scores = np.dot(self.intent_embs_norm, query_norm)
        top_intent_idx = int(np.argmax(intent_scores))
        results.append({
            'score': float(intent_scores[top_intent_idx]),
            'intent_name': self.metadata[top_intent_idx]['intent_name'],
            'category': self.metadata[top_intent_idx]['category'],
            'matched_key': 'intent_embedding',
            'index': top_intent_idx
        })
        
        # Description embedding search
        desc_scores = np.dot(self.desc_embs_norm, query_norm)
        top_desc_idx = int(np.argmax(desc_scores))
        results.append({
            'score': float(desc_scores[top_desc_idx]),
            'intent_name': self.metadata[top_desc_idx]['intent_name'],
            'category': self.metadata[top_desc_idx]['category'],
            'matched_key': 'description_embedding',
            'index': top_desc_idx
        })
        
        # Example embedding search
        example_scores = np.dot(self.example_embs_norm, query_norm)
        top_example_idx = int(np.argmax(example_scores))
        results.append({
            'score': float(example_scores[top_example_idx]),
            'intent_name': self.metadata[top_example_idx]['intent_name'],
            'category': self.metadata[top_example_idx]['category'],
            'matched_key': 'example_embedding',
            'index': top_example_idx
        })
        
        if return_all_matches:
            return {
                'all_matches': results,
                'best_match': max(results, key=lambda x: x['score'])
            }
        
        # Return best match across all keys
        return max(results, key=lambda x: x['score'])
    
    def knn_search_top_k(
        self, 
        query: str, 
        k: int = 3
    ) -> List[Dict]:
        """
        Return top-k matches across all 3 keys combined.
        """
        query_vec = self.model.encode(query)
        query_norm = query_vec / np.linalg.norm(query_vec)
        
        all_matches = {}
        
        # Search each key
        for key, emb_norm in [
            ('intent_embedding', self.intent_embs_norm),
            ('description_embedding', self.desc_embs_norm),
            ('example_embedding', self.example_embs_norm)
        ]:
            scores = np.dot(emb_norm, query_norm)
            top_indices = np.argsort(scores)[::-1][:k]
            
            for idx in top_indices:
                intent_name = self.metadata[idx]['intent_name']
                score = float(scores[idx])
                
                if intent_name not in all_matches or score > all_matches[intent_name]['score']:
                    all_matches[intent_name] = {
                        'score': score,
                        'intent_name': intent_name,
                        'category': self.metadata[idx]['category'],
                        'matched_key': key
                    }
        
        # Sort by score and return top k
        sorted_matches = sorted(all_matches.values(), key=lambda x: x['score'], reverse=True)
        return sorted_matches[:k]
    
    def run_test(self, verbose: bool = True) -> Dict:
        """
        Run the full test suite against all training utterances.
        
        Returns:
            Dictionary with test results and metrics
        """
        print("\n" + "="*70)
        print("🧪 Starting 3-Key KNN Test")
        print("="*70 + "\n")
        
        results = []
        total_utterances = 0
        correct_matches = 0
        correct_top3_matches = 0
        
        # Track metrics by intent and category
        intent_metrics = defaultdict(lambda: {'total': 0, 'correct': 0})
        category_metrics = defaultdict(lambda: {'total': 0, 'correct': 0})
        key_match_counts = defaultdict(int)
        
        # Process each intent's training utterances
        for intent in self.training_data:
            expected_intent = intent.get('intent_name', '')
            expected_category = intent.get('category', '')
            training_utts = intent.get('training_utterances', [])
            
            if verbose:
                print(f"\n📋 Testing intent: {expected_intent}")
                print(f"   Utterances: {len(training_utts)}")
            
            for utt in training_utts:
                total_utterances += 1
                
                # Get best match
                best_match = self.knn_search_3key(utt)
                found_intent = best_match['intent_name']
                found_category = best_match['category']
                matched_key = best_match['matched_key']
                score = best_match['score']
                
                # Get top-3 matches for top-k accuracy
                top3_matches = self.knn_search_top_k(utt, k=3)
                top3_intents = [m['intent_name'] for m in top3_matches]
                
                is_correct = (found_intent == expected_intent)
                is_top3_correct = (expected_intent in top3_intents)
                
                if is_correct:
                    correct_matches += 1
                    intent_metrics[expected_intent]['correct'] += 1
                    category_metrics[expected_category]['correct'] += 1
                    
                if is_top3_correct:
                    correct_top3_matches += 1
                
                intent_metrics[expected_intent]['total'] += 1
                category_metrics[expected_category]['total'] += 1
                key_match_counts[matched_key] += 1
                
                results.append({
                    'utterance': utt,
                    'expected_intent': expected_intent,
                    'found_intent': found_intent,
                    'expected_category': expected_category,
                    'found_category': found_category,
                    'matched_key': matched_key,
                    'score': round(score, 4),
                    'is_correct': is_correct,
                    'is_top3_correct': is_top3_correct,
                    'top3_intents': '|'.join(top3_intents)
                })
        
        # Calculate metrics
        accuracy = (correct_matches / total_utterances * 100) if total_utterances > 0 else 0
        top3_accuracy = (correct_top3_matches / total_utterances * 100) if total_utterances > 0 else 0
        
        summary = {
            'total_utterances': total_utterances,
            'correct_matches': correct_matches,
            'top3_correct_matches': correct_top3_matches,
            'accuracy': round(accuracy, 2),
            'top3_accuracy': round(top3_accuracy, 2),
            'intent_metrics': dict(intent_metrics),
            'category_metrics': dict(category_metrics),
            'key_match_distribution': dict(key_match_counts),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'results': results,
            'summary': summary
        }
    
    def save_results(self, test_results: Dict, csv_path: str = RESULTS_CSV_PATH, summary_path: str = SUMMARY_PATH):
        """Save test results to CSV and summary to text file."""
        
        # Save detailed results to CSV
        print(f"\n📊 Saving detailed results to: {csv_path}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'utterance', 'expected_intent', 'found_intent', 
                'expected_category', 'found_category', 'matched_key',
                'score', 'is_correct', 'is_top3_correct', 'top3_intents'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in test_results['results']:
                writer.writerow(row)
        
        # Save summary to text file
        print(f"📊 Saving summary to: {summary_path}")
        summary = test_results['summary']
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("3-Key KNN Test Summary\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write("="*70 + "\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Utterances Tested: {summary['total_utterances']}\n")
            f.write(f"Correct Matches (Top-1): {summary['correct_matches']}\n")
            f.write(f"Accuracy (Top-1): {summary['accuracy']}%\n")
            f.write(f"Correct Matches (Top-3): {summary['top3_correct_matches']}\n")
            f.write(f"Accuracy (Top-3): {summary['top3_accuracy']}%\n\n")
            
            f.write("KEY MATCH DISTRIBUTION\n")
            f.write("-"*40 + "\n")
            for key, count in sorted(summary['key_match_distribution'].items()):
                pct = count / summary['total_utterances'] * 100
                f.write(f"  {key}: {count} ({pct:.1f}%)\n")
            f.write("\n")
            
            f.write("ACCURACY BY CATEGORY\n")
            f.write("-"*40 + "\n")
            for category, metrics in sorted(summary['category_metrics'].items()):
                cat_acc = metrics['correct'] / metrics['total'] * 100 if metrics['total'] > 0 else 0
                f.write(f"  {category}: {metrics['correct']}/{metrics['total']} ({cat_acc:.1f}%)\n")
            f.write("\n")
            
            f.write("ACCURACY BY INTENT\n")
            f.write("-"*40 + "\n")
            # Sort by accuracy descending
            intent_accs = []
            for intent, metrics in summary['intent_metrics'].items():
                acc = metrics['correct'] / metrics['total'] * 100 if metrics['total'] > 0 else 0
                intent_accs.append((intent, metrics, acc))
            
            for intent, metrics, acc in sorted(intent_accs, key=lambda x: x[2], reverse=True):
                f.write(f"  {intent}: {metrics['correct']}/{metrics['total']} ({acc:.1f}%)\n")
        
        print(f"✅ Results saved successfully!")
    
    def print_summary(self, summary: Dict):
        """Print a formatted summary to console."""
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        print(f"\n🎯 Overall Accuracy:")
        print(f"   Top-1 Accuracy: {summary['accuracy']}% ({summary['correct_matches']}/{summary['total_utterances']})")
        print(f"   Top-3 Accuracy: {summary['top3_accuracy']}% ({summary['top3_correct_matches']}/{summary['total_utterances']})")
        
        print(f"\n🔑 Key Match Distribution:")
        for key, count in sorted(summary['key_match_distribution'].items()):
            pct = count / summary['total_utterances'] * 100
            print(f"   {key}: {count} ({pct:.1f}%)")
        
        print(f"\n📁 Accuracy by Category:")
        for category, metrics in sorted(summary['category_metrics'].items()):
            cat_acc = metrics['correct'] / metrics['total'] * 100 if metrics['total'] > 0 else 0
            print(f"   {category}: {metrics['correct']}/{metrics['total']} ({cat_acc:.1f}%)")
        
        # Show worst performing intents
        print(f"\n⚠️  Lowest Accuracy Intents (if any < 100%):")
        intent_accs = []
        for intent, metrics in summary['intent_metrics'].items():
            acc = metrics['correct'] / metrics['total'] * 100 if metrics['total'] > 0 else 0
            if acc < 100:
                intent_accs.append((intent, metrics, acc))
        
        for intent, metrics, acc in sorted(intent_accs, key=lambda x: x[2])[:10]:
            print(f"   {intent}: {metrics['correct']}/{metrics['total']} ({acc:.1f}%)")
        
        if not intent_accs:
            print("   None - all intents at 100% accuracy! 🎉")
    
    def get_mismatches(self, test_results: Dict) -> List[Dict]:
        """Get all mismatched results for analysis."""
        return [r for r in test_results['results'] if not r['is_correct']]


def main():
    """Main entry point for the test script."""
    print("\n" + "="*70)
    print("🚀 3-Key KNN Test Runner")
    print("="*70)
    
    # Initialize tester
    tester = ThreeKeyKNNTester()
    
    # Run tests
    test_results = tester.run_test(verbose=False)
    
    # Print summary
    tester.print_summary(test_results['summary'])
    
    # Save results
    tester.save_results(test_results)
    
    # Show some mismatches if any
    mismatches = tester.get_mismatches(test_results)
    if mismatches:
        print(f"\n⚠️  Sample Mismatches (first 5):")
        print("-"*70)
        for m in mismatches[:5]:
            print(f"   Utterance: \"{m['utterance'][:50]}...\"")
            print(f"   Expected: {m['expected_intent']}")
            print(f"   Found: {m['found_intent']} (score: {m['score']}, key: {m['matched_key']})")
            print(f"   Top-3: {m['top3_intents']}")
            print()
    
    print("\n✅ Test completed!")
    return test_results


if __name__ == "__main__":
    main()
