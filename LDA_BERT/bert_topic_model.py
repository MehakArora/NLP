"""
Script that contains the BERTopic class.
"""
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class BERTopicExplorer:
    def __init__(self, documents):
        """
        Initialize the BERTopic exploration framework
        
        Args:
            documents (list): List of text documents to cluster
        """
        self.documents = documents
        self.results = []
        
    def _create_embeddings(self, model_name='all-MiniLM-L6-v2'):
        """
        Create embeddings using different sentence transformers
        
        Args:
            model_name (str): Name of the sentence transformer model
        
        Returns:
            numpy array: Document embeddings
        """
        embedding_models = [
            'all-MiniLM-L6-v2',
            'paraphrase-MiniLM-L3-v2',
            'all-mpnet-base-v2'
        ]
        
        embeddings_results = {}
        for model in embedding_models:
            model = SentenceTransformer(model)
            embeddings = model.encode(self.documents)
            embeddings_results[model] = embeddings
        
        return embeddings_results
    
    def _dimensionality_reduction(self, embeddings):
        """
        Apply different dimensionality reduction techniques
        
        Args:
            embeddings (numpy array): Input embeddings
        
        Returns:
            dict: Dimensionality reduced embeddings
        """
        reduction_techniques = {
            'UMAP': umap.UMAP(n_components=5, random_state=42),
            'PCA': StandardScaler().fit_transform(embeddings)
        }
        
        reduced_embeddings = {}
        for name, reducer in reduction_techniques.items():
            if name == 'UMAP':
                reduced_embeddings[name] = reducer.fit_transform(embeddings)
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=5)
                reduced_embeddings[name] = pca.fit_transform(embeddings)
        
        return reduced_embeddings
    
    def _create_vectorizer(self):
        """
        Create different CountVectorizer configurations
        
        Returns:
            list: Vectorizer configurations
        """
        vectorizer_configs = [
            CountVectorizer(stop_words='english', max_df=0.95, min_df=2),
            CountVectorizer(stop_words='english', max_df=0.9, min_df=3),
            CountVectorizer(ngram_range=(1,2), stop_words='english')
        ]
        return vectorizer_configs
    
    def _evaluate_clustering(self, topics, embeddings):
        """
        Evaluate clustering performance using multiple metrics
        
        Args:
            topics (list): Cluster assignments
            embeddings (numpy array): Document embeddings
        
        Returns:
            dict: Clustering performance metrics
        """
        metrics = {
            'silhouette_score': silhouette_score(embeddings, topics),
            'calinski_harabasz': calinski_harabasz_score(embeddings, topics),
            'davies_bouldin': davies_bouldin_score(embeddings, topics)
        }
        return metrics
    
    def optimize_bertopic(self):
        """
        Comprehensive BERTopic model exploration and optimization
        """
        # Hyperparameter grid
        param_grid = {
            'n_clusters': [5, 10, 15, 20],
            'min_cluster_size': [10, 30, 50],
            'nr_topics': [None, 10, 20, 30]
        }
        
        # Embeddings exploration
        embedding_results = self._create_embeddings()
        
        for embedding_model, embeddings in embedding_results.items():
            # Dimensionality reduction
            reduced_embeddings = self._dimensionality_reduction(embeddings)
            
            # Vectorizer configurations
            vectorizers = self._create_vectorizer()
            
            # Hyperparameter tuning
            for params in ParameterGrid(param_grid):
                for vectorizer in vectorizers:
                    try:
                        umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
                        model = BERTopic(
                            embedding_model=embedding_model,
                            umap_model=umap_model,
                            vectorizer_model=vectorizer,
                            nr_topics=params['nr_topics'],
                            min_topic_size=params['min_cluster_size']
                        )
                        
                        topics, probs = model.fit_transform(
                            self.documents, 
                        )
                        
                        # Evaluate clustering
                        performance_metrics = self._evaluate_clustering(
                            topics, umap_model.embedding_
                        )
                        
                        # Store results
                        result = {
                            'embedding_model': str(embedding_model),
                            'dimensionality_reduction': 'UMAP',
                            'vectorizer': str(vectorizer),
                            'hyperparameters': params,
                            'metrics': performance_metrics,
                            'num_topics': len(set(topics))
                        }
                        
                        self.results.append(result)
                        
                    except Exception as e:
                        print(f"Error in configuration: {e}")
    
    def visualize_results(self):
        """
        Visualize and summarize the exploration results
        """
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        df.to_pickle("bertopic_results.pkl")
        import pdb; pdb.set_trace()
        # Plot performance metrics
        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(['silhouette_score', 'calinski_harabasz', 'davies_bouldin'], 1):
            plt.subplot(1, 3, i)
            sns.boxplot(x='embedding_model', y=f'metrics.{metric}', data=df)
            plt.title(f'{metric} by Embedding Model')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Generate summary report
        summary = df.groupby(['embedding_model', 'dimensionality_reduction'])['metrics.silhouette_score'].mean()
        print("\nPerformance Summary:\n", summary) 
        

# Example usage
def main():
    # Sample documents (replace with your actual corpus)
    import sentences
    tree_sentences = sentences.read_sentences('trees_sentences.txt')
    bintree_sentences = sentences.read_sentences('binary_trees_sentences.txt')
    economics_sentences = sentences.read_sentences('economics_sentences.txt')
    healthcare_sentences = sentences.read_sentences('healthcare_sentences.txt')
    documents = tree_sentences + bintree_sentences + economics_sentences + healthcare_sentences
    print(len(documents))
    
    explorer = BERTopicExplorer(documents)
    explorer.optimize_bertopic()
    explorer.visualize_results()

if __name__ == "__main__":
    main()