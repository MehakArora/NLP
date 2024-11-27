import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
def evaluate_clustering(topics, embeddings):
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

def calculate_similarities(vectors):
    """
    Calculate cosine similarities 
    
    Args:
        vectors (np.array): Embeddings or topic distributions
    
    Returns:
        similarities (np.array): Calculated similarities
    """
    from scipy.spatial.distance import cosine

    # Vector cosine similarity
    
    vector_similarities = np.zeros((len(vectors), len(vectors)))
    
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            vector_similarities[i][j] = cosine(vectors[i], vectors[j])
    
    return vector_similarities
    
def analyze_correlation(vector_similarities, topic_similarities):
    """
    Analyze correlation between vector similarities and topic predictions.
    
    Args:
        vector_similarities (np.array): Calculated similarities in embeddings
        topic_similarities (np.array): Calculated topic distribution similarities
    
    Returns:
        dict: Correlation analysis results
    """
    import scipy.stats as stats

    # Pearson correlation
    pearson_corr, p_value_pearson = stats.pearsonr(
        vector_similarities.flatten(), 
        topic_similarities.flatten()
    )
    
    # Spearman correlation
    spearman_corr, p_value_spearman = stats.spearmanr(
        vector_similarities.flatten(), 
        topic_similarities.flatten()
    )
    
    # R-squared
    slope, intercept, r_value, _, _ = stats.linregress(
        vector_similarities.flatten(), 
        topic_similarities.flatten()
    )
    r_squared = r_value ** 2
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': p_value_pearson,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': p_value_spearman,
        'r_squared': r_squared
    }



class LDAModelTrainer:
    def __init__(self, documents):
        """
        Initialize the LDA Model Trainer
        
        Args:
            documents (list): List of text documents to train the model
        """

        # Configure logging
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s', 
            level=logging.INFO
        )
        
        # Store original documents
        self.documents = documents
        
        # Preprocessing attributes
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.processed_docs = None
        self.id2word = None
        self.corpus = None
        
        # Model attributes
        self.lda_model = None
        self.evaluation_metrics = {}
    
    def preprocess(self, remove_stopwords=True, lemmatize = True, min_length=3):
        """
        Preprocess documents by tokenizing and optionally removing stopwords
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize tokens
            min_length (int): Minimum token length to keep
        
        Returns:
            list: Processed documents
        """
        processed_docs = []
        for doc in self.documents:
            # Tokenize
            tokens = word_tokenize(doc.lower())
            
            # Remove stopwords and short tokens
            if remove_stopwords:
                tokens = [
                    token for token in tokens 
                    if token not in self.stop_words 
                    and token.isalnum() 
                    and len(token) >= min_length
                ]

            if lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            processed_docs.append(tokens)
        
        self.processed_docs = processed_docs
        return processed_docs
    
    def create_dictionary_and_corpus(self):
        """
        Create dictionary and corpus for LDA model
        """
        if self.processed_docs is None:
            self.preprocess()
        
        # Create Dictionary
        self.id2word = corpora.Dictionary(self.processed_docs)
        
        # Filter extreme tokens
        self.id2word.filter_extremes(
            no_below=2,  # Ignore tokens that appear in less than 2 documents
            no_above=0.5  # Ignore tokens that appear in more than 50% of documents
        )
        
        # Create Corpus
        self.corpus = [self.id2word.doc2bow(doc) for doc in self.processed_docs]
    
    def train_lda_model(
        self, 
        num_topics=5, 
        passes=15, 
        alpha='auto', 
        eta='auto'
    ):
        """
        Train LDA model
        
        Args:
            num_topics (int): Number of topics to extract
            passes (int): Number of training passes
            alpha (str/float): Document-topic density
            eta (str/float): Topic-word density
        
        Returns:
            gensim.models.ldamodel.LdaModel: Trained LDA model
        """
        if self.corpus is None:
            self.create_dictionary_and_corpus()
        
        # Train LDA model
        self.lda_model = gensim.models.LdaModel( #changed frm LdaMulticore
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            alpha=alpha,
            eta=eta
        )
        
        return self.lda_model
    
    def predict_topics(self):
        """
        Predict topics for corpus
        
        Returns:
            list: Predicted topics for new documents
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet. Please train the model first.")
        
        distributions = np.zeros((len(self.corpus), self.lda_model.num_topics))
        for i in range(len(self.corpus)):
            topics = self.lda_model.get_document_topics(self.corpus[i])
            distributions[i, [idx for idx, _ in topics]] = [prob for _, prob in topics]
        
        return distributions

    def compute_coherence(self, model=None, coherence_type='c_v'):
        """
        Compute model coherence
        
        Args:
            model (LdaModel, optional): LDA model to evaluate
            coherence_type (str): Coherence measurement type
        
        Returns:
            float: Coherence score
        """
        if model is None:
            model = self.lda_model
        
        coherence_model = CoherenceModel(
            model=model, 
            texts=self.processed_docs, 
            dictionary=self.id2word, 
            coherence=coherence_type
        )
        
        return coherence_model.get_coherence()
    
    def compute_perplexity(self, model=None):
        """
        Compute model perplexity
        
        Args:
            model (LdaModel, optional): LDA model to evaluate
        
        Returns:
            float: Perplexity score
        """
        if model is None:
            model = self.lda_model
        
        # Compute perplexity on the training corpus
        perplexity = model.log_perplexity(self.corpus)
        
        # Return exp of negative log perplexity (standard perplexity calculation)
        return np.exp2(-perplexity)
    
    def find_optimal_topics(
        self, 
        max_topics=10, 
        passes = 15,
        start=2, 
        step=1,
        coherence_type='c_v'
    ):
        """
        Find optimal number of topics using multiple metrics
        
        Args:
            max_topics (int): Maximum number of topics to evaluate
            start (int): Starting number of topics
            step (int): Step size for topic increments
            passes (int): Number of training passes
            coherence_type (str): Coherence measurement type
        
        Returns:
            dict: Evaluation metrics for different topic counts
        """
        # Reset evaluation metrics
        self.evaluation_metrics = {
            'num_topics': [],
            'coherence_scores': [],
            'perplexity_scores': []
        }
        
        for num_topics in range(start, max_topics + 1, step):
            # Train model
            model = self.train_lda_model(num_topics=num_topics, passes=passes)
            
            # Compute metrics
            coherence_score = self.compute_coherence(
                model=model, 
                coherence_type=coherence_type
            )
            perplexity_score = self.compute_perplexity(model=model)
            
            # Store metrics
            self.evaluation_metrics['num_topics'].append(num_topics)
            self.evaluation_metrics['coherence_scores'].append(coherence_score)
            self.evaluation_metrics['perplexity_scores'].append(perplexity_score)
        
        return self.evaluation_metrics
    
    def visualize_topic_words(self, num_words=10, figsize=(15, 10)):
        """
        Create visualization of top words per topic.
        
        Args:
            num_words (int): Number of top words to display per topic
            figsize (tuple): Figure size for the plot
        """
        # Prepare the plot
        fig, axes = plt.subplots(
            nrows=len(self.lda_model.print_topics()), 
            ncols=1, 
            figsize=figsize
        )
        fig.suptitle('Top Words per Topic', fontsize=16)
        
        # Extract and plot words for each topic
        for idx, topic in self.lda_model.print_topics(num_words=num_words):
            # Parse topic words and their weights
            words = [word.split('*')[1].strip('"') for word in topic.split('+')]
            weights = [float(word.split('*')[0]) for word in topic.split('+')]
            
            # Plot horizontal bar chart
            axes[idx].barh(words, weights)
            axes[idx].set_title(f'Topic {idx}', fontsize=12)
            axes[idx].invert_yaxis()  # Words from top to bottom
            axes[idx].set_xlabel('Word Weight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_topic_metrics(self):
        """
        Plot topic metrics (Coherence and Perplexity)
        """
        if not self.evaluation_metrics:
            self.find_optimal_topics()
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Coherence Scores
        ax1.plot(
            self.evaluation_metrics['num_topics'], 
            self.evaluation_metrics['coherence_scores'], 
            marker='o'
        )
        ax1.set_title('Topic Coherence Scores')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Coherence Score')
        
        # Perplexity Scores
        ax2.plot(
            self.evaluation_metrics['num_topics'], 
            self.evaluation_metrics['perplexity_scores'], 
            marker='o',
            color='red'
        )
        ax2.set_title('Topic Perplexity Scores')
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Perplexity Score')
        
        plt.tight_layout()
        plt.show()
    
    def get_optimal_topics(self, criteria='balanced'):
        """
        Determine optimal number of topics based on different criteria
        
        Args:
            criteria (str): Method to select optimal topics
                'balanced': Balances coherence and perplexity
                'max_coherence': Maximizes coherence
                'min_perplexity': Minimizes perplexity
        
        Returns:
            int: Optimal number of topics
        """
        if not self.evaluation_metrics:
            self.find_optimal_topics()
        
        # Normalize metrics
        def normalize(scores):
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        coherence_norm = normalize(self.evaluation_metrics['coherence_scores'])
        perplexity_norm = normalize(self.evaluation_metrics['perplexity_scores'])
        
        if criteria == 'max_coherence':
            optimal_index = np.argmax(self.evaluation_metrics['coherence_scores'])
        elif criteria == 'min_perplexity':
            optimal_index = np.argmin(self.evaluation_metrics['perplexity_scores'])
        else:  # balanced approach
            # Maximize coherence and minimize perplexity
            balance_scores = coherence_norm - perplexity_norm
            optimal_index = np.argmax(balance_scores)
        
        return self.evaluation_metrics['num_topics'][optimal_index]
    
    def print_topics(self, num_topics=None, num_words=10):
        """
        Print topics with their top words
        
        Args:
            num_topics (int, optional): Number of topics to print
            num_words (int): Number of top words per topic
        """
        if num_topics is None:
            num_topics = self.lda_model.num_topics
        
        for idx, topic in self.lda_model.print_topics(num_topics=num_topics, num_words=num_words):
            print(f"Topic {idx}: {topic}")

def main():
    # Example usage with slightly more documents
    sample_documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "Natural language processing focuses on the interaction between computers and human language.",
        "Artificial intelligence aims to create intelligent machines that work and react like humans.",
        "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics.",
        "Neural networks are computational models inspired by biological neural networks.",
        "Statistical learning theory provides a framework for machine learning algorithms.",
        "Computer vision uses artificial intelligence to derive meaningful information from digital images.",
        "Robotics involves design, construction, and operation of robots.",
        "Big data analytics processes large and complex data sets to uncover hidden patterns."
    ]
    
    # Initialize and train LDA model
    lda_trainer = LDAModelTrainer(sample_documents)
    
    # Preprocess documents
    lda_trainer.preprocess()
    
    # Find optimal number of topics with metrics
    lda_trainer.find_optimal_topics()
    
    # Plot topic metrics
    lda_trainer.plot_topic_metrics()
    
    # Get optimal number of topics
    optimal_topics = lda_trainer.get_optimal_topics()
    print(f"Optimal number of topics: {optimal_topics}")
    
    # Train final model with optimal topics
    final_model = lda_trainer.train_lda_model(num_topics=optimal_topics)
    
    # Print topics
    lda_trainer.print_topics()

if __name__ == "__main__":
    main()

