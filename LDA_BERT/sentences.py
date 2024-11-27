import random

class SentenceGenerator:
    def __init__(self):
        # Trees in Nature Sentence Templates (kept previous version)
        self.trees_templates = [
            "To the right of the forest path, an old oak tree stood with branches reaching left toward the sky.",
            "Children love to search for interesting bird nests hidden in the dense tree canopy.",
            "The binary patterns of leaf arrangement create a unique natural architecture.",
            "Maple trees have complex relationships with the surrounding forest ecosystem.",
            "We traversed the forest, noting how each tree creates references to its environmental history.",
            "My child collected different leaf shapes, examining their intricate designs.",
            "The forest floor revealed a complex network of tree root relationships.",
            "Ancient trees stand as living references to centuries of environmental change.",
            "Wind moves through tree branches, creating a dynamic search for balance.",
            "The architectural structure of redwood forests creates natural cathedral-like spaces.",
            "Trees on the left side of the valley grow differently from those on the right.",
            "Each tree branch tells a story of survival and adaptation.",
            "The forest ecosystem demonstrates complex interdependent relationships.",
            "Children often explore the hidden worlds within tree bark and branches.",
            "Seasonal changes reveal the dynamic architecture of deciduous trees.",
            "We search the forest for unique tree specimens with unusual growth patterns.",
            "The binary rhythm of tree growth follows mathematical principles of nature.",
            "Tree roots create underground networks with intricate references to each other.",
            "From right to left, the forest landscape changes with different tree species.",
            "The natural world's architecture is most beautiful in ancient forest groves.",
            "Autumn leaves create a colorful carpet as trees shed their summer references.",
            "Wildlife traverses the forest, using trees as highways and homes.",
            "The unique relationships between trees and surrounding plants fascinate botanists.",
            "Each tree creates a complex binary pattern of growth and survival.",
            "Children discover entire ecosystems living within a single tree's branches.",
            "Forest paths traverse through dense networks of interconnected trees.",
            "The left side of the forest differs dramatically from the right in tree composition.",
            "Trees maintain subtle references to environmental changes throughout their lifetime.",
            "We search for the oldest trees, reading their stories through bark and rings.",
            "The architectural beauty of trees reveals nature's most intricate designs."
        ]

        # Binary Trees and Forest Algorithm Sentence Templates
        self.binary_trees_templates = [
            "Random forest algorithms create ensemble decision trees for more robust machine learning predictions.",
            "In machine learning, random forest algorithms combine multiple binary trees to improve predictive accuracy.",
            "The left and right branches of binary trees inspire the design of decision pathways.",
            "The architecture of forest algorithms mimics the complex decision-making processes of natural systems.",
            "A binary tree is a hierarchical data structure with a root node and two child nodes.",
            "In computer science, binary trees efficiently organize data through recursive branching.",
            "Each node in a binary tree contains a value and two potential child pointers.",
            "Binary search trees maintain an ordered structure for rapid data retrieval.",
            "The left subtree of a binary tree always contains smaller values than the root.",
            "Tree traversal algorithms like in-order, pre-order, and post-order help navigate binary structures.",
            "Red-black trees are a type of self-balancing binary search tree.",
            "Heap data structures implement binary tree principles for priority queue management.",
            "Recursive algorithms are particularly effective for manipulating binary tree structures.",
            "The depth of a binary tree determines its computational efficiency.",
            "Binary trees provide a fundamental framework for complex data organization strategies.",
            "Tree rotations help maintain the balance and performance of binary tree structures.",
            "Depth-first and breadth-first searches are common traversal methods for binary trees.",
            "Binary trees can represent hierarchical relationships in various computational models.",
            "The root node serves as the primary entry point for binary tree operations.",
            "Efficient memory allocation is crucial when implementing binary tree data structures.",
            "Mathematical operations and decision trees often utilize binary tree architectures."
        ]

        # Healthcare Sentence Templates
        self.healthcare_templates = [
            "Modern healthcare systems create complex networks of interdependent medical services.",
            "Machine learning algorithms increasingly support diagnostic decision-making in medical research.",
            "Electronic health records provide detailed references for patient medical histories.",
            "Patient care requires intricate references between multiple healthcare specialists and treatment protocols.",
            "Digital health technologies transform the traditional to medical diagnostics.",
            "Healthcare professionals search for innovative solutions to complex medical challenges.",
            "The architecture of healthcare delivery reflects sophisticated interconnected systems.",
            "Precision medicine creates personalized treatment pathways based on individual genetic references.",
            "Medical research traverses multiple disciplines to understand complex human health relationships.",
            "Technology enables more efficient search and analysis of patient medical histories.",
            "Healthcare data creates intricate networks of information about human physiological systems.",
            "Each medical diagnosis represents a complex decision tree of symptoms and potential treatments.",
            "The binary relationship between preventive and reactive healthcare continues to evolve.",
            "Artificial intelligence supports more nuanced medical decision-making processes.",
            "Healthcare ecosystems demonstrate complex interdependencies between technology and human expertise.",
            "Machine learning models optimize patient care through advanced predictive techniques.",
            "Medical researchers explore the intricate relationships between genetic and environmental factors.",
            "Digital health platforms create new references for patient-centered medical information.",
            "The complexity of human health mirrors the sophisticated networks in advanced medical systems.",
            "Healthcare innovation continues to search for more holistic approaches to treatment and prevention."
        ]

        # Economics Sentence Templates
        self.economics_templates = [
            "Economic systems create complex networks of interdependent market relationships.",
            "Machine learning algorithms increasingly support predictive economic modeling.",
            "The left and right economic policies represent dynamic approaches to resource allocation.",
            "Global economic networks require intricate references between multiple market factors.",
            "Digital technologies transform the traditional binary approach to economic analysis.",
            "Economic researchers search for innovative solutions to complex market challenges.",
            "The architecture of economic systems reflects sophisticated interconnected global markets.",
            "Predictive economic models create nuanced pathways for understanding market trends.",
            "Economic research traverses multiple disciplines to understand complex financial relationships.",
            "Technology enables more efficient search and analysis of economic data and trends.",
            "Economic data creates intricate networks of information about global market behaviors.",
            "Each economic decision represents a complex decision tree of potential outcomes.",
            "The binary relationship between supply and demand continues to evolve.",
            "Artificial intelligence supports more nuanced economic decision-making processes.",
            "Economic ecosystems demonstrate complex interdependencies between local and global markets.",
            "Predictive models optimize economic forecasting through advanced analytical techniques.",
            "Economic researchers explore the intricate relationships between market variables.",
            "Digital platforms create new references for economic information and analysis.",
            "The complexity of global markets mirrors the sophisticated networks in economic systems.",
            "Economic innovation continues to search for more holistic approaches to resource management."
        ]

    def generate_sentences(self, topic='trees', num_sentences=50):
        """
        Generate a specified number of sentences on a given topic
        
        Args:
            topic (str): 'trees', 'binary_trees', 'healthcare', or 'economics'
            num_sentences (int): Number of sentences to generate
        
        Returns:
            list: Generated sentences
        """
        if topic == 'trees':
            sentences = random.choices(self.trees_templates, k=num_sentences)
        elif topic == 'binary_trees':
            sentences = random.choices(self.binary_trees_templates, k=num_sentences)
        elif topic == 'healthcare':
            sentences = random.choices(self.healthcare_templates, k=num_sentences)
        elif topic == 'economics':
            sentences = random.choices(self.economics_templates, k=num_sentences)
        else:
            raise ValueError("Topic must be 'trees', 'binary_trees', 'healthcare', or 'economics'")
        
        return sentences
    
    def generate_documents(self, topics, num_documents = 50, sentences_per_doc = 5):
        """
        Generate a specified number of documents on given topics
        
        Args:
            topics (list): List of topics to generate documents for
            num_documents (int): Number of documents to generate
            sentences_per_doc (int): Number of sentences per document
        
        Returns:
            list: Generated documents
        """
        documents = []
        for _ in range(num_documents):
            document = []
            topic = random.choice(topics)
            sentences = self.generate_sentences(topic, num_sentences=sentences_per_doc)
            # Add all sentences to the document with a space between each sentence
            document = ''.join(sentences)

            documents += [document]
        return documents
    
    def save_sentences(self, sentences, filename):
        """
        Save generated sentences to a text file
        
        Args:
            sentences (list): Sentences to save
            filename (str): Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')

def main():
    generator = SentenceGenerator()
    
    # Generate sentences for all topics
    topics = ['trees', 'binary_trees', 'healthcare', 'economics']
    all_sentences = []
    for topic in topics:
        sentences = generator.generate_sentences(topic, 50)
        generator.save_sentences(sentences, f'{topic}_sentences.txt')
        
        print(f"\nSample {topic.capitalize()} Sentences:")
        for sentence in sentences[:5]:
            print(sentence)
    print("\nSentences generated and saved successfully!")
    
def read_sentences(filename):
    """
    Read sentences from a text file
    
    Args:
        filename (str): Input filename
    
    Returns:
        list: List of sentences
    """
    with open(filename, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences


if __name__ == "__main__":
    main()