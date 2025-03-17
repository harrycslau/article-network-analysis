import spacy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import networkx.algorithms.community as nx_comm
from matplotlib.cm import viridis

# Parameters
filepath = "rawdata/manuscript.txt"
top_n = 150
min_weight = 4
min_degree = 2

# Language Setting For English
lang_name  = "english"
lang_model = "en_core_web_sm"

# Language Setting For Finnish
#lang_name  = "finnish"
#lang_model = "fi_core_news_sm"


# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words(lang_name))

# Load SpaCy NLP model
nlp = spacy.load(lang_model)

def preprocess_text(text):
    doc = nlp(text)
    words = []
    for token in doc:
        lemma = token.lemma_.strip().lower()
        # Keep only alphabetic tokens with length >= 2, not in stopwords,
        # and ensure lemma itself is not empty after processing
        if token.is_alpha and len(lemma) >= 2 and lemma not in stop_words and lemma:
            words.append(lemma)
    
    # Create bigrams
    bigrams = [' '.join(pair) for pair in ngrams(words, 2)]  
    return words + bigrams  # Combine words and bigrams


def remove_empty_nodes(G):
    """Remove any nodes that are empty strings or contain only whitespace."""
    to_remove = [n for n in G.nodes() if not n or not n.strip()]  # Empty or whitespace-only
    if to_remove:
        print(f"Removing {len(to_remove)} empty or whitespace-only nodes")
        G.remove_nodes_from(to_remove)
    return G

def filter_low_degree_nodes(G, min_degree=2):
    """Removes nodes with very low degree to simplify the network."""
    to_remove = [node for node in G.nodes() if G.degree(node) < min_degree]
    G.remove_nodes_from(to_remove)
    return G

def build_cooccurrence_graph(words, window_size=3):
    """Builds a word co-occurrence graph based on a sliding window."""
    G = nx.Graph()
    
    # Filter out any empty strings or whitespace-only nodes
    words = [w for w in words if w and w.strip()]
    
    # Create edges from word co-occurrences
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        for w1, w2 in combinations(window, 2):
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)
    
    return G

def filter_graph(G, top_n=100, min_weight=3):
    """Filters the graph by keeping only the top N most connected words and removing weak edges."""
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:top_n]
    
    # Create a subgraph with the top nodes
    G_filtered = G.subgraph(top_nodes).copy()
    
    # Remove weak edges
    G_filtered = nx.Graph((u, v, d) for u, v, d in G_filtered.edges(data=True) if d['weight'] >= min_weight)
    
    return G_filtered

def analyze_graph(G):
    """Compute centrality metrics and detect key nodes."""
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Identify top 10 most central words
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return top_degree, top_betweenness

def detect_communities(G):
    """Detects communities in the graph using greedy modularity maximization."""
    communities = list(nx_comm.greedy_modularity_communities(G))
    
    # Create a mapping of node to community index
    node_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_community[node] = i
    
    # Print information about communities
    print(f"Detected {len(communities)} communities")
    for i, community in enumerate(communities[:5]):  # Show top 5 communities
        top_nodes = sorted(community, key=lambda n: G.degree(n), reverse=True)[:5]
        print(f"Community {i}: Size {len(community)}, Top nodes: {top_nodes}")
    
    return communities, node_community

def plot_network(G, node_community=None):
    """Visualizes the word network graph with community-based coloring."""
    # Debug: Check for empty nodes before plotting
    empty_nodes = [n for n in G.nodes() if not n or not n.strip()]
    if empty_nodes:
        print(f"Warning: Found {len(empty_nodes)} empty nodes: {empty_nodes}")
        # Remove them to prevent visualization issues
        G.remove_nodes_from(empty_nodes)
    
    # Skip visualization if graph is empty
    if len(G) == 0:
        print("Graph is empty after removing problematic nodes.")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Adjust node positions for better spacing
    pos = nx.spring_layout(G, seed=42, k=0.7)  
    
    # Compute degree-based node sizes
    node_degrees = np.array([G.degree(n) for n in G.nodes()])
    # Protect against empty graphs or divisions by zero
    max_degree = max(node_degrees) if node_degrees.size > 0 and max(node_degrees) > 0 else 1
    node_sizes = 300 + (node_degrees / max_degree * 1000)  # Scales node size
    
    # Color nodes based on community if available, otherwise use degree centrality
    if node_community:
        # Map each node to its community index
        colors = [node_community.get(node, 0) for node in G.nodes()]
        cmap = plt.cm.tab20  # Use tab20 for more distinct colors between communities
    else:
        colors = node_degrees / max_degree if max_degree > 0 else [0.5] * len(G)
        cmap = plt.cm.Blues
    
    # Draw network
    nx.draw(G, pos, node_color=colors, cmap=cmap, with_labels=False, 
            node_size=node_sizes, edge_color='gray', alpha=0.6)
    
    # Alternative: Display labels for important nodes (degree > 1) and any peripheral nodes
    central_nodes = {node: node for node in G.nodes() if G.degree(node) > 1 and node}
    peripheral_nodes = {node: node for node in G.nodes() if G.degree(node) == 1 and node}

    # Draw labels for central nodes with standard size
    nx.draw_networkx_labels(G, pos, central_nodes, font_size=10, font_color='black')

    # Draw labels for peripheral nodes with smaller size
    nx.draw_networkx_labels(G, pos, peripheral_nodes, font_size=8, font_color='darkgray')
    
    plt.title("Word Co-occurrence Network with Community Detection")
    plt.show()


if __name__ == "__main__":
    # Load manuscript text from file
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Preprocess text
    words = preprocess_text(text)
    
    # Build network
    G = build_cooccurrence_graph(words)

    # Remove empty nodes (if any)
    G = remove_empty_nodes(G)

    # Filter network
    G = filter_graph(G, top_n, min_weight)
    G = filter_low_degree_nodes(G, min_degree)

    # Detect communities
    communities, node_community = detect_communities(G)
    
    # Analyze network
    top_degree, top_betweenness = analyze_graph(G)
    
    # Print key words by centrality
    print("Top Words by Degree Centrality:", top_degree)
    print("Top Words by Betweenness Centrality:", top_betweenness)

    # Save graph to file for further analysis
    nx.write_graphml(G, "network_analysis.graphml")

    # Visualize graph with community colors
    plot_network(G, node_community)
