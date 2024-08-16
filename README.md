# Graphit for AutoCAD
## Description

The AutoCAD to Graph Data Extraction Tool is a utility designed to convert AutoCAD drawings into structured graph representations, provides knowledge representation by preserving the spatial and relational information inherent in the original vector data. Aiming to facilitate advanced processing and analysis using Graph Neural Networks (GNNs).  

 ## Features

* __Node Extraction:__ Extracts key geometric points (e.g., line endpoints, circle centers) from AutoCAD entities and represents them as nodes.
* __Edge Creation:__ Defines relationships and connections between nodes (e.g., lines, circles) and represents them as edges with attributes.
* __Object Level Statistics:__ Derives node & edge level statistics : node degree, eigenvector centrality, betweenness centrality.
* __Attribute Handling:__ Captures additional attributes of entities (e.g., length of lines, radius of circles) and includes them in the graph representation.
* __Directed Relationships:__ Supports hierarchical and part-of relationships, allowing for complex graph structures and dependencies.
* __JSON Output:__ Outputs the graph representation in JSON format, suitable for loading into Python for further processing with libraries like NetworkX.

## Use Cases

* __Graph Neural Networks (GNNs):__ Prepare AutoCAD drawing data for advanced machine learning tasks using GNNs.
* __Object Recognition:__ Enable object recognition and classification tasks by converting geometric entities into structured graph data.
* __Data Analysis:__ Facilitate spatial and relational analysis of drawing data in a graph-based framework.
