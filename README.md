# Graphit for AutoCAD
## Description

The AutoCAD to Graph Data Extraction Tool is a utility designed to convert AutoCAD drawings into structured graph representations, provides knowledge representation by preserving the spatial and relational information inherent in the original vector data. Aiming to facilitate advanced processing and analysis using Graph Neural Networks (GNNs).  

 ## Features

    - Node Extraction: Extracts key geometric points (e.g., line endpoints, circle centers) from AutoCAD entities and represents them as nodes.
    * Edge Creation: Defines relationships and connections between nodes (e.g., lines, circles) and represents them as edges with attributes.
    * Attribute Handling: Captures additional attributes of entities (e.g., length of lines, radius of circles) and includes them in the graph representation.
    * Directed Relationships: Supports hierarchical and part-of relationships, allowing for complex graph structures and dependencies.
    * JSON Output: Outputs the graph representation in JSON format, suitable for loading into Python for further processing with libraries like NetworkX.
