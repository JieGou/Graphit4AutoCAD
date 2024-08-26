# Graphit for AutoCAD
## Overview

The AutoCAD to Graph Data Extraction Tool is a utility designed to convert AutoCAD drawings into structured graph representations, provides knowledge representation by preserving the spatial and relational information inherent in the original vector data. Aiming to facilitate advanced processing and analysis using Graph Neural Networks (GNNs).  

 ## Features

* __Node Extraction:__ Extracts key geometric points (e.g., line endpoints, circle centers) from AutoCAD entities and represents them as nodes.
* __Edge Creation:__ Defines relationships and connections between nodes (e.g., lines, circles) and represents them as edges with attributes.
* __Object Level Features:__ Derives node & edge level statistics : node degree, eigenvector centrality, betweenness centrality, closeness centrality, clustering coefficient.
* __Graph Level Features:__ Derives graph level statistics : average node degrees, graph density, graph diameter, node degree kernel, degree distribution kernel, connected components, average path length, assortivity coefficient, transitivity (global clustering coefficient), spectral radius .
* __Attribute Handling:__ Captures additional attributes of entities (e.g., length of lines, radius of circles) and includes them in the graph representation.
* __Directed Relationships:__ Supports hierarchical and part-of relationships, allowing for complex graph structures and dependencies.
* __JSON Output:__ Outputs the graph representation in JSON format, suitable for loading into Python for further processing with libraries like NetworkX.

## Use Cases

* __Graph Neural Networks (GNNs):__ Prepare AutoCAD drawing data for advanced machine learning tasks using GNNs.
* __Object Recognition:__ Enable object recognition and classification tasks by converting geometric entities into structured graph data.
* __Data Analysis:__ Facilitate spatial and relational analysis of drawing data in a graph-based framework.

## Installation

**1- Clone or Download the Repository:**
```ruby
git clone https://github.com/janusquadrifrons/Graphit4AutoCAD
```
**2- Build the Project:** Open the solution file in your preferred IDE and build the project. Make sure you have all dependencies installed, including ML.NET and the necessary AutoCAD libraries.

**3- Load the Plug-in in AutoCAD:** Type the NETLOAD command in AutoCAD to load the plug-in within an active session.

## Usage with Commands

#### EXTRACTTOGRAPH      
-Function       : Extracts relevant data from the current drawing and saves it as a .json file.\
-Purpose        : The file can be used to train the model.
