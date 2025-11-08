## FlavorGraph - Interactive Ingredient Pairing Web App

An interactive web application for exploring ingredient pairings using a force-directed graph visualization, similar to [epicure.kaikaku.ai](https://epicure.kaikaku.ai).

### Features

- 🎨 **Interactive Graph Visualization**: Force-directed graph with D3.js showing ingredient relationships
- 🔍 **Search Functionality**: Search and filter ingredients in real-time
- 🎯 **Node Selection**: Click on nodes to explore connections and get pairing recommendations
- 🔎 **Zoom Controls**: Zoom in/out and reset view controls
- 🌈 **Color-Coded Categories**: Ingredients are color-coded by category (protein, dairy, vegetable, etc.)
- 📊 **Pairing Recommendations**: Get top pairing recommendations for any ingredient

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The app automatically downloads the FlavorGraph data from GitHub on first run. This may take a few moments as it loads ~8K nodes and ~147K edges.

### Running the App

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

### Usage

- **Search**: Type in the search box to find ingredients
- **Click Nodes**: Click on any node to see its connections and get pairing recommendations
- **Zoom**: Use the zoom controls in the bottom right corner
- **Drag**: Drag nodes to rearrange the graph
- **Reset**: Click the reset button to return to the full graph view

### Project Structure

```
DairyNet/
├── app.py              # Flask backend with API endpoints
├── templates/
│   └── index.html      # Frontend with D3.js visualization
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### API Endpoints

- `GET /` - Main page
- `GET /api/graph?center=<ingredient>&depth=<depth>&max_nodes=<n>` - Get graph data
- `GET /api/search?q=<query>` - Search for ingredients
- `GET /api/recommendations?ingredient=<name>&top_n=<n>&method=<method>` - Get pairing recommendations
  - Methods: `hybrid` (default), `chemical`, `recipe`
- `GET /api/ingredients` - Get all ingredients (limited to 1000 for performance)
- `GET /api/stats` - Get graph statistics

### Technologies

- **Backend**: Flask, NetworkX, Pandas
- **Frontend**: D3.js, HTML5, CSS3
- **Data**: Real FlavorGraph data from [GitHub](https://github.com/lamypark/FlavorGraph/tree/master)
  - 8,298 nodes (6,653 ingredients + 1,645 compounds)
  - 147,179 edges (111,355 ingredient-ingredient + 35,440 ingredient-compound + 384 ingredient-drug)
