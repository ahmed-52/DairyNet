from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import networkx as nx
import pandas as pd
import json
from typing import List, Dict, Optional
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# ============================================================================
# UNIVERSAL FLAVOR GRAPH CLASS
# ============================================================================

class UniversalFlavorGraph:
    """
    Complete food pairing system using chemical compounds and recipe data
    Works with ANY ingredient in FlavorGraph
    """
    
    def __init__(self, edges_df, nodes_df):
        self.edges_df = edges_df
        self.nodes_df = nodes_df
        
        # Create mappings
        self.node_id_to_name = dict(zip(nodes_df['node_id'], nodes_df['name']))
        self.node_name_to_id = {str(v).lower(): k for k, v in self.node_id_to_name.items()}
        
        # Store node types
        self.node_types = dict(zip(nodes_df['node_id'], nodes_df['node_type']))
        
        # Identify different node types
        self.ingredient_ids = set()
        self.compound_ids = set()
        self.drug_ids = set()
        
        # Build all graphs
        self.build_graphs()
        
    def build_graphs(self):
        """Build separate graphs for different relationship types"""
        
        print("\n🔨 Building knowledge graphs...")
        
        # 1. Ingredient-Ingredient graph (recipe co-occurrence)
        self.ing_graph = nx.Graph()
        ing_edges = self.edges_df[self.edges_df['edge_type'] == 'ingr-ingr']
        for _, row in ing_edges.iterrows():
            self.ing_graph.add_edge(row['id_1'], row['id_2'], 
                                   weight=row['score'])
            self.ingredient_ids.add(row['id_1'])
            self.ingredient_ids.add(row['id_2'])
        
        # 2. Ingredient-Compound bipartite graph
        self.compound_graph = nx.Graph()
        comp_edges = self.edges_df[self.edges_df['edge_type'] == 'ingr-fcomp']
        for _, row in comp_edges.iterrows():
            self.compound_graph.add_edge(row['id_1'], row['id_2'], 
                                        weight=row.get('score', 1.0))
            self.ingredient_ids.add(row['id_1'])
            self.compound_ids.add(row['id_2'])
        
        # 3. Ingredient-Drug graph (optional for health benefits)
        self.drug_graph = nx.Graph()
        drug_edges = self.edges_df[self.edges_df['edge_type'] == 'ingr-dcomp']
        for _, row in drug_edges.iterrows():
            self.drug_graph.add_edge(row['id_1'], row['id_2'], 
                                    weight=row.get('score', 1.0))
            self.ingredient_ids.add(row['id_1'])
            self.drug_ids.add(row['id_2'])
        
        print(f"✅ Built graphs:")
        print(f"   • {len(self.ingredient_ids)} food ingredients")
        print(f"   • {len(self.compound_ids)} flavor compounds")
        print(f"   • {len(self.drug_ids)} drug compounds")
        print(f"   • {self.ing_graph.number_of_edges()} ingredient-ingredient edges")
        print(f"   • {self.compound_graph.number_of_edges()} ingredient-compound edges")
        print(f"   • {self.drug_graph.number_of_edges()} ingredient-drug edges")
    
    def search_ingredient(self, query: str) -> List[Dict]:
        """Search for ingredients by name"""
        query = query.lower()
        matches = []
        
        for node_id in self.ingredient_ids:
            name = self.node_id_to_name.get(node_id, '')
            if query in str(name).lower():
                matches.append({
                    'id': node_id,
                    'name': name,
                    'num_compounds': len(self.get_compounds_for_ingredient(name)),
                    'num_recipes': self.ing_graph.degree(node_id) if node_id in self.ing_graph else 0
                })
        
        return sorted(matches, key=lambda x: x['num_compounds'], reverse=True)
    
    def get_ingredient_id(self, ingredient_name: str) -> Optional[int]:
        """Get ingredient ID from name (fuzzy match)"""
        ingredient_name = str(ingredient_name).lower()
        
        # Exact match
        if ingredient_name in self.node_name_to_id:
            return self.node_name_to_id[ingredient_name]
        
        # Partial match
        for name, nid in self.node_name_to_id.items():
            if ingredient_name in name and nid in self.ingredient_ids:
                return nid
        
        return None
    
    def get_compounds_for_ingredient(self, ingredient_name: str) -> List[Dict]:
        """Get all flavor compounds for an ingredient"""
        ingredient_id = self.get_ingredient_id(ingredient_name)
        
        if not ingredient_id or ingredient_id not in self.compound_graph:
            return []
        
        compounds = []
        for neighbor in self.compound_graph.neighbors(ingredient_id):
            if neighbor in self.compound_ids:
                compounds.append({
                    'id': neighbor,
                    'name': self.node_id_to_name.get(neighbor, f"Compound_{neighbor}"),
                    'weight': self.compound_graph[ingredient_id][neighbor]['weight']
                })
        
        return sorted(compounds, key=lambda x: x['weight'], reverse=True)
    
    def get_ingredients_with_compound(self, compound_id: int) -> List[Dict]:
        """Get all ingredients that contain a specific compound"""
        if compound_id not in self.compound_graph:
            return []
        
        ingredients = []
        for neighbor in self.compound_graph.neighbors(compound_id):
            if neighbor in self.ingredient_ids:
                ingredients.append({
                    'id': neighbor,
                    'name': self.node_id_to_name.get(neighbor, f"Ingredient_{neighbor}"),
                    'weight': self.compound_graph[neighbor][compound_id]['weight']
                })
        
        return ingredients
    
    def compound_similarity(self, ingredient1: str, ingredient2: str) -> float:
        """Calculate Jaccard similarity based on shared compounds"""
        compounds1 = set(c['id'] for c in self.get_compounds_for_ingredient(ingredient1))
        compounds2 = set(c['id'] for c in self.get_compounds_for_ingredient(ingredient2))
        
        if not compounds1 or not compounds2:
            return 0.0
        
        intersection = len(compounds1.intersection(compounds2))
        union = len(compounds1.union(compounds2))
        
        return intersection / union if union > 0 else 0.0
    
    def recipe_cooccurrence_score(self, ingredient1: str, ingredient2: str) -> float:
        """Get recipe co-occurrence score"""
        id1 = self.get_ingredient_id(ingredient1)
        id2 = self.get_ingredient_id(ingredient2)
        
        if not id1 or not id2:
            return 0.0
        
        if self.ing_graph.has_edge(id1, id2):
            return self.ing_graph[id1][id2]['weight']
        
        return 0.0
    
    def recommend_pairings(self, 
                          base_ingredient: str, 
                          method: str = 'hybrid',
                          top_n: int = 10,
                          min_shared_compounds: int = 1,
                          category_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Recommend ingredient pairings
        
        Methods:
        - 'chemical': Based purely on chemical compound similarity
        - 'recipe': Based purely on recipe co-occurrence
        - 'hybrid': Combines both (default)
        """
        
        base_id = self.get_ingredient_id(base_ingredient)
        if not base_id:
            return []
        
        base_compounds = self.get_compounds_for_ingredient(base_ingredient)
        
        if not base_compounds and method in ['chemical', 'hybrid']:
            method = 'recipe'
        
        candidates = defaultdict(lambda: {
            'shared_compounds': [],
            'compound_similarity': 0.0,
            'recipe_score': 0.0,
            'final_score': 0.0
        })
        
        # Method 1: Chemical similarity
        if method in ['chemical', 'hybrid']:
            for compound in base_compounds:
                ingredients_with_compound = self.get_ingredients_with_compound(compound['id'])
                
                for ing in ingredients_with_compound:
                    if ing['id'] != base_id:
                        ing_name = ing['name']
                        candidates[ing_name]['shared_compounds'].append(compound['name'])
        
        # Method 2: Recipe co-occurrence
        if method in ['recipe', 'hybrid']:
            if base_id in self.ing_graph:
                for neighbor in self.ing_graph.neighbors(base_id):
                    ing_name = self.node_id_to_name.get(neighbor)
                    if ing_name:
                        candidates[ing_name]['recipe_score'] = self.ing_graph[base_id][neighbor]['weight']
        
        # Calculate final scores
        recommendations = []
        
        for ing_name, data in candidates.items():
            # Filter by minimum shared compounds
            if len(data['shared_compounds']) < min_shared_compounds:
                continue
            
            # Calculate compound similarity
            if method in ['chemical', 'hybrid']:
                data['compound_similarity'] = self.compound_similarity(base_ingredient, ing_name)
            
            # Combined score
            if method == 'chemical':
                data['final_score'] = data['compound_similarity']
            elif method == 'recipe':
                data['final_score'] = data['recipe_score']
            else:  # hybrid
                data['final_score'] = (0.6 * data['compound_similarity']) + (0.4 * data['recipe_score'])
            
            recommendations.append({
                'ingredient': ing_name,
                'compound_similarity': data['compound_similarity'],
                'recipe_score': data['recipe_score'],
                'final_score': data['final_score'],
                'shared_compounds': data['shared_compounds'][:5],
                'num_shared_compounds': len(data['shared_compounds'])
            })
        
        # Sort by final score
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return recommendations[:top_n]
    
    def get_health_compounds_for_ingredient(self, ingredient_name: str) -> List[Dict]:
        """Get all health/drug compounds for an ingredient"""
        ingredient_id = self.get_ingredient_id(ingredient_name)
        
        if not ingredient_id or ingredient_id not in self.drug_graph:
            return []
        
        compounds = []
        for neighbor in self.drug_graph.neighbors(ingredient_id):
            if neighbor in self.drug_ids:
                compounds.append({
                    'id': neighbor,
                    'name': self.node_id_to_name.get(neighbor, f"Health_Compound_{neighbor}"),
                    'weight': self.drug_graph[ingredient_id][neighbor]['weight']
                })
        
        return sorted(compounds, key=lambda x: x['weight'], reverse=True)
    
    def get_shared_health_compounds(self, ingredient1: str, ingredient2: str) -> List[Dict]:
        """Get shared health compounds between two ingredients"""
        compounds1 = {c['id']: c for c in self.get_health_compounds_for_ingredient(ingredient1)}
        compounds2 = {c['id']: c for c in self.get_health_compounds_for_ingredient(ingredient2)}
        
        shared_ids = set(compounds1.keys()).intersection(set(compounds2.keys()))
        
        return [
            {
                'id': cid,
                'name': self.node_id_to_name.get(cid, f"Health_Compound_{cid}"),
                'in_ing1': compounds1[cid]['weight'],
                'in_ing2': compounds2[cid]['weight']
            }
            for cid in shared_ids
        ]
    
    def get_graph_data(self, center_ingredient: Optional[str] = None, depth: int = 1, max_nodes: int = 50):
        """Get graph data for visualization - only shows directly connected nodes when ingredient is selected"""
        nodes = []
        edges = []
        
        if center_ingredient:
            center_id = self.get_ingredient_id(center_ingredient)
            if center_id is None or center_id not in self.ing_graph:
                return {"nodes": [], "edges": []}
            
            # Only get DIRECT neighbors (depth=1) of the center ingredient
            subgraph_nodes = {center_id}
            
            # Get only direct neighbors
            if center_id in self.ing_graph:
                neighbors = list(self.ing_graph.neighbors(center_id))
                # Sort by edge weight and take top connections
                neighbor_weights = [(nid, self.ing_graph[center_id][nid].get('weight', 0)) 
                                   for nid in neighbors]
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                
                # Limit to top connected neighbors
                top_neighbors = [nid for nid, _ in neighbor_weights[:max_nodes]]
                subgraph_nodes.update(top_neighbors)
            
            # Get edges - only edges connected to center
            for node in subgraph_nodes:
                if node == center_id:
                    # All edges from center
                    if node in self.ing_graph:
                        for neighbor in self.ing_graph.neighbors(node):
                            if neighbor in subgraph_nodes:
                                # Get health information for this edge
                                ing1_name = self.node_id_to_name.get(node, '')
                                ing2_name = self.node_id_to_name.get(neighbor, '')
                                shared_health = self.get_shared_health_compounds(ing1_name, ing2_name)
                                
                                edge_data = {
                                    "source": node,
                                    "target": neighbor,
                                    "weight": self.ing_graph[node][neighbor].get('weight', 0.5),
                                    "recipe_score": self.ing_graph[node][neighbor].get('weight', 0.5)
                                }
                                
                                # Add health information
                                if shared_health:
                                    edge_data["health_compounds"] = [c['name'] for c in shared_health[:3]]
                                    edge_data["num_health_compounds"] = len(shared_health)
                                    edge_data["has_health_benefits"] = True
                                else:
                                    edge_data["health_compounds"] = []
                                    edge_data["num_health_compounds"] = 0
                                    edge_data["has_health_benefits"] = False
                                
                                edges.append(edge_data)
        else:
            # Return sample of all nodes (too many to show all)
            all_ingredient_ids = list(self.ingredient_ids)
            # Get top connected ingredients
            node_degrees = [(nid, self.ing_graph.degree(nid) if nid in self.ing_graph else 0) 
                           for nid in all_ingredient_ids]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            subgraph_nodes = {nid for nid, _ in node_degrees[:max_nodes]}
            
            # Get edges within subgraph
            for node in subgraph_nodes:
                if node in self.ing_graph:
                    for neighbor in self.ing_graph.neighbors(node):
                        if neighbor in subgraph_nodes and node < neighbor:
                            ing1_name = self.node_id_to_name.get(node, '')
                            ing2_name = self.node_id_to_name.get(neighbor, '')
                            shared_health = self.get_shared_health_compounds(ing1_name, ing2_name)
                            
                            edge_data = {
                                "source": node,
                                "target": neighbor,
                                "weight": self.ing_graph[node][neighbor].get('weight', 0.5),
                                "recipe_score": self.ing_graph[node][neighbor].get('weight', 0.5)
                            }
                            
                            if shared_health:
                                edge_data["health_compounds"] = [c['name'] for c in shared_health[:3]]
                                edge_data["num_health_compounds"] = len(shared_health)
                                edge_data["has_health_benefits"] = True
                            else:
                                edge_data["health_compounds"] = []
                                edge_data["num_health_compounds"] = 0
                                edge_data["has_health_benefits"] = False
                            
                            edges.append(edge_data)
        
        # Create nodes with categories
        for node_id in subgraph_nodes:
            if node_id not in self.ingredient_ids:
                continue
                
            name = self.node_id_to_name.get(node_id, f"Ingredient_{node_id}")
            node_type = self.node_types.get(node_id, 'ingredient')
            
            # Categorize based on name patterns (since we don't have explicit categories)
            category = self._categorize_ingredient(name)
            
            is_center = (center_ingredient and 
                        self.get_ingredient_id(center_ingredient) == node_id)
            
            nodes.append({
                "id": node_id,
                "name": name,
                "category": category,
                "degree": self.ing_graph.degree(node_id) if node_id in self.ing_graph else 0,
                "isCenter": is_center
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def _categorize_ingredient(self, name: str) -> str:
        """Categorize ingredient based on name patterns"""
        name_lower = name.lower()
        
        # Protein
        protein_keywords = ['beef', 'chicken', 'pork', 'lamb', 'turkey', 'duck', 'fish', 
                           'salmon', 'tuna', 'shrimp', 'crab', 'sausage', 'bacon', 'ham']
        if any(kw in name_lower for kw in protein_keywords):
            return "protein"
        
        # Dairy
        dairy_keywords = ['cheese', 'milk', 'cream', 'butter', 'yogurt', 'sour cream']
        if any(kw in name_lower for kw in dairy_keywords):
            return "dairy"
        
        # Vegetable
        vegetable_keywords = ['tomato', 'onion', 'garlic', 'potato', 'carrot', 'broccoli',
                              'pepper', 'mushroom', 'spinach', 'lettuce', 'cucumber', 'corn',
                              'pea', 'bean', 'squash', 'zucchini', 'celery', 'cabbage']
        if any(kw in name_lower for kw in vegetable_keywords):
            return "vegetable"
        
        # Grain
        grain_keywords = ['rice', 'wheat', 'flour', 'bread', 'pasta', 'noodle', 'barley',
                         'oats', 'quinoa', 'tortilla', 'cracker']
        if any(kw in name_lower for kw in grain_keywords):
            return "grain"
        
        # Fruit
        fruit_keywords = ['apple', 'banana', 'orange', 'lemon', 'lime', 'berry', 'strawberry',
                         'grape', 'peach', 'pear', 'cherry', 'pineapple', 'mango', 'avocado']
        if any(kw in name_lower for kw in fruit_keywords):
            return "fruit"
        
        # Spice/Herb
        spice_keywords = ['basil', 'oregano', 'thyme', 'rosemary', 'sage', 'parsley', 'cilantro',
                         'chili', 'pepper', 'cumin', 'coriander', 'paprika', 'curry', 'mustard',
                         'sauce', 'hot', 'spice']
        if any(kw in name_lower for kw in spice_keywords):
            return "spice"
        
        # Sweet
        sweet_keywords = ['chocolate', 'sugar', 'vanilla', 'cocoa', 'honey', 'syrup', 'candy',
                        'sweet', 'caramel', 'fudge']
        if any(kw in name_lower for kw in sweet_keywords):
            return "sweet"
        
        # Beverage
        beverage_keywords = ['beer', 'wine', 'coffee', 'tea', 'juice', 'stock', 'broth']
        if any(kw in name_lower for kw in beverage_keywords):
            return "beverage"
        
        return "other"

# ============================================================================
# LOAD DATA AND INITIALIZE
# ============================================================================

print("📥 Loading FlavorGraph data from GitHub...")
try:
    edges_df = pd.read_csv("https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input/edges_191120.csv")
    nodes_df = pd.read_csv("https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input/nodes_191120.csv")
    print(f"✅ Loaded {len(nodes_df)} nodes and {len(edges_df)} edges")
    
    print("\n🚀 Initializing Universal Flavor Graph...")
    flavor_graph = UniversalFlavorGraph(edges_df, nodes_df)
    print("✅ Ready!\n")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("⚠️  Falling back to mock data...")
    # Fallback to mock data if GitHub is unavailable
    flavor_graph = None

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graph')
def get_graph():
    """Get graph data"""
    if flavor_graph is None:
        return jsonify({"nodes": [], "edges": []})
    
    center = request.args.get('center')
    depth = int(request.args.get('depth', 1))  # Default to 1 (direct neighbors only)
    max_nodes = int(request.args.get('max_nodes', 50))  # Reduced default
    
    data = flavor_graph.get_graph_data(center_ingredient=center, depth=depth, max_nodes=max_nodes)
    return jsonify(data)

@app.route('/api/search')
def search():
    """Search for ingredients"""
    if flavor_graph is None:
        return jsonify([])
    
    query = request.args.get('q', '')
    results = flavor_graph.search_ingredient(query)
    
    # Format for frontend
    formatted_results = [{
        "id": r['id'],
        "name": r['name'],
        "degree": r['num_recipes']
    } for r in results]
    
    return jsonify(formatted_results)

@app.route('/api/recommendations')
def recommendations():
    """Get pairing recommendations"""
    if flavor_graph is None:
        return jsonify([])
    
    ingredient = request.args.get('ingredient')
    top_n = int(request.args.get('top_n', 10))
    method = request.args.get('method', 'hybrid')
    
    results = flavor_graph.recommend_pairings(ingredient, method=method, top_n=top_n)
    
    # Format for frontend
    formatted_results = [{
        "ingredient": r['ingredient'],
        "score": r['final_score'],
        "id": flavor_graph.get_ingredient_id(r['ingredient']),
        "compound_similarity": r['compound_similarity'],
        "recipe_score": r['recipe_score']
    } for r in results]
    
    return jsonify(formatted_results)

@app.route('/api/ingredients')
def get_ingredients():
    """Get all ingredients"""
    if flavor_graph is None:
        return jsonify([])
    
    ingredients = [{"id": node_id, "name": flavor_graph.node_id_to_name[node_id]} 
                   for node_id in list(flavor_graph.ingredient_ids)[:1000]]  # Limit for performance
    return jsonify(ingredients)

@app.route('/api/stats')
def get_stats():
    """Get graph statistics"""
    if flavor_graph is None:
        return jsonify({})
    
    return jsonify({
        "total_ingredients": len(flavor_graph.ingredient_ids),
        "total_compounds": len(flavor_graph.compound_ids),
        "total_edges": flavor_graph.ing_graph.number_of_edges()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
