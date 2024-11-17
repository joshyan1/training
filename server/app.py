from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory storage for now
nodes = {}

@app.route('/register', methods=['POST'])
def register_node():
    """Register a new compute node."""
    data = request.json
    node_id = data['node_id']
    nodes[node_id] = {
        "compute_power": data['compute_power'],
        "location": data['location'],
    }
    return jsonify({"message": f"Node {node_id} registered successfully", "nodes": nodes}), 200

@app.route('/nodes', methods=['GET'])
def get_nodes():
    """Get all registered nodes."""
    return jsonify(nodes), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
