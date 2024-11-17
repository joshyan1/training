from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory storage for now
nodes = {}
master_ip = None

@app.route('/register', methods=['POST'])
def register_node():
    """Register a new node."""
    data = request.json
    global master_ip
    if "master_ip" in data:
        master_ip = data["master_ip"]
        print(f"Master registered with IP: {master_ip}")
    else:
        print(f"Worker registered: {data}")
    nodes[data["node_id"]] = data
    return jsonify({"message": f"Node {data['node_id']} registered successfully", "nodes": nodes}), 200

@app.route('/get_master_ip', methods=['GET'])
def get_master_ip():
    """Provide the master IP to workers."""
    if master_ip:
        return jsonify({"master_ip": master_ip}), 200
    else:
        return jsonify({"error": "Master IP not available"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11435)
