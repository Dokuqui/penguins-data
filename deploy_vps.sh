#!/bin/bash
AZURE_IP="YOUR_VM_IP_HERE"
USER="azureuser"
KEY_PATH="~/.ssh/id_rsa" 

echo "🐧 Starting Deployment to Azure VM ($AZURE_IP)..."

echo "📦 Packaging project..."
tar --exclude='mongo_data' --exclude='cassandra_data' --exclude='__pycache__' -czf penguin_project.tar.gz work/ Dockerfile docker-compose.prod.yml
echo "🚀 Uploading to VPS..."
scp -i $KEY_PATH penguin_project.tar.gz $USER@$AZURE_IP:~/

echo "🔧 Configuring Remote Server..."
ssh -i $KEY_PATH $USER@$AZURE_IP << 'EOF'
    # A. Install Docker & Compose
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # B. Setup Project
    mkdir -p penguin_app
    tar -xzf penguin_project.tar.gz -C penguin_app
    cd penguin_app
    
    # C. Launch Containers
    echo "🔥 Launching Docker Stack..."
    sudo docker-compose -f docker-compose.prod.yml up -d --build
    
    # D. Cleanup
    rm ../penguin_project.tar.gz
EOF

echo "✅ DEPLOYMENT COMPLETE!"
echo "➡️  Dashboard: http://$AZURE_IP"
echo "➡️  API Docs:  http://$AZURE_IP:8000/docs"
