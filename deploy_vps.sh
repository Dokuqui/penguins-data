#!/bin/bash

echo "🐧 Deploying PenguinOps..."

docker-compose -f docker-compose.prod.yml down --remove-orphans

docker-compose -f docker-compose.prod.yml up -d --build

docker image prune -f

echo "✅ Deployment Started!"
echo "⏳ Note: Please wait 2-3 minutes for Spark training to finish."
