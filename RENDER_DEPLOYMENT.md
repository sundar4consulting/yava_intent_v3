# YAVA Intent Classifier V2 - Render Deployment Guide

Complete guide to deploy Elasticsearch-backed RAG+LLM intent classifier on Render.

---

## 📋 Prerequisites

### 1. Render Account
- Sign up at [render.com](https://render.com)
- Connect your GitHub repository (or use Render Git)

### 2. Elasticsearch Cluster
You need an accessible Elasticsearch cluster with:
- **URL**: Publicly accessible endpoint (e.g., Elastic Cloud, AWS OpenSearch)
- **Credentials**: Username/password (Basic Auth) or API key
- **Index**: `yava-intent-examples` populated with training data
- **Version**: Elasticsearch 8.x recommended

**Options for Elasticsearch:**
- **Elastic Cloud** (recommended): [cloud.elastic.co](https://cloud.elastic.co)
- **AWS OpenSearch**: Managed Elasticsearch on AWS
- **Self-hosted**: Ensure public URL or VPN/VPC connection

### 3. OpenAI API Key
- Get from [platform.openai.com](https://platform.openai.com)
- Required for LLM features (gpt-4o model)

---

## 🚀 Deployment Methods

### Method 1: Render Dashboard (Easiest)

#### Step 1: Create New Service
1. Log into Render Dashboard
2. Click **"New +"** → **"Background Worker"** (or **"Web Service"** if API)
3. Connect your Git repository

#### Step 2: Configure Service
**Service Settings:**
```
Name: yava-intent-classifier-v2
Environment: Python 3
Region: Oregon (or closest to your Elasticsearch cluster)
Branch: main
Build Command: pip install -r requirements.txt
Start Command: python -m src.classifier_v2
```

**For Web Service (REST API):**
```
Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 src.api:app
```

#### Step 3: Set Environment Variables
In **Environment** section, add:

```bash
# Elasticsearch Configuration
ELASTICSEARCH_HOST=https://your-cluster.es.elastic.cloud:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_elasticsearch_password  # Mark as secret
ELASTICSEARCH_INDEX=yava-intent-examples
ELASTICSEARCH_VERIFY_CERTS=true

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key  # Mark as secret
OPENAI_MODEL=gpt-4o
LLM_THRESHOLD=0.75
ENABLE_LLM=true

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
PYTHON_VERSION=3.11.0
```

**🔒 Security Note:** Mark `ELASTICSEARCH_PASSWORD` and `OPENAI_API_KEY` as **secrets** (click 🔒 icon)

#### Step 4: Deploy
1. Click **"Create Background Worker"** (or **"Create Web Service"**)
2. Render will build and deploy automatically
3. Monitor logs in **Logs** tab

---

### Method 2: Render Blueprint (Infrastructure as Code)

#### Step 1: Use render.yaml
The included `render.yaml` defines your infrastructure.

#### Step 2: Deploy via Dashboard
1. Go to **Render Dashboard** → **Blueprints**
2. Click **"New Blueprint Instance"**
3. Select repository with `render.yaml`
4. Render reads config and creates service automatically

#### Step 3: Set Secret Environment Variables
After creation, go to service settings and add:
- `ELASTICSEARCH_PASSWORD`
- `OPENAI_API_KEY`

---

### Method 3: Docker Deployment

#### Step 1: Build Docker Image
```bash
docker build -t yava-classifier-v2 .
```

#### Step 2: Push to Registry
```bash
# Render Container Registry (or Docker Hub)
docker tag yava-classifier-v2 registry.render.com/your-service/yava-classifier-v2
docker push registry.render.com/your-service/yava-classifier-v2
```

#### Step 3: Create Service on Render
1. **New +** → **Private Service**
2. Select **Docker**
3. Image URL: `registry.render.com/your-service/yava-classifier-v2`
4. Set environment variables (same as Method 1)

---

## 🔧 Configuration Details

### Elasticsearch Setup

#### Option A: Elastic Cloud (Recommended)
1. Create deployment at [cloud.elastic.co](https://cloud.elastic.co)
2. Get **Cloud ID** or **Elasticsearch endpoint**
3. Create user with `read/write` permissions for `yava-intent-examples` index
4. Set credentials in Render environment variables

**Example URL:**
```
https://my-deployment-abc123.es.us-east-1.aws.elastic.cloud:9200
```

#### Option B: AWS OpenSearch
1. Create OpenSearch domain in AWS
2. Configure **public access** or **VPN/VPC** connection
3. Create **master user** credentials
4. Set Fine-Grained Access Control (FGAC) permissions
5. Use endpoint in `ELASTICSEARCH_HOST`

**Example URL:**
```
https://search-my-domain-abc123.us-east-1.es.amazonaws.com
```

#### Option C: Self-Hosted Elasticsearch
- Ensure Elasticsearch is publicly accessible (or use Render Private Services + VPN)
- Configure SSL/TLS certificates
- Set `ELASTICSEARCH_VERIFY_CERTS=false` if using self-signed certs

### Populate Index with Training Data

Before deploying, your `yava-intent-examples` index must contain training examples.

**Create Population Script:**
```python
# scripts/populate_elasticsearch.py
import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Connect to ES
es = Elasticsearch(
    hosts=[os.getenv("ELASTICSEARCH_HOST")],
    basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
)

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example training data (expand with your 700 examples)
training_data = [
    {
        "intent_id": "INT-001",
        "intent_name": "pharmacy",
        "intent_category": "pharmacy",
        "example_utterance": "I need to refill my prescription",
        "metadata": {"lob": "Commercial", "priority": 2}
    },
    # Add 699 more examples...
]

# Index documents
for doc in training_data:
    # Generate embedding
    doc["embedding"] = model.encode(doc["example_utterance"]).tolist()
    doc["created_at"] = "2025-01-26T00:00:00Z"
    
    # Index document
    es.index(index="yava-intent-examples", document=doc)

print(f"✅ Indexed {len(training_data)} training examples")
```

**Run locally before deploying:**
```bash
python scripts/populate_elasticsearch.py
```

---

## 🧪 Testing Deployment

### Test Connection
```python
# test_render_deployment.py
import os
from src.classifier_v2 import get_hybrid_classifier_v2

# Initialize classifier (uses environment variables)
classifier = get_hybrid_classifier_v2()

# Test classification
result = classifier.classify(
    utterance="I need a prescription refill",
    session_id="test_001"
)

print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Method: {result['classification_method']}")
```

### Check Elasticsearch Connection
```bash
# In Render shell (Render Dashboard → Shell tab)
python -c "
from elasticsearch import Elasticsearch
import os
es = Elasticsearch(
    hosts=[os.getenv('ELASTICSEARCH_HOST')],
    basic_auth=(os.getenv('ELASTICSEARCH_USERNAME'), os.getenv('ELASTICSEARCH_PASSWORD'))
)
print(es.info())
"
```

### View Logs
Monitor Render logs for:
- ✅ Elasticsearch connection success
- ✅ OpenAI API key validation
- ✅ Index stats (document count)
- ❌ Any errors (auth failures, index missing, etc.)

---

## 📊 Service Types

### Option 1: Background Worker
**Use Case:** Asynchronous classification, batch processing, scheduled tasks

**Start Command:**
```bash
python -m src.classifier_v2
```

**Pros:**
- No HTTP overhead
- Can run continuously
- Good for event-driven architectures

**Cons:**
- No REST API
- Requires job queue integration (Celery, RQ, etc.)

---

### Option 2: Web Service (REST API)
**Use Case:** Real-time classification via HTTP endpoints

**Create API Wrapper (`src/api.py`):**
```python
from flask import Flask, request, jsonify
from src.classifier_v2 import get_hybrid_classifier_v2
import os

app = Flask(__name__)
classifier = get_hybrid_classifier_v2()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "yava-classifier-v2"})

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    result = classifier.classify(
        utterance=data.get('utterance'),
        session_id=data.get('session_id', 'default')
    )
    return jsonify(result)

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(classifier.get_metrics())

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
```

**Start Command:**
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 src.api:app
```

**Test API:**
```bash
curl -X POST https://your-service.onrender.com/classify \
  -H "Content-Type: application/json" \
  -d '{"utterance": "I need a prescription refill", "session_id": "user123"}'
```

---

## 💰 Cost Estimation

### Render Pricing
- **Starter Plan**: $7/month (512 MB RAM, 0.5 CPU)
- **Standard Plan**: $25/month (2 GB RAM, 1 CPU)
- **Pro Plan**: $85/month (4 GB RAM, 2 CPU)

**Recommendation:** Start with **Starter** for testing, upgrade to **Standard** for production.

### Elasticsearch Pricing
- **Elastic Cloud**: ~$45-95/month (8GB deployment)
- **AWS OpenSearch**: ~$50-100/month (t3.small.search)

### OpenAI Pricing
- **GPT-4o**: $5/1M input tokens, $15/1M output tokens
- **Estimated**: ~$0.02-0.05 per classification (if LLM invoked)
- **Monthly**: $50-200 depending on volume

**Total Monthly Cost:** ~$100-400 depending on scale

---

## 🔐 Security Best Practices

### 1. Secret Management
- ✅ Use Render **environment variables** (not hardcoded)
- ✅ Mark `ELASTICSEARCH_PASSWORD` and `OPENAI_API_KEY` as **secrets**
- ✅ Never commit `.env` to Git (already in `.gitignore`)

### 2. SSL/TLS
- ✅ Set `ELASTICSEARCH_VERIFY_CERTS=true` for production
- ✅ Use HTTPS for Elasticsearch endpoints
- ⚠️ Only set `verify_certs=false` for development/self-signed certs

### 3. Access Control
- ✅ Use least-privilege Elasticsearch user (not superuser)
- ✅ Restrict Elasticsearch user to `yava-intent-examples` index only
- ✅ Enable IP whitelisting on Elasticsearch cluster (if supported)

### 4. API Security (if Web Service)
- Add API key authentication to `/classify` endpoint
- Implement rate limiting (Flask-Limiter)
- Use CORS protection for browser clients

---

## 🐛 Troubleshooting

### Issue 1: Elasticsearch Connection Failed
**Error:** `❌ Elasticsearch connection failed: ConnectionError`

**Solutions:**
1. Check `ELASTICSEARCH_HOST` format: `https://host:9200`
2. Verify username/password in Render environment variables
3. Test connectivity: `curl -u elastic:password https://your-es-host.com:9200`
4. Check firewall: Ensure Elasticsearch allows Render IP ranges
5. For self-signed certs: Set `ELASTICSEARCH_VERIFY_CERTS=false`

### Issue 2: Index Not Found
**Error:** `index_not_found_exception`

**Solutions:**
1. Verify index exists: `GET /yava-intent-examples` in Kibana/DevTools
2. Create index: `python scripts/populate_elasticsearch.py`
3. Check index name spelling in environment variables

### Issue 3: OpenAI API Errors
**Error:** `❌ LLM Error: Incorrect API key`

**Solutions:**
1. Verify `OPENAI_API_KEY` starts with `sk-`
2. Check API key is active at [platform.openai.com](https://platform.openai.com)
3. Ensure sufficient credits in OpenAI account

### Issue 4: Out of Memory
**Error:** `Killed` or OOMKilled in logs

**Solutions:**
1. Upgrade to **Standard Plan** (2 GB RAM)
2. Reduce `--workers` in gunicorn (if web service)
3. Optimize model loading (lazy load sentence-transformers)

### Issue 5: Slow Classification
**Response time >5 seconds**

**Solutions:**
1. Check Elasticsearch cluster performance
2. Increase `num_candidates` in kNN search
3. Use Elasticsearch index caching
4. Consider deploying Elasticsearch in same region as Render service

---

## 📈 Monitoring & Observability

### Render Built-in Monitoring
- **Metrics**: CPU, Memory, Network in Render Dashboard
- **Logs**: Real-time logs in Logs tab
- **Alerts**: Set up email alerts for service failures

### Custom Metrics Endpoint
```python
# Add to src/api.py
@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify({
        **classifier.get_metrics(),
        "elasticsearch_stats": classifier.rag_classifier.vector_store.get_index_stats()
    })
```

### Elasticsearch Monitoring
- Use **Kibana** to monitor index health
- Track search latency in Elasticsearch Monitoring tab
- Set up alerts for disk space, memory usage

---

## 🔄 CI/CD Pipeline

### Auto-Deploy on Git Push
1. In Render service settings, enable **Auto-Deploy**
2. Push to `main` branch triggers deployment
3. Monitor build logs in Render Dashboard

### GitHub Actions (Optional)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Render
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Render
        run: curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK }}
```

---

## 📚 Additional Resources

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Elasticsearch Python Client**: [elastic.co/guide/en/elasticsearch/client/python-api](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.html)
- **OpenAI API**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Sentence Transformers**: [sbert.net](https://www.sbert.net)

---

## ✅ Deployment Checklist

- [ ] Elasticsearch cluster running and accessible
- [ ] `yava-intent-examples` index created and populated (700+ examples)
- [ ] OpenAI API key obtained and credits added
- [ ] Render account created and Git connected
- [ ] Environment variables set in Render Dashboard
- [ ] Secrets marked as encrypted (🔒 icon)
- [ ] Service deployed (Background Worker or Web Service)
- [ ] Logs checked for connection success
- [ ] Test classification performed successfully
- [ ] Metrics endpoint verified (if web service)
- [ ] Monitoring/alerts configured

---

## 🎉 Success!

Your YAVA Intent Classifier V2 is now running on Render with Elasticsearch backend!

**Next Steps:**
1. Integrate with Watson Orchestrator via API calls
2. Monitor performance and costs
3. Scale Elasticsearch cluster as needed
4. Optimize LLM threshold based on accuracy vs. cost tradeoff

**Need Help?**
- Render Support: [render.com/support](https://render.com/support)
- Elasticsearch Forums: [discuss.elastic.co](https://discuss.elastic.co)
