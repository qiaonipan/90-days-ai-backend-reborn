# Production-Grade AI Traffic Sentinel on AWS

Real-Time AI Traffic Monitoring and Anomaly Alerting System

An end-to-end production-grade project practicing distributed systems reliability engineering, cloud-native architecture, and GenAI RAG.

## Project Goals

Simulate real production traffic scenarios and implement:

- High-availability ingress (API Gateway + Lambda)
- Backpressure handling (SQS buffering)
- Auto-scaling workers (EKS + HPA)
- RAG root cause diagnosis (Bedrock + PostgreSQL Vector)
- Full observability (CloudWatch + OpenTelemetry)
- Infrastructure as Code (Terraform for fully automated deployment)

## Architecture Diagram (Text Version)

```
External Traffic
      ↓
API Gateway → Lambda (Canary via CodeDeploy)
      ↓
SQS Queue (Backpressure Buffer)
      ↓
EKS Cluster (Workers: FastAPI + Celery)
      ↓
→ Bedrock (LLM Diagnosis)
→ RDS PostgreSQL (Vector Store) + ElastiCache Redis (Semantic Cache)
→ CloudWatch + X-Ray (Metrics + Tracing)
```

## Tech Stack

- **IaC**: Terraform
- **Compute**: Amazon EKS (Kubernetes), AWS Lambda
- **Queue**: Amazon SQS
- **DB**: Amazon RDS PostgreSQL (Multi-AZ + Vector extension)
- **Cache**: Amazon ElastiCache Redis
- **AI**: AWS Bedrock (Claude / Llama)
- **Observability**: CloudWatch, X-Ray, OpenTelemetry
- **CI/CD**: GitHub Actions

## Directory Structure (Planned)

```
.
├── terraform/              # All AWS resources as IaC
│   ├── modules/
│   ├── main.tf
│   └── variables.tf
├── backend/                # FastAPI + Celery Worker
│   ├── app/
│   ├── worker.py
│   └── Dockerfile
├── frontend/               # Streamlit / React (optional)
├── scripts/                # Local testing scripts
└── README.md
```