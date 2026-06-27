# Phase 6: Cloud Deployment Guide (AWS EKS)

This guide walks you through pushing your Dockerized RAG pipeline to a production-grade Kubernetes cluster on Amazon Web Services (AWS).

## Step 1: Provision the Infrastructure
Before you can run Kubernetes manifests, you need a cluster.

1. **Install tools:** Install the `aws-cli`, `eksctl`, and `kubectl` on your local machine.
2. **Configure AWS:** Run `aws configure` and enter your credentials.
3. **Spin up EKS Cluster:** 
   ```bash
   eksctl create cluster \
     --name prod-cluster \
     --region us-east-1 \
     --nodegroup-name standard-workers \
     --node-type t3.medium \
     --nodes 3 \
     --nodes-min 1 \
     --nodes-max 4
   ```
   *Note: This creates the underlying EC2 instances to host your RAG app.*

## Step 2: Push Docker Image to ECR
AWS needs a registry to store your Docker images before K8s can pull them.

1. **Create Repo:**
   ```bash
   aws ecr create-repository --repository-name enterprise-rag
   ```
2. **Build & Push (Local Method):**
   *(Note: This step is already automated by our GitHub Actions pipeline, but here is how to do it manually)*
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t enterprise-rag -f Dockerfile ..
   docker tag enterprise-rag:latest <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag:latest
   docker push <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag:latest
   ```

## Step 3: Deploy to Kubernetes
Now that the cluster is running and your image is in ECR, apply the configurations!

1. **Create Namespace:**
   ```bash
   kubectl create namespace rag-production
   ```
2. **Apply Manifests:**
   Run these from the `phase6_devops/k8s/` directory.
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f ingress.yaml
   ```

## Step 4: Verification & Scaling
Check if your pods are running:
```bash
kubectl get pods -n rag-production
```

**Need to handle more traffic?** Simply scale up the deployment:
```bash
kubectl scale deployment enterprise-rag-deployment --replicas=10 -n rag-production
```

You now have a production-grade RAG pipeline hosted in the cloud, capable of handling enterprise-level traffic!
