# Docker Deployment Guide

This directory contains all Docker-related files for deploying the Python UV Template application.

## 📁 Files Overview

- **`Dockerfile`** - Production-ready multi-stage build
- **`Dockerfile.dev`** - Development environment with hot reload
- **`docker-compose.yml`** - Production deployment with optional services
- **`docker-compose.dev.yml`** - Development environment with databases
- **`.dockerignore`** - Files to exclude from Docker context
- **`docker-entrypoint.sh`** - Entrypoint script with health checks
- **`README.md`** - This documentation

## 🚀 Quick Start

### Development Environment

```bash
# Start development environment with hot reload
docker-compose -f deploy/docker-compose.dev.yml up --build

# Run with specific services
docker-compose -f deploy/docker-compose.dev.yml up app-dev postgres-dev

# Run in background
docker-compose -f deploy/docker-compose.dev.yml up -d
```

### Production Deployment

```bash
# Build and start production environment
docker-compose -f deploy/docker-compose.yml up --build -d

# View logs
docker-compose -f deploy/docker-compose.yml logs -f

# Stop services
docker-compose -f deploy/docker-compose.yml down
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application
ENVIRONMENT=production
DEBUG=false
PYTHONPATH=/app/src

# Database (optional)
DATABASE_URL=postgresql://user:password@postgres:5432/dbname
RUN_MIGRATIONS=true

# Static files (optional)
COLLECT_STATIC=true

# Security
SECRET_KEY=your-secret-key-here
```

### Docker Build Arguments

```bash
# Custom Python version
docker build --build-arg PYTHON_VERSION=3.13 -f deploy/Dockerfile .

# Custom uv version
docker build --build-arg UV_VERSION=latest -f deploy/Dockerfile .
```

## 📦 Docker Images

### Production Image Features

- ✅ Multi-stage build for smaller image size
- ✅ Non-root user for security
- ✅ Health checks included
- ✅ Optimized for production workloads
- ✅ Only production dependencies

### Development Image Features

- ✅ All development tools included
- ✅ Hot reload support
- ✅ Pre-commit hooks available
- ✅ Full development environment

## 🛠️ Advanced Usage

### Custom Dockerfile

```dockerfile
# Extend the base image
FROM python-uv-template:latest

# Add custom dependencies
COPY requirements-custom.txt .
RUN uv pip install -r requirements-custom.txt

# Add custom configuration
COPY custom-config.yml /app/config/
```

### Docker Compose Override

Create `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  app:
    environment:
      - CUSTOM_VAR=custom_value
    volumes:
      - ./custom-config:/app/config
```

### Health Checks

The containers include health checks:

```bash
# Check container health
docker ps
docker inspect --format='{{.State.Health.Status}}' container_name

# Custom health check
docker exec container_name python -c "
import requests
response = requests.get('http://localhost:8000/health')
exit(0 if response.status_code == 200 else 1)
"
```

## 🔍 Debugging

### Access Container Shell

```bash
# Production container
docker exec -it python-uv-template /bin/bash

# Development container
docker exec -it python-uv-template-dev /bin/bash
```

### View Logs

```bash
# All services
docker-compose -f deploy/docker-compose.yml logs

# Specific service
docker-compose -f deploy/docker-compose.yml logs app

# Follow logs
docker-compose -f deploy/docker-compose.yml logs -f app
```

### Debug Build Issues

```bash
# Build with no cache
docker-compose -f deploy/docker-compose.yml build --no-cache

# Build with verbose output
docker build --progress=plain -f deploy/Dockerfile .
```

## 🚀 Deployment Strategies

### Local Development

```bash
# Quick start
make docker-dev

# With specific services
docker-compose -f deploy/docker-compose.dev.yml up app-dev postgres-dev
```

### Staging Environment

```bash
# Use production compose with staging overrides
docker-compose -f deploy/docker-compose.yml -f deploy/docker-compose.staging.yml up -d
```

### Production Deployment

```bash
# Pull latest images
docker-compose -f deploy/docker-compose.yml pull

# Deploy with zero downtime
docker-compose -f deploy/docker-compose.yml up -d --no-deps app

# Scale services
docker-compose -f deploy/docker-compose.yml up -d --scale app=3
```

## 📊 Monitoring

### Container Metrics

```bash
# Resource usage
docker stats

# Container processes
docker exec python-uv-template ps aux

# Disk usage
docker system df
```

### Application Monitoring

Add monitoring tools to your compose file:

```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## 🔒 Security Best Practices

- ✅ Non-root user in containers
- ✅ Multi-stage builds to reduce attack surface
- ✅ No secrets in Dockerfiles
- ✅ Regular base image updates
- ✅ Health checks for reliability
- ✅ Resource limits in production

## 🧹 Cleanup

```bash
# Remove containers and networks
docker-compose -f deploy/docker-compose.yml down

# Remove containers, networks, and volumes
docker-compose -f deploy/docker-compose.yml down -v

# Remove images
docker-compose -f deploy/docker-compose.yml down --rmi all

# Clean up Docker system
docker system prune -a
```

## 📝 Customization

1. **Update container names** in compose files
2. **Modify port mappings** as needed
3. **Add environment-specific variables**
4. **Configure volume mounts** for persistence
5. **Add additional services** (Redis, Elasticsearch, etc.)

This Docker setup provides a complete containerization solution for development, testing, and production deployments.
