# Docker Basics Guide

## Author: Maria Lopez
## Topic: DevOps

Docker is an open-source platform for developing, shipping, and running applications in isolated environments called **containers**. Containers bundle an application with all its dependencies, ensuring consistent behavior across different environments.

## Why Docker?

- **Consistency**: "Works on my machine" problems disappear — containers run identically everywhere.
- **Isolation**: Each container runs independently, avoiding dependency conflicts.
- **Efficiency**: Containers share the host OS kernel, making them lighter than virtual machines.
- **Portability**: A Docker image can run on any machine with Docker installed.

## Key Concepts

### Images
A Docker image is a read-only template containing the application code, runtime, libraries, and configuration. Images are built from a `Dockerfile`.

### Containers
A container is a running instance of an image. Containers are isolated but share the host OS kernel.

### Dockerfile
A `Dockerfile` is a script that defines how to build an image.

```dockerfile
# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port and run
EXPOSE 8000
CMD ["python", "app.py"]
```

### Docker Hub
Docker Hub is a public registry where you can pull pre-built images (e.g., `nginx`, `postgres`, `redis`).

## Essential Docker Commands

```bash
# Build an image from Dockerfile
docker build -t my-app:1.0 .

# Run a container
docker run -d -p 8080:8000 --name my-container my-app:1.0

# List running containers
docker ps

# Stop a container
docker stop my-container

# Remove a container
docker rm my-container

# View logs
docker logs my-container

# Pull an image from Docker Hub
docker pull postgres:15

# List local images
docker images

# Remove an image
docker rmi my-app:1.0
```

## Docker Compose

Docker Compose lets you define multi-container applications in a single YAML file.

```yaml
version: "3.9"
services:
  web:
    build: .
    ports:
      - "8080:8000"
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
```

Run with: `docker compose up -d`

## Volumes and Networking

- **Volumes**: Persist data outside the container lifecycle (`docker volume create my-vol`)
- **Networks**: Containers in the same Compose file can communicate by service name

## Best Practices

1. Use slim base images (e.g., `python:3.11-slim`) to reduce image size.
2. Combine `RUN` commands to minimize image layers.
3. Never store secrets in Dockerfiles — use environment variables or secrets managers.
4. Use `.dockerignore` to exclude unnecessary files from the build context.
5. Tag images with specific versions, not just `latest`.
