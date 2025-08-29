# Synesthesia AI Operations Runbook

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Troubleshooting](#troubleshooting)
6. [Incident Response](#incident-response)
7. [Maintenance](#maintenance)
8. [Security](#security)
9. [Backup & Recovery](#backup--recovery)
10. [Performance Tuning](#performance-tuning)

## System Overview

Synesthesia AI is a multi-modal content generation platform that creates aligned text, visual, and audio content using advanced AI models. The system is built as a microservices architecture with the following key components:

- **API Gateway**: FastAPI-based REST API
- **Web Frontend**: Next.js React application
- **Worker Services**: Multi-agent content generation
- **Data Stores**: PostgreSQL, Redis, OpenSearch
- **Message Queue**: NATS JetStream
- **Monitoring**: Prometheus, Grafana, Sentry

## Architecture

### High-Level Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Web App   │    │  API Gateway │    │   Workers   │
│  (Next.js)  │◄──►│  (FastAPI)   │◄──►│ (Multi-Agent)│
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ PostgreSQL  │    │    Redis    │    │ OpenSearch  │
│ (Primary DB)│    │   (Cache)   │    │  (Search)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Service Dependencies

- **API** depends on: PostgreSQL, Redis, OpenSearch, NATS
- **Web** depends on: API
- **Workers** depend on: PostgreSQL, Redis, NATS, External APIs (OpenAI)
- **All services** depend on: Monitoring stack

## Deployment

### Environments

- **Development**: Local Docker Compose
- **Staging**: Docker Compose with staging configuration
- **Production**: Docker Swarm or Kubernetes with HA setup

### Deployment Process

#### Staging Deployment

```bash
# Deploy to staging
./scripts/deploy.sh -e staging deploy

# Check status
./scripts/deploy.sh -e staging status

# View logs
./scripts/deploy.sh -e staging logs
```

#### Production Deployment

```bash
# Canary deployment (10% traffic)
./scripts/deploy.sh -e production -c 10 canary

# Monitor canary for 10 minutes, then promote
./scripts/deploy.sh -e production promote

# Or rollback if issues
./scripts/deploy.sh -e production rollback
```

### Environment Variables

#### Required for All Environments

```bash
# Database
POSTGRES_PASSWORD=<secure_password>
DATABASE_URL=postgresql://synesthesia:${POSTGRES_PASSWORD}@postgres:5432/synesthesia

# Redis
REDIS_PASSWORD=<secure_password>
REDIS_URL=redis://redis:6379/0

# API Keys
OPENAI_API_KEY=<openai_api_key>
JWT_SECRET=<jwt_secret>

# AWS (for S3 storage)
AWS_ACCESS_KEY_ID=<aws_access_key>
AWS_SECRET_ACCESS_KEY=<aws_secret_key>
S3_BUCKET=<s3_bucket_name>

# Monitoring
SENTRY_DSN=<sentry_dsn>
```

#### Production-Specific

```bash
# SSL Certificates
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Backup
S3_BACKUP_BUCKET=<backup_bucket>

# Monitoring
OTEL_ENDPOINT=<otel_collector_endpoint>
OTEL_API_KEY=<otel_api_key>
```

## Monitoring

### Key Metrics

#### System Health
- **API Response Time**: < 500ms (95th percentile)
- **Success Rate**: > 99.5%
- **CPU Usage**: < 80%
- **Memory Usage**: < 85%
- **Disk Usage**: < 90%

#### Business Metrics
- **Story Pack Generation Rate**: Requests per minute
- **Generation Success Rate**: > 95%
- **Average Generation Time**: < 45 seconds
- **User Satisfaction Score**: > 4.5/5

### Monitoring Stack

#### Prometheus Metrics
- **API Metrics**: `/metrics` endpoint
- **System Metrics**: Node Exporter
- **Database Metrics**: PostgreSQL Exporter
- **Redis Metrics**: Redis Exporter

#### Grafana Dashboards
- **System Overview**: High-level health metrics
- **API Performance**: Request rates, response times, errors
- **Infrastructure**: CPU, memory, disk, network
- **Business Metrics**: Generation rates, success rates

#### Alerting Rules

```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: High error rate detected

# High response time
- alert: HighResponseTime
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: High response time detected
```

### Log Management

#### Log Locations
- **API Logs**: `/var/log/synesthesia/api/`
- **Worker Logs**: `/var/log/synesthesia/workers/`
- **Nginx Logs**: `/var/log/nginx/`
- **System Logs**: `/var/log/syslog`

#### Log Levels
- **ERROR**: System errors, exceptions
- **WARN**: Performance issues, recoverable errors
- **INFO**: Normal operations, user actions
- **DEBUG**: Detailed debugging information

## Troubleshooting

### Common Issues

#### API Not Responding

**Symptoms**: HTTP 502/503 errors, timeouts

**Diagnosis**:
```bash
# Check API container status
docker-compose ps api

# Check API logs
docker-compose logs api

# Check health endpoint
curl http://localhost:8000/v1/health
```

**Solutions**:
1. Restart API service: `docker-compose restart api`
2. Check database connectivity
3. Verify environment variables
4. Check resource usage (CPU/Memory)

#### Database Connection Issues

**Symptoms**: Database connection errors in logs

**Diagnosis**:
```bash
# Check PostgreSQL status
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U synesthesia -d synesthesia -c "SELECT 1;"
```

**Solutions**:
1. Restart PostgreSQL: `docker-compose restart postgres`
2. Check disk space
3. Verify credentials
4. Check connection pool settings

#### Worker Queue Backlog

**Symptoms**: Slow story pack generation, queue growing

**Diagnosis**:
```bash
# Check worker status
docker-compose ps | grep worker

# Check NATS queue status
docker-compose exec nats nats stream info

# Check worker logs
docker-compose logs narrative-worker
```

**Solutions**:
1. Scale up workers: `docker-compose up -d --scale narrative-worker=4`
2. Check external API rate limits
3. Optimize worker performance
4. Clear stuck jobs

#### High Memory Usage

**Symptoms**: OOM kills, slow performance

**Diagnosis**:
```bash
# Check memory usage
docker stats

# Check system memory
free -h

# Check for memory leaks in logs
grep -i "memory\|oom" /var/log/syslog
```

**Solutions**:
1. Restart affected services
2. Increase memory limits
3. Optimize application code
4. Add swap space (temporary)

### Performance Issues

#### Slow API Responses

**Investigation Steps**:
1. Check database query performance
2. Verify Redis cache hit rates
3. Check external API response times
4. Review application profiling data

**Optimization Actions**:
1. Add database indexes
2. Implement query optimization
3. Increase cache TTL
4. Add request caching

#### Database Performance

**Common Queries**:
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check database connections
SELECT count(*) FROM pg_stat_activity;

-- Check table sizes
SELECT schemaname,tablename,attname,n_distinct,correlation 
FROM pg_stats;
```

## Incident Response

### Severity Levels

#### P0 - Critical
- **Definition**: Complete service outage
- **Response Time**: 15 minutes
- **Examples**: API completely down, data loss

#### P1 - High
- **Definition**: Major functionality impaired
- **Response Time**: 1 hour
- **Examples**: Story generation failing, authentication issues

#### P2 - Medium
- **Definition**: Minor functionality impaired
- **Response Time**: 4 hours
- **Examples**: Slow responses, non-critical features down

#### P3 - Low
- **Definition**: Minor issues, cosmetic problems
- **Response Time**: 24 hours
- **Examples**: UI glitches, documentation errors

### Incident Response Process

1. **Detection**: Automated alerts or user reports
2. **Assessment**: Determine severity and impact
3. **Response**: Assign incident commander and team
4. **Communication**: Update status page and stakeholders
5. **Resolution**: Implement fix and verify
6. **Post-mortem**: Document lessons learned

### Emergency Contacts

```
Incident Commander: [Primary On-Call]
Engineering Lead: [Engineering Manager]
DevOps Lead: [DevOps Engineer]
Product Owner: [Product Manager]
```

### Rollback Procedures

#### API Rollback
```bash
# Quick rollback to previous version
./scripts/deploy.sh -e production rollback

# Manual rollback
export VERSION=<previous_version>
docker-compose -f docker-compose.prod.yml up -d api
```

#### Database Rollback
```bash
# Rollback migrations (use with caution)
docker-compose exec api alembic downgrade -1

# Restore from backup
./scripts/restore-db.sh <backup_timestamp>
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- [ ] Check system health dashboards
- [ ] Review error logs
- [ ] Verify backup completion
- [ ] Monitor resource usage

#### Weekly
- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Clean up old logs
- [ ] Test backup restoration

#### Monthly
- [ ] Security audit
- [ ] Performance review
- [ ] Capacity planning
- [ ] Update documentation

### Maintenance Windows

**Staging**: Anytime (with notification)
**Production**: 
- **Preferred**: Sunday 2:00-4:00 AM UTC
- **Emergency**: Anytime with approval

### Update Procedures

#### Security Updates
```bash
# Update base images
docker pull postgres:16
docker pull redis:7-alpine
docker pull nginx:1.25-alpine

# Rebuild and deploy
./scripts/deploy.sh -e staging build
./scripts/deploy.sh -e staging deploy
```

#### Application Updates
```bash
# Deploy to staging first
./scripts/deploy.sh -e staging -v v1.2.3 deploy

# Run integration tests
npm run test:integration

# Deploy to production with canary
./scripts/deploy.sh -e production -v v1.2.3 canary
```

## Security

### Security Monitoring

#### Key Security Metrics
- **Failed authentication attempts**: < 100/hour
- **Rate limit violations**: < 1000/hour
- **Suspicious IP activity**: Monitor and block
- **SSL certificate expiry**: > 30 days remaining

#### Security Alerts
- **Brute force attacks**: > 50 failed logins from single IP
- **SQL injection attempts**: Detected in logs
- **Unusual API usage patterns**: Anomaly detection
- **Certificate expiry warnings**: < 30 days

### Security Procedures

#### Incident Response
1. **Isolate**: Block malicious IPs
2. **Assess**: Determine scope of breach
3. **Contain**: Limit further damage
4. **Eradicate**: Remove threats
5. **Recover**: Restore normal operations
6. **Learn**: Update security measures

#### Access Management
```bash
# Add user to security group
usermod -a -G security <username>

# Rotate API keys (quarterly)
./scripts/rotate-keys.sh

# Update SSL certificates
./scripts/update-ssl.sh
```

## Backup & Recovery

### Backup Strategy

#### Database Backups
- **Frequency**: Every 6 hours
- **Retention**: 30 days local, 1 year S3
- **Type**: Full backup + WAL archiving

```bash
# Manual backup
./scripts/backup-db.sh

# Restore from backup
./scripts/restore-db.sh 2024-01-15_14-30-00
```

#### File Backups
- **User uploads**: Real-time to S3
- **Generated content**: Real-time to S3
- **Configuration files**: Daily to S3

#### Recovery Procedures

##### Database Recovery
```bash
# Stop services
docker-compose stop api workers

# Restore database
./scripts/restore-db.sh <timestamp>

# Start services
docker-compose start api workers

# Verify integrity
./scripts/verify-db.sh
```

##### Complete System Recovery
```bash
# Restore from infrastructure backup
terraform apply -var="restore_from_backup=true"

# Deploy application
./scripts/deploy.sh -e production deploy

# Restore data
./scripts/restore-all-data.sh <backup_date>
```

### Disaster Recovery

#### RTO/RPO Targets
- **RTO** (Recovery Time Objective): 4 hours
- **RPO** (Recovery Point Objective): 1 hour

#### DR Procedures
1. **Assess damage**: Determine scope of disaster
2. **Activate DR site**: Spin up backup infrastructure
3. **Restore data**: From latest backups
4. **Redirect traffic**: Update DNS to DR site
5. **Verify functionality**: Run smoke tests
6. **Communicate**: Update stakeholders

## Performance Tuning

### Database Optimization

#### PostgreSQL Tuning
```sql
-- Connection pooling
max_connections = 200
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 16MB

-- Query optimization
enable_seqscan = off  -- For specific queries
random_page_cost = 1.1  -- For SSD storage
```

#### Index Optimization
```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_story_packs_created_at 
ON story_packs(created_at DESC);

CREATE INDEX CONCURRENTLY idx_story_packs_status 
ON story_packs(status) WHERE status IN ('pending', 'processing');
```

### Application Optimization

#### API Performance
- **Connection pooling**: 20 connections per worker
- **Request timeout**: 30 seconds
- **Cache TTL**: 5 minutes for static data
- **Rate limiting**: 100 requests/minute per user

#### Worker Optimization
- **Concurrency**: 4 workers per service type
- **Batch processing**: Process 5 jobs at once
- **Circuit breaker**: Fail fast on external API errors
- **Retry logic**: Exponential backoff with jitter

### Monitoring Performance

#### Key Performance Indicators
```bash
# API response time
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/v1/health"

# Database query time
SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 5;

# Redis performance
redis-cli --latency-history -i 1

# Worker queue depth
nats stream info STORY_GENERATION
```

---

## Emergency Procedures

### Complete System Failure

1. **Immediate Actions** (0-15 minutes):
   - Activate incident response team
   - Check infrastructure status
   - Enable maintenance page
   - Notify stakeholders

2. **Assessment** (15-30 minutes):
   - Determine root cause
   - Assess data integrity
   - Estimate recovery time
   - Plan recovery strategy

3. **Recovery** (30+ minutes):
   - Restore from backups if needed
   - Redeploy services
   - Verify functionality
   - Gradually restore traffic

### Data Corruption

1. **Stop all writes** to affected systems
2. **Assess scope** of corruption
3. **Restore from last known good backup**
4. **Replay transactions** if possible
5. **Verify data integrity**
6. **Resume normal operations**

### Security Breach

1. **Isolate** affected systems
2. **Preserve** evidence
3. **Assess** scope of breach
4. **Notify** relevant authorities
5. **Remediate** vulnerabilities
6. **Monitor** for further activity

---

*This runbook should be reviewed and updated quarterly. Last updated: [Current Date]*
