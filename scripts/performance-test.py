#!/usr/bin/env python3
"""
Performance testing and chaos engineering for Synesthesia AI
Comprehensive load testing, stress testing, and failure simulation
"""

import asyncio
import aiohttp
import time
import json
import random
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import psutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Performance test configuration."""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 10
    requests_per_user: int = 20
    ramp_up_time: int = 30  # seconds
    test_duration: int = 300  # seconds
    think_time_min: float = 1.0  # seconds
    think_time_max: float = 3.0  # seconds
    timeout: float = 30.0  # seconds
    auth_token: Optional[str] = None


@dataclass
class RequestResult:
    """Individual request result."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TestResults:
    """Aggregated test results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    errors: Dict[str, int]
    test_duration: float
    concurrent_users: int


class PerformanceTester:
    """Performance testing orchestrator."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Test scenarios
        self.scenarios = [
            self._health_check_scenario,
            self._auth_scenario,
            self._project_list_scenario,
            self._story_pack_generation_scenario,
            self._brand_kit_scenario,
            self._reports_scenario,
        ]
    
    async def run_performance_test(self) -> TestResults:
        """Run comprehensive performance test."""
        logger.info(f"Starting performance test with {self.config.concurrent_users} users")
        logger.info(f"Test duration: {self.config.test_duration}s")
        
        self.start_time = datetime.now()
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        
        # Create user tasks
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(
                self._simulate_user(user_id, semaphore)
            )
            tasks.append(task)
            
            # Ramp up gradually
            if self.config.ramp_up_time > 0:
                await asyncio.sleep(self.config.ramp_up_time / self.config.concurrent_users)
        
        # Wait for all users to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = datetime.now()
        
        return self._calculate_results()
    
    async def _simulate_user(self, user_id: int, semaphore: asyncio.Semaphore):
        """Simulate a single user's behavior."""
        async with semaphore:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            ) as session:
                
                test_end_time = self.start_time + timedelta(seconds=self.config.test_duration)
                
                while datetime.now() < test_end_time:
                    # Select random scenario
                    scenario = random.choice(self.scenarios)
                    
                    try:
                        await scenario(session, user_id)
                    except Exception as e:
                        logger.error(f"User {user_id} scenario failed: {e}")
                        self.results.append(RequestResult(
                            endpoint="scenario_error",
                            method="ANY",
                            status_code=0,
                            response_time=0.0,
                            success=False,
                            error=str(e)
                        ))
                    
                    # Think time between requests
                    think_time = random.uniform(
                        self.config.think_time_min,
                        self.config.think_time_max
                    )
                    await asyncio.sleep(think_time)
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        endpoint: str,
        **kwargs
    ) -> RequestResult:
        """Make HTTP request and record metrics."""
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        # Add auth header if available
        headers = kwargs.get('headers', {})
        if self.config.auth_token:
            headers['Authorization'] = f'Bearer {self.config.auth_token}'
            kwargs['headers'] = headers
        
        try:
            async with session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                
                # Read response to ensure complete request
                await response.read()
                
                result = RequestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=200 <= response.status < 400
                )
                
                self.results.append(result)
                return result
                
        except Exception as e:
            response_time = time.time() - start_time
            result = RequestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
            
            self.results.append(result)
            return result
    
    # Test scenarios
    
    async def _health_check_scenario(self, session: aiohttp.ClientSession, user_id: int):
        """Health check scenario."""
        await self._make_request(session, 'GET', '/v1/health')
    
    async def _auth_scenario(self, session: aiohttp.ClientSession, user_id: int):
        """Authentication scenario."""
        # Register user
        user_data = {
            'email': f'testuser{user_id}_{int(time.time())}@example.com',
            'password': 'testpassword123',
            'full_name': f'Test User {user_id}'
        }
        
        await self._make_request(
            session, 'POST', '/v1/auth/register',
            json=user_data
        )
        
        # Login
        login_data = {
            'email': user_data['email'],
            'password': user_data['password']
        }
        
        await self._make_request(
            session, 'POST', '/v1/auth/login',
            json=login_data
        )
    
    async def _project_list_scenario(self, session: aiohttp.ClientSession, user_id: int):
        """Project listing scenario."""
        await self._make_request(session, 'GET', '/v1/projects/')
    
    async def _story_pack_generation_scenario(self, session: aiohttp.ClientSession, user_id: int):
        """Story pack generation scenario."""
        story_data = {
            'name': f'Performance Test Story {user_id}',
            'prompt': 'Create a short story about technology and innovation',
            'project_id': 'test-project-id'
        }
        
        await self._make_request(
            session, 'POST', '/v1/storypacks/generate',
            json=story_data
        )
        
        # List story packs
        await self._make_request(session, 'GET', '/v1/storypacks/')
    
    async def _brand_kit_scenario(self, session: aiohttp.ClientSession, user_id: int):
        """Brand kit scenario."""
        brand_data = {
            'name': f'Performance Test Brand {user_id}',
            'project_id': 'test-project-id',
            'color_palette': {
                'primary': '#3B82F6',
                'secondary': '#6B7280'
            }
        }
        
        await self._make_request(
            session, 'POST', '/v1/brand-kits/',
            json=brand_data
        )
    
    async def _reports_scenario(self, session: aiohttp.ClientSession, user_id: int):
        """Reports scenario."""
        await self._make_request(session, 'GET', '/v1/reports/dashboard-metrics')
    
    def _calculate_results(self) -> TestResults:
        """Calculate aggregated test results."""
        if not self.results:
            return TestResults(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0.0,
                avg_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                p50_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                requests_per_second=0.0,
                errors={},
                test_duration=0.0,
                concurrent_users=self.config.concurrent_users
            )
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time for r in self.results]
        response_times.sort()
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]
        
        # Count errors
        errors = {}
        for result in failed_results:
            error_key = f"{result.status_code}: {result.error or 'Unknown error'}"
            errors[error_key] = errors.get(error_key, 0) + 1
        
        # Calculate test duration
        test_duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        return TestResults(
            total_requests=len(self.results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            success_rate=len(successful_results) / len(self.results) * 100,
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            min_response_time=min(response_times) if response_times else 0.0,
            max_response_time=max(response_times) if response_times else 0.0,
            p50_response_time=percentile(response_times, 50),
            p95_response_time=percentile(response_times, 95),
            p99_response_time=percentile(response_times, 99),
            requests_per_second=len(self.results) / test_duration if test_duration > 0 else 0.0,
            errors=errors,
            test_duration=test_duration,
            concurrent_users=self.config.concurrent_users
        )


class ChaosEngineer:
    """Chaos engineering for failure simulation."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.chaos_scenarios = [
            self._network_latency_chaos,
            self._service_unavailable_chaos,
            self._database_connection_chaos,
            self._memory_pressure_chaos,
            self._cpu_spike_chaos,
        ]
    
    async def run_chaos_test(self, duration: int = 300):
        """Run chaos engineering tests."""
        logger.info(f"Starting chaos engineering test for {duration} seconds")
        
        end_time = datetime.now() + timedelta(seconds=duration)
        
        while datetime.now() < end_time:
            # Select random chaos scenario
            scenario = random.choice(self.chaos_scenarios)
            
            try:
                await scenario()
            except Exception as e:
                logger.error(f"Chaos scenario failed: {e}")
            
            # Wait before next chaos event
            await asyncio.sleep(random.uniform(30, 120))
    
    async def _network_latency_chaos(self):
        """Simulate network latency issues."""
        logger.info("Simulating network latency chaos")
        
        # In a real implementation, this would:
        # - Add artificial delays to network requests
        # - Use tools like tc (traffic control) on Linux
        # - Simulate packet loss
        
        # For demo, we'll just log the simulation
        await asyncio.sleep(random.uniform(5, 15))
        logger.info("Network latency chaos completed")
    
    async def _service_unavailable_chaos(self):
        """Simulate service unavailability."""
        logger.info("Simulating service unavailable chaos")
        
        # In a real implementation, this would:
        # - Temporarily stop services
        # - Block specific endpoints
        # - Return 503 errors
        
        await asyncio.sleep(random.uniform(10, 30))
        logger.info("Service unavailable chaos completed")
    
    async def _database_connection_chaos(self):
        """Simulate database connection issues."""
        logger.info("Simulating database connection chaos")
        
        # In a real implementation, this would:
        # - Close database connections
        # - Simulate connection timeouts
        # - Create connection pool exhaustion
        
        await asyncio.sleep(random.uniform(5, 20))
        logger.info("Database connection chaos completed")
    
    async def _memory_pressure_chaos(self):
        """Simulate memory pressure."""
        logger.info("Simulating memory pressure chaos")
        
        # In a real implementation, this would:
        # - Allocate large amounts of memory
        # - Trigger garbage collection
        # - Simulate memory leaks
        
        await asyncio.sleep(random.uniform(15, 45))
        logger.info("Memory pressure chaos completed")
    
    async def _cpu_spike_chaos(self):
        """Simulate CPU spikes."""
        logger.info("Simulating CPU spike chaos")
        
        # In a real implementation, this would:
        # - Create CPU-intensive tasks
        # - Spawn multiple threads
        # - Simulate high CPU usage
        
        await asyncio.sleep(random.uniform(10, 30))
        logger.info("CPU spike chaos completed")


class SystemMonitor:
    """System resource monitoring during tests."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.monitoring = False
    
    async def start_monitoring(self, interval: float = 5.0):
        """Start system monitoring."""
        self.monitoring = True
        
        while self.monitoring:
            try:
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
                }
                
                self.metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': statistics.mean(memory_values),
            'max_memory_percent': max(memory_values),
            'sample_count': len(self.metrics),
            'monitoring_duration': len(self.metrics) * 5  # Assuming 5s intervals
        }


def print_results(results: TestResults, system_summary: Dict[str, Any]):
    """Print formatted test results."""
    print("\n" + "="*80)
    print("PERFORMANCE TEST RESULTS")
    print("="*80)
    
    print(f"\nTest Configuration:")
    print(f"  Concurrent Users: {results.concurrent_users}")
    print(f"  Test Duration: {results.test_duration:.1f}s")
    print(f"  Total Requests: {results.total_requests}")
    
    print(f"\nSuccess Metrics:")
    print(f"  Successful Requests: {results.successful_requests}")
    print(f"  Failed Requests: {results.failed_requests}")
    print(f"  Success Rate: {results.success_rate:.2f}%")
    print(f"  Requests/Second: {results.requests_per_second:.2f}")
    
    print(f"\nResponse Time Metrics:")
    print(f"  Average: {results.avg_response_time*1000:.1f}ms")
    print(f"  Minimum: {results.min_response_time*1000:.1f}ms")
    print(f"  Maximum: {results.max_response_time*1000:.1f}ms")
    print(f"  50th Percentile: {results.p50_response_time*1000:.1f}ms")
    print(f"  95th Percentile: {results.p95_response_time*1000:.1f}ms")
    print(f"  99th Percentile: {results.p99_response_time*1000:.1f}ms")
    
    if results.errors:
        print(f"\nErrors:")
        for error, count in results.errors.items():
            print(f"  {error}: {count}")
    
    if system_summary:
        print(f"\nSystem Resource Usage:")
        print(f"  Average CPU: {system_summary.get('avg_cpu_percent', 0):.1f}%")
        print(f"  Peak CPU: {system_summary.get('max_cpu_percent', 0):.1f}%")
        print(f"  Average Memory: {system_summary.get('avg_memory_percent', 0):.1f}%")
        print(f"  Peak Memory: {system_summary.get('max_memory_percent', 0):.1f}%")
    
    print("\n" + "="*80)


async def main():
    """Main performance testing function."""
    parser = argparse.ArgumentParser(description='Synesthesia AI Performance Testing')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL for testing')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--requests', type=int, default=20, help='Requests per user')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--ramp-up', type=int, default=30, help='Ramp up time in seconds')
    parser.add_argument('--auth-token', help='Authentication token')
    parser.add_argument('--chaos', action='store_true', help='Enable chaos engineering')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestConfig(
        base_url=args.url,
        concurrent_users=args.users,
        requests_per_user=args.requests,
        test_duration=args.duration,
        ramp_up_time=args.ramp_up,
        auth_token=args.auth_token
    )
    
    # Initialize components
    tester = PerformanceTester(config)
    monitor = SystemMonitor()
    
    # Start system monitoring
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    
    # Start chaos engineering if enabled
    chaos_task = None
    if args.chaos:
        chaos_engineer = ChaosEngineer(args.url)
        chaos_task = asyncio.create_task(chaos_engineer.run_chaos_test(args.duration))
    
    try:
        # Run performance test
        results = await tester.run_performance_test()
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        await asyncio.sleep(1)  # Allow final metrics collection
        monitor_task.cancel()
        
        if chaos_task:
            chaos_task.cancel()
    
    # Get system summary
    system_summary = monitor.get_summary()
    
    # Print results
    print_results(results, system_summary)
    
    # Save results to file if requested
    if args.output:
        output_data = {
            'test_results': asdict(results),
            'system_summary': system_summary,
            'test_config': asdict(config),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    # Exit with appropriate code
    if results.success_rate < 95.0:
        print("\nWARNING: Success rate below 95%")
        sys.exit(1)
    
    if results.p95_response_time > 5.0:  # 5 seconds
        print("\nWARNING: 95th percentile response time above 5 seconds")
        sys.exit(1)
    
    print("\nPerformance test completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
