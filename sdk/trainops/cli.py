"""
TrainOps CLI - Command-line interface for viewing training runs
"""
import sys
import argparse
from datetime import datetime
from typing import Optional
import requests
from tabulate import tabulate


class TrainOpsCLI:
    """CLI for TrainOps Observatory"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url.rstrip('/')
    
    def _get(self, endpoint: str, params: Optional[dict] = None):
        """Make GET request to API"""
        try:
            response = requests.get(f"{self.api_url}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: Could not connect to TrainOps API at {self.api_url}")
            print(f"Details: {e}")
            print("\nMake sure the backend is running:")
            print("  docker-compose up -d backend")
            sys.exit(1)
    
    def _format_duration(self, seconds: Optional[float]) -> str:
        """Format duration in human-readable form"""
        if not seconds:
            return "N/A"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def _format_timestamp(self, timestamp: Optional[str]) -> str:
        """Format ISO timestamp"""
        if not timestamp:
            return "N/A"
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return timestamp
    
    def list_runs(self, project: Optional[str] = None, status: Optional[str] = None, limit: int = 20):
        """List training runs"""
        params = {'limit': limit}
        if project:
            params['project'] = project
        if status:
            params['status'] = status
        
        data = self._get('/api/runs', params=params)
        runs = data.get('runs', [])
        
        if not runs:
            print("No runs found.")
            return
        
        # Prepare table data
        table_data = []
        for run in runs:
            # Calculate duration
            started = run.get('started_at')
            ended = run.get('ended_at')
            if started:
                start_dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
                if ended:
                    end_dt = datetime.fromisoformat(ended.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                else:
                    duration = (datetime.now() - start_dt).total_seconds()
            else:
                duration = None
            
            table_data.append([
                run['id'][:8],  # Short ID
                run['name'][:30],  # Truncate long names
                run['project'],
                run['status'],
                run.get('instance_type', 'N/A'),
                self._format_duration(duration),
                self._format_timestamp(started),
            ])
        
        headers = ['ID', 'Name', 'Project', 'Status', 'Instance', 'Duration', 'Started']
        print(tabulate(table_data, headers=headers, tablefmt='simple'))
        print(f"\nTotal: {len(runs)} runs")
    
    def show_run(self, run_id: str):
        """Show detailed information about a run"""
        data = self._get(f'/api/runs/{run_id}')
        
        print("\n" + "="*70)
        print(f"Run: {data['name']}")
        print("="*70)
        
        # Basic info
        print(f"\nID:              {data['id']}")
        print(f"Project:         {data['project']}")
        print(f"Status:          {data['status']}")
        print(f"Instance Type:   {data.get('instance_type', 'N/A')}")
        
        # Timing
        started = data.get('started_at')
        ended = data.get('ended_at')
        if started:
            print(f"Started:         {self._format_timestamp(started)}")
        if ended:
            print(f"Ended:           {self._format_timestamp(ended)}")
        
        # Duration
        if started:
            start_dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
            if ended:
                end_dt = datetime.fromisoformat(ended.replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds()
            else:
                duration = (datetime.now() - start_dt).total_seconds()
            print(f"Duration:        {self._format_duration(duration)}")
        
        # Summary statistics
        summary = data.get('summary', {})
        if summary:
            print("\n" + "-"*70)
            print("Performance Summary")
            print("-"*70)
            
            if summary.get('avg_gpu_util') is not None:
                print(f"Avg GPU Util:    {summary['avg_gpu_util']:.1f}%")
            if summary.get('avg_cpu_util') is not None:
                print(f"Avg CPU Util:    {summary['avg_cpu_util']:.1f}%")
            if summary.get('avg_throughput') is not None:
                print(f"Avg Throughput:  {summary['avg_throughput']:.1f} samples/sec")
            if summary.get('max_gpu_memory') is not None:
                print(f"Max GPU Memory:  {summary['max_gpu_memory']:.1f}%")
            print(f"Metrics Count:   {summary.get('metric_count', 0)}")
        
        # Tags
        if data.get('tags'):
            print("\n" + "-"*70)
            print("Tags")
            print("-"*70)
            for key, value in data['tags'].items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        print(f"\nView metrics: trainops runs metrics {run_id}")
        print(f"View in dashboard: http://localhost:3000/runs/{run_id}")
        print()
    
    def show_metrics(self, run_id: str, limit: int = 50, tail: bool = False):
        """Show metrics for a run"""
        data = self._get(f'/api/runs/{run_id}/metrics', params={'limit': limit})
        metrics = data.get('metrics', [])
        
        if not metrics:
            print(f"No metrics found for run {run_id}")
            return
        
        # Reverse if showing tail
        if tail:
            metrics = list(reversed(metrics[-limit:]))
        
        # Prepare table
        table_data = []
        for m in metrics:
            table_data.append([
                self._format_timestamp(m['timestamp']),
                f"{m.get('gpu_util', 0):.1f}%" if m.get('gpu_util') else "N/A",
                f"{m.get('cpu_util', 0):.1f}%" if m.get('cpu_util') else "N/A",
                f"{m.get('throughput', 0):.1f}" if m.get('throughput') else "N/A",
                f"{m.get('gpu_memory_util', 0):.1f}%" if m.get('gpu_memory_util') else "N/A",
            ])
        
        headers = ['Timestamp', 'GPU%', 'CPU%', 'Throughput', 'GPU Mem%']
        print(f"\nMetrics for run {run_id[:8]}... (showing {len(metrics)})")
        print(tabulate(table_data, headers=headers, tablefmt='simple'))
        
        # Show custom metrics if any
        custom_metrics = [m for m in metrics if m.get('custom')]
        if custom_metrics:
            print(f"\n{len(custom_metrics)} metrics have custom data")
            print("Use --show-custom to display")
    
    def list_projects(self):
        """List all projects"""
        data = self._get('/api/projects')
        projects = data.get('projects', [])
        
        if not projects:
            print("No projects found.")
            return
        
        table_data = [[p['name'], p['run_count']] for p in projects]
        headers = ['Project', 'Runs']
        print(tabulate(table_data, headers=headers, tablefmt='simple'))
    
    def show_stats(self):
        """Show overall statistics"""
        data = self._get('/api/stats')
        
        print("\n" + "="*50)
        print("TrainOps Observatory - Statistics")
        print("="*50)
        print(f"\nTotal Runs:      {data.get('total_runs', 0)}")
        print(f"Running:         {data.get('running_runs', 0)}")
        print(f"Completed:       {data.get('completed_runs', 0)}")
        print(f"Total Metrics:   {data.get('total_metrics', 0):,}")
        print("\n" + "="*50 + "\n")
    
    def delete_run(self, run_id: str, confirm: bool = False):
        """Delete a run"""
        if not confirm:
            response = input(f"Delete run {run_id}? This cannot be undone. [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        try:
            response = requests.delete(f"{self.api_url}/api/runs/{run_id}", timeout=10)
            response.raise_for_status()
            print(f"✓ Run {run_id} deleted successfully")
        except requests.exceptions.RequestException as e:
            print(f"Error deleting run: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='trainops',
        description='TrainOps Observatory CLI - Monitor ML training workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trainops runs list                    # List all runs
  trainops runs list --project mnist    # List runs for project
  trainops runs show <run-id>           # Show run details
  trainops runs metrics <run-id>        # Show metrics
  trainops projects                     # List projects
  trainops stats                        # Show statistics
        """
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:5000',
        help='API URL (default: http://localhost:5000)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # runs command
    runs_parser = subparsers.add_parser('runs', help='Manage training runs')
    runs_subparsers = runs_parser.add_subparsers(dest='subcommand')
    
    # runs list
    list_parser = runs_subparsers.add_parser('list', help='List training runs')
    list_parser.add_argument('--project', help='Filter by project')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--limit', type=int, default=20, help='Max results')
    
    # runs show
    show_parser = runs_subparsers.add_parser('show', help='Show run details')
    show_parser.add_argument('run_id', help='Run ID')
    
    # runs metrics
    metrics_parser = runs_subparsers.add_parser('metrics', help='Show run metrics')
    metrics_parser.add_argument('run_id', help='Run ID')
    metrics_parser.add_argument('--limit', type=int, default=50, help='Max metrics')
    metrics_parser.add_argument('--tail', action='store_true', help='Show last N metrics')
    
    # runs delete
    delete_parser = runs_subparsers.add_parser('delete', help='Delete a run')
    delete_parser.add_argument('run_id', help='Run ID')
    delete_parser.add_argument('--yes', action='store_true', help='Skip confirmation')
    
    # projects command
    subparsers.add_parser('projects', help='List all projects')
    
    # stats command
    subparsers.add_parser('stats', help='Show platform statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    cli = TrainOpsCLI(api_url=args.api_url)
    
    try:
        if args.command == 'runs':
            if args.subcommand == 'list':
                cli.list_runs(project=args.project, status=args.status, limit=args.limit)
            elif args.subcommand == 'show':
                cli.show_run(args.run_id)
            elif args.subcommand == 'metrics':
                cli.show_metrics(args.run_id, limit=args.limit, tail=args.tail)
            elif args.subcommand == 'delete':
                cli.delete_run(args.run_id, confirm=args.yes)
            else:
                runs_parser.print_help()
        
        elif args.command == 'projects':
            cli.list_projects()
        
        elif args.command == 'stats':
            cli.show_stats()
        
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == '__main__':
    main()
