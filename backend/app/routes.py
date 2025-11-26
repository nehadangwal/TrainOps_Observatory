"""
API routes for TrainOps Observatory
"""
from datetime import datetime
from flask import Blueprint, request, jsonify
from sqlalchemy import desc, func
from app.models import db, Run, Metric
from app.analyzer import BottleneckAnalyzer
from app.cost_estimator import CostEstimator

api_bp = Blueprint('api', __name__)


@api_bp.route('/runs', methods=['POST'])
def create_run():
    """Create a new training run"""
    data = request.get_json()
    
    # Validate required fields
    if not data.get('name'):
        return jsonify({'error': 'name is required'}), 400
    
    run = Run(
        name=data['name'],
        project=data.get('project', 'default'),
        instance_type=data.get('instance_type'),
        tags=data.get('tags', {}),
        started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else datetime.utcnow(),
        status='running'
    )
    
    db.session.add(run)
    db.session.commit()
    
    return jsonify({
        'run_id': str(run.id),
        'message': 'Run created successfully'
    }), 201


@api_bp.route('/runs', methods=['GET'])
def list_runs():
    """List all training runs with optional filtering"""
    project = request.args.get('project')
    status = request.args.get('status')
    limit = request.args.get('limit', 50, type=int)
    
    query = Run.query
    
    if project:
        query = query.filter_by(project=project)
    if status:
        query = query.filter_by(status=status)
    
    runs = query.order_by(desc(Run.created_at)).limit(limit).all()
    
    return jsonify({
        'runs': [run.to_dict() for run in runs],
        'total': len(runs)
    })


@api_bp.route('/runs/<run_id>', methods=['GET'])
def get_run(run_id):
    """Get details of a specific run"""
    run = Run.query.get(run_id)
    
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    # Get summary statistics
    metrics_summary = db.session.query(
        func.avg(Metric.gpu_util).label('avg_gpu_util'),
        func.avg(Metric.cpu_util).label('avg_cpu_util'),
        func.avg(Metric.throughput).label('avg_throughput'),
        func.max(Metric.gpu_memory_util).label('max_gpu_memory'),
        func.count(Metric.id).label('metric_count')
    ).filter_by(run_id=run_id).first()
    
    result = run.to_dict()
    result['summary'] = {
        'avg_gpu_util': round(metrics_summary.avg_gpu_util, 2) if metrics_summary.avg_gpu_util else None,
        'avg_cpu_util': round(metrics_summary.avg_cpu_util, 2) if metrics_summary.avg_cpu_util else None,
        'avg_throughput': round(metrics_summary.avg_throughput, 2) if metrics_summary.avg_throughput else None,
        'max_gpu_memory': round(metrics_summary.max_gpu_memory, 2) if metrics_summary.max_gpu_memory else None,
        'metric_count': metrics_summary.metric_count,
    }
    
    return jsonify(result)


@api_bp.route('/runs/<run_id>', methods=['PATCH'])
def update_run(run_id):
    """Update run status or metadata"""
    run = Run.query.get(run_id)
    
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    data = request.get_json()
    
    if 'status' in data:
        run.status = data['status']
    if 'ended_at' in data:
        run.ended_at = datetime.fromisoformat(data['ended_at'])
    if 'total_cost' in data:
        run.total_cost = data['total_cost']
    
    db.session.commit()
    
    return jsonify({
        'message': 'Run updated successfully',
        'run': run.to_dict()
    })


@api_bp.route('/runs/<run_id>', methods=['DELETE'])
def delete_run(run_id):
    """Delete a run and all its metrics"""
    run = Run.query.get(run_id)
    
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    db.session.delete(run)
    db.session.commit()
    
    return jsonify({'message': 'Run deleted successfully'}), 200


@api_bp.route('/runs/<run_id>/metrics', methods=['POST'])
def ingest_metrics(run_id):
    """Ingest batch of metrics for a run"""
    run = Run.query.get(run_id)
    
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    data = request.get_json()
    metrics_data = data.get('metrics', [])
    
    if not metrics_data:
        return jsonify({'error': 'No metrics provided'}), 400
    
    # Batch insert metrics
    metrics = []
    for m in metrics_data:
        metric = Metric(
            run_id=run_id,
            timestamp=datetime.fromisoformat(m['timestamp']),
            elapsed_time=m.get('elapsed_time'),
            cpu_util=m.get('cpu_util'),
            system_memory=m.get('system_memory'),
            io_read_mb=m.get('io_read_mb'),
            io_write_mb=m.get('io_write_mb'),
            gpu_util=m.get('gpu_util'),
            gpu_memory_used=m.get('gpu_memory_used'),
            gpu_memory_total=m.get('gpu_memory_total'),
            gpu_memory_util=m.get('gpu_memory_util'),
            throughput=m.get('throughput'),
            custom=m.get('custom')
        )
        metrics.append(metric)
    
    db.session.bulk_save_objects(metrics)
    db.session.commit()
    
    return jsonify({
        'message': f'{len(metrics)} metrics ingested successfully'
    }), 201


@api_bp.route('/runs/<run_id>/metrics', methods=['GET'])
def get_metrics(run_id):
    """Get metrics for a run"""
    run = Run.query.get(run_id)
    
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    # Pagination
    limit = request.args.get('limit', 1000, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Time range filtering
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    query = Metric.query.filter_by(run_id=run_id)
    
    if start_time:
        query = query.filter(Metric.timestamp >= datetime.fromisoformat(start_time))
    if end_time:
        query = query.filter(Metric.timestamp <= datetime.fromisoformat(end_time))
    
    metrics = query.order_by(Metric.timestamp).offset(offset).limit(limit).all()
    
    return jsonify({
        'metrics': [m.to_dict() for m in metrics],
        'total': len(metrics)
    })


@api_bp.route('/projects', methods=['GET'])
def list_projects():
    """List all unique projects"""
    projects = db.session.query(
        Run.project,
        func.count(Run.id).label('run_count')
    ).group_by(Run.project).all()
    
    return jsonify({
        'projects': [
            {'name': p.project, 'run_count': p.run_count}
            for p in projects
        ]
    })


@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get overall platform statistics"""
    total_runs = db.session.query(func.count(Run.id)).scalar()
    running_runs = db.session.query(func.count(Run.id)).filter_by(status='running').scalar()
    total_metrics = db.session.query(func.count(Metric.id)).scalar()
    
    return jsonify({
        'total_runs': total_runs,
        'running_runs': running_runs,
        'completed_runs': total_runs - running_runs,
        'total_metrics': total_metrics,
    })

    # In routes.py, add the following two functions:

@api_bp.route('/runs/<uuid:run_id>/analysis', methods=['GET'])
def get_run_analysis(run_id):
    """Get the bottleneck analysis report for a run."""
    run = Run.query.get(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    # Analyzer requires metrics data, which is summarized/queried internally
    analyzer = BottleneckAnalyzer(run_id=str(run.id))
    try:
        report = analyzer.generate_analysis_report()
        return jsonify(report)
    except Exception as e:
        # Handle case where run might have no metrics yet
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@api_bp.route('/runs/<uuid:run_id>/cost', methods=['GET'])
def get_run_cost_report(run_id):
    """Get the cost estimation and optimization report for a run."""
    run = Run.query.get(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404
        
    estimator = CostEstimator(run_id=str(run.id))
    try:
        report = estimator.generate_cost_report()
        return jsonify(report)
    except Exception as e:
        # Handle case where run might not be finished or instance type is missing
        return jsonify({'error': f'Cost estimation failed: {str(e)}'}), 500


@api_bp.route('/runs/compare', methods=['GET'])
def compare_runs():
    """Compare two runs side-by-side using summarized metrics."""
    run_ids = request.args.getlist('run_id') # Expects: ?run_id=<id1>&run_id=<id2>
    
    if len(run_ids) != 2:
        return jsonify({'error': 'Must provide exactly two run_id parameters for comparison'}), 400
    
    runs = Run.query.filter(Run.id.in_(run_ids)).all()
    
    if len(runs) != 2:
        return jsonify({'error': 'One or both runs not found'}), 404
        
    # Fetch high-level analysis summary for visualization
    comparison_data = []
    for run in runs:
        analyzer = BottleneckAnalyzer(run_id=str(run.id))
        
        # NOTE: We skip fetching the full cost report for simplicity here, 
        # relying on the pre-computed run.total_cost field.
        
        comparison_data.append({
            'run_id': str(run.id),
            'name': run.name,
            'project': run.project,
            'total_cost': run.total_cost if run.total_cost is not None else 0.0,
            'avg_gpu_util': run.avg_gpu_util if run.avg_gpu_util is not None else 0.0,
            'avg_throughput': run.avg_throughput if run.avg_throughput is not None else 0.0,
            'bottleneck_type': analyzer.get_primary_bottleneck_type(), # Assumes analyzer has this simple method
        })
        
    return jsonify({
        'runs': comparison_data
    })
