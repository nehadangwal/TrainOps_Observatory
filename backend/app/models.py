"""
Database models for TrainOps Observatory
"""
from datetime import datetime
from uuid import uuid4
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID, JSONB

db = SQLAlchemy()


class Run(db.Model):
    """Training run metadata"""
    __tablename__ = 'runs'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = db.Column(db.String(255), nullable=False)
    project = db.Column(db.String(255), nullable=False, default='default')
    instance_type = db.Column(db.String(50))
    status = db.Column(db.String(20), default='running')  # running, completed, failed
    tags = db.Column(JSONB, default={})
    
    started_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    
    # Metrics
    metrics = db.relationship('Metric', backref='run', lazy='dynamic', cascade='all, delete-orphan')
    
    # Computed fields (updated periodically)
    total_cost = db.Column(db.Numeric(10, 2))
    avg_gpu_util = db.Column(db.Float)
    avg_throughput = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Run {self.name} ({self.id})>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'name': self.name,
            'project': self.project,
            'instance_type': self.instance_type,
            'status': self.status,
            'tags': self.tags,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'total_cost': float(self.total_cost) if self.total_cost else None,
            'avg_gpu_util': self.avg_gpu_util,
            'avg_throughput': self.avg_throughput,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
    
    @property
    def duration_seconds(self):
        """Calculate run duration in seconds"""
        if not self.started_at:
            return 0
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
    
    @property
    def duration_hours(self):
        """Calculate run duration in hours"""
        return self.duration_seconds / 3600


class Metric(db.Model):
    """Time-series metrics data"""
    __tablename__ = 'metrics'
    
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    run_id = db.Column(UUID(as_uuid=True), db.ForeignKey('runs.id'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # System metrics
    elapsed_time = db.Column(db.Float)
    cpu_util = db.Column(db.Float)
    system_memory = db.Column(db.Float)
    io_read_mb = db.Column(db.Float)
    io_write_mb = db.Column(db.Float)
    
    # GPU metrics
    gpu_util = db.Column(db.Float)
    gpu_memory_used = db.Column(db.Float)
    gpu_memory_total = db.Column(db.Float)
    gpu_memory_util = db.Column(db.Float)
    
    # Training metrics
    throughput = db.Column(db.Float)
    
    # Custom metrics (flexible JSON storage)
    custom = db.Column(JSONB)
    
    def __repr__(self):
        return f'<Metric {self.run_id} @ {self.timestamp}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'run_id': str(self.run_id),
            'timestamp': self.timestamp.isoformat(),
            'elapsed_time': self.elapsed_time,
            'cpu_util': self.cpu_util,
            'system_memory': self.system_memory,
            'io_read_mb': self.io_read_mb,
            'io_write_mb': self.io_write_mb,
            'gpu_util': self.gpu_util,
            'gpu_memory_used': self.gpu_memory_used,
            'gpu_memory_total': self.gpu_memory_total,
            'gpu_memory_util': self.gpu_memory_util,
            'throughput': self.throughput,
            'custom': self.custom,
        }


def init_db(app):
    """Initialize database with TimescaleDB extensions"""
    db.init_app(app)
    
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Create TimescaleDB hypertable for metrics
        try:
            db.session.execute(db.text("""
                SELECT create_hypertable('metrics', 'timestamp', 
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            """))
            db.session.commit()
            print("✓ Created TimescaleDB hypertable for metrics")
        except Exception as e:
            print(f"Note: TimescaleDB hypertable creation: {e}")
            db.session.rollback()
        
        # Create indexes for common queries
        try:
            db.session.execute(db.text("""
                CREATE INDEX IF NOT EXISTS idx_metrics_run_timestamp 
                ON metrics (run_id, timestamp DESC);
            """))
            db.session.execute(db.text("""
                CREATE INDEX IF NOT EXISTS idx_runs_project_created 
                ON runs (project, created_at DESC);
            """))
            db.session.commit()
            print("✓ Created indexes")
        except Exception as e:
            print(f"Note: Index creation: {e}")
            db.session.rollback()
