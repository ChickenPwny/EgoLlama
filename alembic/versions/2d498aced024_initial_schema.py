"""Initial schema

Revision ID: 2d498aced024
Revises: 
Create Date: 2025-11-27 00:24:53.191830

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2d498aced024'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - create all tables."""
    # Create llama_models table
    op.create_table(
        'llama_models',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('model_path', sa.String(length=255), nullable=False),
        sa.Column('loaded_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_llama_models_name', 'llama_models', ['name'], unique=False)
    op.create_index('ix_llama_models_is_active', 'llama_models', ['is_active'], unique=False)
    
    # Create llama_inference_requests table
    op.create_table(
        'llama_inference_requests',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('request_time', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('response_time', sa.DateTime(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('response', sa.Text(), nullable=True),
        sa.Column('request_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('response_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('latency', sa.Float(), nullable=True),
        sa.Column('status_code', sa.Integer(), nullable=False, server_default='200'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['llama_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_llama_inference_requests_model_id', 'llama_inference_requests', ['model_id'], unique=False)
    op.create_index('ix_llama_inference_requests_request_time', 'llama_inference_requests', ['request_time'], unique=False)
    op.create_index('llama_inference_model_time_idx', 'llama_inference_requests', ['model_id', 'request_time'], unique=False)
    
    # Create llama_performance_metrics table
    op.create_table(
        'llama_performance_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('gpu_utilization', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('memory_used', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('memory_total', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('power_usage', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('gpu_temperature', sa.Float(), nullable=False, server_default='0.0'),
        sa.ForeignKeyConstraint(['model_id'], ['llama_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_llama_performance_metrics_model_id', 'llama_performance_metrics', ['model_id'], unique=False)
    op.create_index('ix_llama_performance_metrics_timestamp', 'llama_performance_metrics', ['timestamp'], unique=False)
    op.create_index('llama_perf_model_time_idx', 'llama_performance_metrics', ['model_id', 'timestamp'], unique=False)
    
    # Create llama_cache_entries table
    op.create_table(
        'llama_cache_entries',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('response', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_accessed', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('hit_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('extra_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['llama_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_llama_cache_entries_model_id', 'llama_cache_entries', ['model_id'], unique=False)
    op.create_index('ix_llama_cache_entries_prompt', 'llama_cache_entries', ['prompt'], unique=False)
    op.create_index('ix_llama_cache_entries_last_accessed', 'llama_cache_entries', ['last_accessed'], unique=False)
    op.create_index('llama_cache_model_prompt_idx', 'llama_cache_entries', ['model_id', 'prompt'], unique=False)
    
    # Create llama_gateway_health table
    op.create_table(
        'llama_gateway_health',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('gpu_available', sa.Boolean(), nullable=False),
        sa.Column('cache_hit_rate', sa.Float(), nullable=False),
        sa.Column('active_models', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_llama_gateway_health_timestamp', 'llama_gateway_health', ['timestamp'], unique=False)
    op.create_index('ix_llama_gateway_health_status', 'llama_gateway_health', ['status'], unique=False)
    op.create_index('llama_health_status_time_idx', 'llama_gateway_health', ['status', 'timestamp'], unique=False)
    
    # Create llama_prompt_templates table
    op.create_table(
        'llama_prompt_templates',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('template_text', sa.Text(), nullable=False),
        sa.Column('recommended_model_id', sa.Integer(), nullable=True),
        sa.Column('recommended_temperature', sa.Float(), nullable=False, server_default='0.3'),
        sa.Column('recommended_max_tokens', sa.Integer(), nullable=False, server_default='512'),
        sa.Column('config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('use_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['recommended_model_id'], ['llama_models.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_llama_prompt_templates_category', 'llama_prompt_templates', ['category'], unique=False)
    op.create_index('ix_llama_prompt_templates_recommended_model_id', 'llama_prompt_templates', ['recommended_model_id'], unique=False)
    op.create_index('llama_template_use_count_idx', 'llama_prompt_templates', ['use_count'], unique=False)


def downgrade() -> None:
    """Downgrade schema - drop all tables."""
    op.drop_index('llama_template_use_count_idx', table_name='llama_prompt_templates')
    op.drop_index('ix_llama_prompt_templates_recommended_model_id', table_name='llama_prompt_templates')
    op.drop_index('ix_llama_prompt_templates_category', table_name='llama_prompt_templates')
    op.drop_table('llama_prompt_templates')
    
    op.drop_index('llama_health_status_time_idx', table_name='llama_gateway_health')
    op.drop_index('ix_llama_gateway_health_status', table_name='llama_gateway_health')
    op.drop_index('ix_llama_gateway_health_timestamp', table_name='llama_gateway_health')
    op.drop_table('llama_gateway_health')
    
    op.drop_index('llama_cache_model_prompt_idx', table_name='llama_cache_entries')
    op.drop_index('ix_llama_cache_entries_last_accessed', table_name='llama_cache_entries')
    op.drop_index('ix_llama_cache_entries_prompt', table_name='llama_cache_entries')
    op.drop_index('ix_llama_cache_entries_model_id', table_name='llama_cache_entries')
    op.drop_table('llama_cache_entries')
    
    op.drop_index('llama_perf_model_time_idx', table_name='llama_performance_metrics')
    op.drop_index('ix_llama_performance_metrics_timestamp', table_name='llama_performance_metrics')
    op.drop_index('ix_llama_performance_metrics_model_id', table_name='llama_performance_metrics')
    op.drop_table('llama_performance_metrics')
    
    op.drop_index('llama_inference_model_time_idx', table_name='llama_inference_requests')
    op.drop_index('ix_llama_inference_requests_request_time', table_name='llama_inference_requests')
    op.drop_index('ix_llama_inference_requests_model_id', table_name='llama_inference_requests')
    op.drop_table('llama_inference_requests')
    
    op.drop_index('ix_llama_models_is_active', table_name='llama_models')
    op.drop_index('ix_llama_models_name', table_name='llama_models')
    op.drop_table('llama_models')
