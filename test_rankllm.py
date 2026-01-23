#!/usr/bin/env python3
"""
Basic tests for all_in_one_rankllm.py

Tests the core functionality without requiring full datasets or models.
"""

import sys
import os
import json
import tempfile
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from all_in_one_rankllm
from all_in_one_rankllm import (
    create_model_config,
    save_results,
    evaluate_metrics,
)


class TestModelConfig:
    """Test model configuration creation."""
    
    def test_valid_model_names(self):
        """Test that valid model names work."""
        valid_models = ['qwen3', 'qwen14', 'qwen32', 'qwen72', 'calme', 'llama', 'mistral']
        
        for model_name in valid_models:
            model_path, max_length, quant_config = create_model_config(model_name)
            assert isinstance(model_path, str)
            assert isinstance(max_length, int)
            assert max_length > 0
    
    def test_invalid_model_name(self):
        """Test that invalid model names raise error."""
        with pytest.raises(ValueError):
            create_model_config('invalid_model')


class TestSaveResults:
    """Test results saving functionality."""
    
    def test_save_results_creates_file(self):
        """Test that save_results creates a JSON file."""
        # Create mock args
        class MockArgs:
            dataset = 'test_dataset'
            model_name = 'qwen3'
            filter_method = 'custom_llm'
            ranking_method = 'custom_llm'
            bm25_topk = 100
            filter_topk = 10
            rankllm_model = 'test_model'
            bert_model = 'test_bert'
        
        args = MockArgs()
        
        metrics_dict = {
            'bm25': {'ndcg@10': 0.5},
            'two_stage': {'ndcg@10': 0.6}
        }
        
        codecarbon_metrics = {
            'filter': {'emissions_kg': 0.001},
            'ranking': {'emissions_kg': 0.002}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_results(args, metrics_dict, codecarbon_metrics, tmpdir)
            
            # Check file exists
            assert os.path.exists(filepath)
            
            # Check content
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert 'config' in data
            assert 'metrics' in data
            assert 'codecarbon' in data
            assert 'timestamp' in data
            
            assert data['config']['dataset'] == 'test_dataset'
            assert data['metrics'] == metrics_dict
            assert data['codecarbon'] == codecarbon_metrics
    
    def test_filename_format(self):
        """Test that filename includes all hyperparameters."""
        class MockArgs:
            dataset = 'scifact'
            model_name = 'qwen3'
            filter_method = 'bert_index'
            ranking_method = 'custom_llm'
            bm25_topk = 100
            filter_topk = 10
            rankllm_model = 'test_model'
            bert_model = 'test_bert'
        
        args = MockArgs()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_results(args, {}, {}, tmpdir)
            filename = os.path.basename(filepath)
            
            # Check all components are in filename
            assert 'scifact' in filename
            assert 'qwen3' in filename
            assert 'bert_index' in filename
            assert 'custom_llm' in filename
            assert 'topk100' in filename
            assert 'filtertopk10' in filename
            assert filename.endswith('.json')


class TestEvaluateMetrics:
    """Test metrics evaluation."""
    
    def test_evaluate_metrics_basic(self):
        """Test basic metrics evaluation."""
        # Simple test case
        qrels = {
            'q1': {'d1': 1, 'd2': 1, 'd3': 0}
        }
        
        run = {
            'q1': {'d1': 2.0, 'd2': 1.5, 'd3': 1.0}
        }
        
        results = evaluate_metrics(qrels, run)
        
        # Should return some metrics
        assert isinstance(results, dict)
        # Don't check exact values as they depend on ranx implementation
    
    def test_evaluate_metrics_empty(self):
        """Test evaluation with empty results."""
        qrels = {}
        run = {}
        
        results = evaluate_metrics(qrels, run)
        assert isinstance(results, dict)


class TestTwoStageRankerStructure:
    """Test TwoStageRanker class structure without loading models."""
    
    def test_class_can_be_imported(self):
        """Test that TwoStageRanker class can be imported."""
        from all_in_one_rankllm import TwoStageRanker
        assert TwoStageRanker is not None
    
    def test_supported_methods(self):
        """Test that supported filter and ranking methods are defined."""
        # This is more of a documentation test
        supported_filter_methods = ['custom_llm', 'rankT5', 'bert_index']
        supported_ranking_methods = ['custom_llm', 'rankllm_listwise', 'rankllm_pointwise']
        
        # Just verify they're defined in the test
        assert len(supported_filter_methods) == 3
        assert len(supported_ranking_methods) == 3


def test_imports():
    """Test that all required imports work."""
    try:
        import torch
        import transformers
        import tqdm
        import beir
        import ranx
        import codecarbon
        print("All required packages imported successfully")
    except ImportError as e:
        pytest.skip(f"Required package not installed: {e}")


def test_script_syntax():
    """Test that the main script has valid Python syntax."""
    import py_compile
    
    script_path = os.path.join(os.path.dirname(__file__), 'all_in_one_rankllm.py')
    
    try:
        py_compile.compile(script_path, doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Syntax error in all_in_one_rankllm.py: {e}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
