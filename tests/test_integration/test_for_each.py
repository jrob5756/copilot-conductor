"""Integration tests for for-each (dynamic parallel) execution.

Tests cover:
- Template variable injection ({{ <var> }}, {{ _index }}, {{ _key }})
- For-each execution with batching
- Failure modes (fail_fast, continue_on_error, all_or_nothing)
- Output aggregation (list and dict outputs)
- Empty array handling
"""

import pytest

from copilot_conductor.config.schema import (
    AgentDef,
    ContextConfig,
    ForEachDef,
    LimitsConfig,
    OutputField,
    RouteDef,
    RuntimeConfig,
    WorkflowConfig,
    WorkflowDef,
)
from copilot_conductor.engine.workflow import WorkflowEngine
from copilot_conductor.executor.template import TemplateRenderer


class TestLoopVariableTemplateRendering:
    """Tests for template rendering with loop variables.
    
    These tests verify that loop variables ({{ var }}, {{ _index }}, {{ _key }})
    are properly injected into context and accessible in agent templates.
    """

    def test_render_template_with_loop_variable(self):
        """Test that loop variable ({{ <var> }}) is accessible in templates."""
        renderer = TemplateRenderer()
        
        # Context with injected loop variable
        context = {
            "kpi": {"kpi_id": "K1", "name": "Revenue"},
            "_index": 0,
        }
        
        # Template using loop variable
        template = "Analyzing KPI: {{ kpi.kpi_id }} - {{ kpi.name }}"
        result = renderer.render(template, context)
        
        assert result == "Analyzing KPI: K1 - Revenue"

    def test_render_template_with_index_variable(self):
        """Test that {{ _index }} is accessible in templates."""
        renderer = TemplateRenderer()
        
        context = {
            "kpi": {"kpi_id": "K5"},
            "_index": 4,
        }
        
        template = "Processing item #{{ _index + 1 }}: {{ kpi.kpi_id }}"
        result = renderer.render(template, context)
        
        assert result == "Processing item #5: K5"

    def test_render_template_with_key_variable(self):
        """Test that {{ _key }} is accessible when key_by is used."""
        renderer = TemplateRenderer()
        
        context = {
            "kpi": {"kpi_id": "KPI_123", "value": 100},
            "_index": 2,
            "_key": "KPI_123",
        }
        
        template = "Key={{ _key }}, Index={{ _index }}, Value={{ kpi.value }}"
        result = renderer.render(template, context)
        
        assert result == "Key=KPI_123, Index=2, Value=100"

    def test_render_template_with_all_loop_variables(self):
        """Test using all loop variables together in one template."""
        renderer = TemplateRenderer()
        
        context = {
            "item": {"id": "ABC", "status": "active"},
            "_index": 7,
            "_key": "ABC",
        }
        
        template = """
        Item: {{ item.id }}
        Status: {{ item.status }}
        Position: {{ _index }}
        Key: {{ _key }}
        """.strip()
        
        result = renderer.render(template, context)
        
        assert "Item: ABC" in result
        assert "Status: active" in result
        assert "Position: 7" in result
        assert "Key: ABC" in result

    def test_render_template_with_simple_string_item(self):
        """Test loop variable when item is a simple string (not a dict)."""
        renderer = TemplateRenderer()
        
        context = {
            "color": "blue",
            "_index": 1,
        }
        
        template = "Color #{{ _index }}: {{ color }}"
        result = renderer.render(template, context)
        
        assert result == "Color #1: blue"

    def test_render_template_with_number_item(self):
        """Test loop variable when item is a number."""
        renderer = TemplateRenderer()
        
        context = {
            "score": 95,
            "_index": 3,
        }
        
        template = "Score[{{ _index }}] = {{ score }}"
        result = renderer.render(template, context)
        
        assert result == "Score[3] = 95"

    def test_render_template_with_list_item(self):
        """Test loop variable when item is a list."""
        renderer = TemplateRenderer()
        
        context = {
            "batch": ["a", "b", "c"],
            "_index": 0,
        }
        
        template = "Batch {{ _index }}: {{ batch | join(', ') }}"
        result = renderer.render(template, context)
        
        assert result == "Batch 0: a, b, c"

    def test_render_template_with_nested_item_access(self):
        """Test accessing deeply nested fields in loop variable."""
        renderer = TemplateRenderer()
        
        context = {
            "kpi": {
                "id": "revenue",
                "metrics": {
                    "current": 1000,
                    "target": 1500,
                },
                "tags": ["financial", "quarterly"],
            },
            "_index": 0,
        }
        
        template = (
            "KPI {{ kpi.id }}: "
            "{{ kpi.metrics.current }}/{{ kpi.metrics.target }} "
            "({{ kpi.tags[0] }})"
        )
        result = renderer.render(template, context)
        
        assert result == "KPI revenue: 1000/1500 (financial)"

    def test_render_template_with_workflow_and_loop_variables(self):
        """Test that loop variables coexist with workflow context."""
        renderer = TemplateRenderer()
        
        context = {
            "workflow": {"input": {"goal": "analyze all"}},
            "finder": {"output": {"total": 50}},
            "kpi": {"kpi_id": "K1"},
            "_index": 0,
        }
        
        template = (
            "Goal: {{ workflow.input.goal }} | "
            "Total: {{ finder.output.total }} | "
            "Current: {{ kpi.kpi_id }} (#{{ _index }})"
        )
        result = renderer.render(template, context)
        
        assert result == "Goal: analyze all | Total: 50 | Current: K1 (#0)"

    def test_render_template_conditional_with_index(self):
        """Test conditional logic based on _index."""
        renderer = TemplateRenderer()
        
        context = {
            "item": "test",
            "_index": 0,
        }
        
        template = "{% if _index == 0 %}First item{% else %}Item #{{ _index }}{% endif %}"
        result = renderer.render(template, context)
        
        assert result == "First item"
        
        # Test non-zero index
        context["_index"] = 5
        result = renderer.render(template, context)
        assert result == "Item #5"

    def test_render_template_loop_over_item_fields(self):
        """Test using Jinja2 loop over fields in the loop variable."""
        renderer = TemplateRenderer()
        
        context = {
            "kpi": {"id": "K1", "name": "Revenue", "value": 100},
            "_index": 0,
        }
        
        template = "{% for key, val in kpi.items() %}{{ key }}={{ val }} {% endfor %}"
        result = renderer.render(template, context)
        
        # Result should contain all key-value pairs
        assert "id=K1" in result
        assert "name=Revenue" in result
        assert "value=100" in result

    def test_render_template_with_key_without_index(self):
        """Test template that uses _key but not _index."""
        renderer = TemplateRenderer()
        
        context = {
            "kpi": {"status": "active"},
            "_index": 99,
            "_key": "KPI_XYZ",
        }
        
        # Template only uses _key, not _index
        template = "Processing {{ _key }}: {{ kpi.status }}"
        result = renderer.render(template, context)
        
        assert result == "Processing KPI_XYZ: active"

    def test_render_template_missing_loop_variable(self):
        """Test that missing loop variables cause template errors."""
        renderer = TemplateRenderer()
        
        # Context missing the loop variable
        context = {
            "_index": 0,
        }
        
        template = "{{ kpi.kpi_id }}"
        
        # Should raise an error due to undefined variable
        with pytest.raises(Exception):  # TemplateError or UndefinedError
            renderer.render(template, context)

    def test_render_template_with_filters_on_loop_variable(self):
        """Test using Jinja2 filters on loop variables."""
        renderer = TemplateRenderer()
        
        context = {
            "kpi": {"name": "revenue growth"},
            "_index": 0,
        }
        
        template = "KPI: {{ kpi.name | upper }}"
        result = renderer.render(template, context)
        
        assert result == "KPI: REVENUE GROWTH"

    def test_render_template_zero_index(self):
        """Test that _index=0 renders correctly (not treated as falsy)."""
        renderer = TemplateRenderer()
        
        context = {
            "item": "first",
            "_index": 0,
        }
        
        # This should show "Index: 0", not be treated as missing
        template = "Index: {{ _index }}"
        result = renderer.render(template, context)
        
        assert result == "Index: 0"

    def test_render_template_empty_string_key(self):
        """Test that empty string key renders correctly."""
        renderer = TemplateRenderer()
        
        context = {
            "item": "value",
            "_index": 0,
            "_key": "",
        }
        
        template = "Key: '{{ _key }}'"
        result = renderer.render(template, context)
        
        assert result == "Key: ''"
