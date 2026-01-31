# Dynamic Parallel Agents Implementation Plan

## 1. Problem Statement

Copilot Conductor currently supports static parallel execution where agent groups are predefined in the workflow YAML at design time. However, there are scenarios where the number and nature of parallel tasks are only known at runtime—when one agent produces a list that should spawn N parallel instances of another agent.

**Current limitation**: Processing items sequentially in loops (e.g., 50 KPIs taking 50 sequential iterations).

**Desired capability**: Dynamic parallel execution where runtime data (e.g., a list of KPIs returned by an agent) spawns N parallel instances of a template agent, similar to functional programming's `map` operation or Kubernetes Jobs' parallelism model.

This solution builds upon the existing static parallel infrastructure (introduced in `parallel-agent-execution.plan.md`) to add runtime-determined parallelism via a `for_each` agent type.

## 2. Goals and Non-Goals

### Goals

1. **Dynamic parallelism**: Enable agents to spawn parallel instances based on runtime array data from previous agents
2. **Template instantiation**: Support inline agent definitions as templates that get instantiated per array item
3. **Concurrency control**: Provide `max_concurrent` limits to prevent runaway parallelism
4. **Failure modes**: Reuse proven failure handling from static parallel groups (fail_fast, continue_on_error, all_or_nothing)
5. **Array output access**: Support both index-based (`outputs[0]`) and optional key-based (`outputs["KPI123"]`) result access
6. **Code reuse**: Leverage 80%+ of static parallel infrastructure (context snapshots, asyncio.gather, error aggregation)
7. **Backward compatibility**: Zero breaking changes to existing workflows
8. **Empty array handling**: Gracefully handle empty source arrays without errors

### Non-Goals

1. **Nested for-each loops**: Explicitly forbidden (same as nested static parallel groups)
2. **Streaming results**: Results only accessible after all instances complete (future enhancement)
3. **Partial result access**: Cannot access incomplete results mid-execution
4. **Agent template references**: Only inline agent definitions supported in MVP (referencing existing agents deferred)
5. **Retry mechanisms**: Re-running failed items from errors array (future enhancement)
6. **Dynamic concurrency adjustment**: max_concurrent is fixed at workflow definition time
7. **Cross-instance communication**: Parallel instances are fully isolated

## 3. Requirements

### Functional Requirements

**FR-1: ForEach Agent Type**
- New `type: for_each` agent definition in YAML schema
- Required fields: `source` (array reference), `as` (loop variable name), `agent` (template definition)
- Optional fields: `max_concurrent` (default: 10), `failure_mode` (default: fail_fast), `key_by` (for keyed output access)

**FR-2: Array Resolution**
- Resolve `source` references like `finder.output.kpis` to runtime array values
- Support references to workflow inputs, agent outputs, and parallel group outputs
- Validate source resolves to array type at runtime
- Handle empty arrays gracefully (skip execution, return `{outputs: [], errors: [], count: 0}`)

**FR-3: Template Instantiation**
- For each array item, instantiate agent template with loop variable injection
- Loop variables available: `{{ <loop_var> }}` (item), `{{ <loop_var>_index }}` (0-based index)
- Deep copy agent template for each instance to prevent mutation
- Render prompts with loop variable context before execution

**FR-4: Concurrency Control**
- Implement batching when array length exceeds `max_concurrent`
- Execute up to `max_concurrent` instances simultaneously using asyncio.Semaphore
- Sequential batches until all items processed
- Default max_concurrent: 10 (prevents unbounded parallelism)

**FR-5: Output Structure**
- Store results as: `{outputs: [...], errors: [...], count: N}`
- `outputs`: Array of successful agent outputs (preserves order)
- `errors`: Array of error objects `{index: N, message: str, exception_type: str}`
- `count`: Total number of items processed
- When `key_by` specified: `outputs` becomes dict keyed by extracted field

**FR-6: Template Access Patterns**
- Index-based: `{{ for_each_group.outputs[0].field }}`
- Length checks: `{{ for_each_group.outputs | length }}`
- Iteration: `{% for result in for_each_group.outputs %}`
- Key-based (when key_by set): `{{ for_each_group.outputs["KPI123"].field }}`
- Error access: `{{ for_each_group.errors }}`

**FR-7: Validation**
- Validate `source` reference syntax at workflow load time
- Validate `agent` template is valid AgentDef
- Validate `for_each` agents cannot be nested
- Validate `for_each` agents cannot contain routes (routes defined at for_each level)
- Validate `max_concurrent` is positive integer

### Non-Functional Requirements

**NFR-1: Performance**
- Batching prevents memory exhaustion on large arrays (10k+ items)
- Context snapshot overhead linear with number of concurrent instances (O(max_concurrent))
- No significant performance difference vs static parallel for equivalent agent counts

**NFR-2: Memory Management**
- Maximum concurrent context snapshots limited by `max_concurrent`
- Results array grows linearly with input array size (acceptable for MVP)
- Deep copy overhead acceptable for small-to-medium contexts (<100KB)

**NFR-3: Error Messages**
- Failed instances clearly identify index/key and error details
- Suggest checking source array and template configuration
- Verbose mode shows per-instance execution timing

**NFR-4: Backward Compatibility**
- No changes to existing agent types or parallel groups
- Existing workflows execute identically
- No performance regression for non-for_each workflows

## 4. Solution Architecture

### 4.1 Overview

The `for_each` agent type extends the workflow execution engine to support dynamic parallelism. When execution reaches a for-each agent:

1. **Array Resolution**: Resolve `source` reference to runtime array value from context
2. **Validation**: Verify source is array type and not empty (or handle gracefully)
3. **Template Preparation**: Deep copy agent template for instantiation
4. **Batched Execution**: Process array in batches of `max_concurrent` items:
   - For each item: Create context snapshot + inject loop variables
   - Execute batch using `asyncio.gather()` (reusing parallel infrastructure)
   - Collect outputs and errors
5. **Output Aggregation**: Structure results as array or dict (if key_by specified)
6. **Failure Handling**: Apply failure_mode policy (fail_fast, continue_on_error, all_or_nothing)
7. **Storage**: Store aggregated output in WorkflowContext under for_each agent name

### 4.2 Key Components

#### 4.2.1 ForEachDef (New - config/schema.py)

```python
class ForEachDef(BaseModel):
    """Definition for a dynamic parallel for-each agent group."""
    
    name: str
    """Unique identifier for this for-each group."""
    
    type: Literal["for_each"]
    """Agent type identifier."""
    
    description: str | None = None
    """Human-readable description."""
    
    source: str
    """Reference to array in context (e.g., 'finder.output.kpis', 'workflow.input.items')."""
    
    as_: str = Field(alias="as")
    """Variable name for loop items (e.g., 'kpi', 'item')."""
    
    agent: AgentDef
    """Template agent definition to instantiate for each item."""
    
    max_concurrent: int = Field(default=10, ge=1, le=100)
    """Maximum number of concurrent agent instances."""
    
    failure_mode: Literal["fail_fast", "continue_on_error", "all_or_nothing"] = "fail_fast"
    """
    Failure handling mode:
    - fail_fast: Stop on first failure
    - continue_on_error: Collect errors, continue if at least one succeeds
    - all_or_nothing: All must succeed or entire group fails
    """
    
    key_by: str | None = None
    """Optional field path to extract from items as output dict key (e.g., 'kpi.id')."""
    
    routes: list[RouteDef] = Field(default_factory=list)
    """Routing rules evaluated after for-each completion."""
    
    @field_validator("agent")
    @classmethod
    def validate_agent_no_routes(cls, v: AgentDef) -> AgentDef:
        """Ensure template agent doesn't define routes."""
        if v.routes:
            raise ValueError("Template agent in for_each cannot define routes")
        return v
    
    @model_validator(mode="after")
    def validate_source_format(self) -> ForEachDef:
        """Validate source reference format."""
        if not self.source or "." not in self.source:
            raise ValueError(
                f"source must be a context reference (e.g., 'agent.output.field')"
            )
        return self
```

#### 4.2.2 Array Resolution (Enhanced - engine/context.py)

```python
# Add to WorkflowContext class

def resolve_array_reference(self, reference: str) -> list[Any]:
    """Resolve a context reference to an array value.
    
    Args:
        reference: Dot-notation reference (e.g., 'finder.output.kpis')
        
    Returns:
        The resolved array value.
        
    Raises:
        ValueError: If reference doesn't resolve to an array.
    """
    # Parse reference into parts
    parts = reference.split(".")
    
    # Start with workflow inputs or agent outputs
    if parts[0] == "workflow":
        # workflow.input.items
        if len(parts) < 3 or parts[1] != "input":
            raise ValueError(f"Invalid workflow reference: {reference}")
        value = self.workflow_inputs.get(parts[2])
    else:
        # agent_name.output.field or parallel_group.outputs
        agent_name = parts[0]
        if agent_name not in self.agent_outputs:
            raise ValueError(f"Agent output not found: {agent_name}")
        
        value = self.agent_outputs[agent_name]
        
        # Navigate nested path
        for part in parts[1:]:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                raise ValueError(f"Cannot access {part} on non-dict value")
    
    # Validate it's an array
    if not isinstance(value, list):
        raise ValueError(
            f"Source '{reference}' resolved to {type(value).__name__}, expected array"
        )
    
    return value
```

#### 4.2.3 For-Each Execution (New - engine/workflow.py)

```python
async def _execute_for_each_group(self, for_each: ForEachDef) -> ForEachGroupOutput:
    """Execute template agent for each item in source array.
    
    This method:
    1. Resolves source reference to array
    2. Creates batches based on max_concurrent
    3. For each batch: Instantiates template + injects loop vars + executes in parallel
    4. Aggregates outputs and errors
    5. Applies failure mode policy
    
    Args:
        for_each: The for-each group definition.
        
    Returns:
        ForEachGroupOutput with outputs array/dict and errors.
        
    Raises:
        ExecutionError: Based on failure_mode and execution results.
    """
    # Resolve source array
    try:
        items = self.context.resolve_array_reference(for_each.source)
    except ValueError as e:
        raise ExecutionError(
            f"Failed to resolve for_each source '{for_each.source}': {e}",
            suggestion="Ensure source references a valid array in context"
        )
    
    # Handle empty array
    if not items:
        _verbose_log(f"For-each '{for_each.name}': source array is empty, skipping")
        return ForEachGroupOutput(outputs=[], errors=[], count=0)
    
    _verbose_log_for_each_start(for_each.name, len(items), for_each.max_concurrent)
    _group_start = _time.time()
    
    # Prepare output structure
    output_list: list[dict[str, Any]] = []
    output_dict: dict[str, dict[str, Any]] = {}
    error_list: list[dict[str, Any]] = []
    
    # Create batches
    batches = [
        items[i:i + for_each.max_concurrent]
        for i in range(0, len(items), for_each.max_concurrent)
    ]
    
    # Process each batch
    for batch_idx, batch in enumerate(batches):
        batch_start_index = batch_idx * for_each.max_concurrent
        
        async def execute_instance(item: Any, index: int) -> tuple[int, Any, str | None]:
            """Execute template agent for one item.
            
            Returns:
                Tuple of (index, output, key_value)
            """
            _inst_start = _time.time()
            try:
                # Create context snapshot
                instance_context = copy.deepcopy(self.context)
                
                # Inject loop variables into context
                loop_vars = {
                    for_each.as_: item,
                    f"{for_each.as_}_index": index,
                }
                
                # Build agent context with loop vars injected
                agent_context = instance_context.build_for_agent(
                    f"{for_each.name}[{index}]",
                    for_each.agent.input,
                    mode=self.config.workflow.context.mode,
                )
                agent_context.update(loop_vars)
                
                # Render prompt with loop vars
                rendered_agent = self._render_agent_template(for_each.agent, agent_context)
                
                # Execute agent
                output = await self.executor.execute(rendered_agent, agent_context)
                _inst_elapsed = _time.time() - _inst_start
                
                _verbose_log_for_each_instance_complete(
                    for_each.name, index, _inst_elapsed, output.model, output.tokens_used
                )
                
                # Extract key if key_by specified
                key_value = None
                if for_each.key_by:
                    key_value = self._extract_key_value(item, for_each.key_by)
                
                return (index, output.content, key_value)
                
            except Exception as e:
                _inst_elapsed = _time.time() - _inst_start
                _verbose_log_for_each_instance_failed(
                    for_each.name, index, _inst_elapsed, type(e).__name__, str(e)
                )
                
                # Attach metadata for error handling
                e._for_each_index = index  # type: ignore
                raise
        
        # Execute batch based on failure mode
        if for_each.failure_mode == "fail_fast":
            try:
                batch_results = await asyncio.gather(
                    *[
                        execute_instance(item, batch_start_index + i)
                        for i, item in enumerate(batch)
                    ],
                    return_exceptions=False,
                )
                # Store successful results
                for index, content, key_value in batch_results:
                    if for_each.key_by and key_value:
                        output_dict[key_value] = content
                    else:
                        output_list.append(content)
                        
            except Exception as e:
                index = getattr(e, '_for_each_index', 'unknown')
                raise ExecutionError(
                    f"For-each '{for_each.name}' failed at index {index} (fail_fast mode): {type(e).__name__}: {e}",
                    suggestion="Check template agent configuration and source array items"
                ) from e
        
        elif for_each.failure_mode in ("continue_on_error", "all_or_nothing"):
            batch_results = await asyncio.gather(
                *[
                    execute_instance(item, batch_start_index + i)
                    for i, item in enumerate(batch)
                ],
                return_exceptions=True,
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    index = getattr(result, '_for_each_index', -1)
                    error_list.append({
                        "index": index,
                        "exception_type": type(result).__name__,
                        "message": str(result),
                    })
                else:
                    index, content, key_value = result
                    if for_each.key_by and key_value:
                        output_dict[key_value] = content
                    else:
                        output_list.append(content)
    
    # Apply failure mode policy
    _group_elapsed = _time.time() - _group_start
    
    if for_each.failure_mode == "all_or_nothing" and error_list:
        _verbose_log_for_each_summary(
            for_each.name, len(output_list or output_dict), len(error_list), _group_elapsed
        )
        raise ExecutionError(
            f"For-each '{for_each.name}' failed: {len(error_list)} of {len(items)} instances failed (all_or_nothing mode)",
            suggestion=f"Check errors: {error_list[:3]}"
        )
    
    if for_each.failure_mode == "continue_on_error" and not (output_list or output_dict):
        _verbose_log_for_each_summary(
            for_each.name, 0, len(error_list), _group_elapsed
        )
        raise ExecutionError(
            f"For-each '{for_each.name}' failed: all {len(items)} instances failed",
            suggestion=f"Check errors: {error_list[:3]}"
        )
    
    _verbose_log_for_each_summary(
        for_each.name,
        len(output_list or output_dict),
        len(error_list),
        _group_elapsed
    )
    
    # Return structured output
    outputs = output_dict if for_each.key_by else output_list
    return ForEachGroupOutput(outputs=outputs, errors=error_list, count=len(items))
```

#### 4.2.4 Output Structure (New - engine/workflow.py)

```python
@dataclass
class ForEachGroupOutput:
    """Output structure for for-each group execution."""
    
    outputs: list[dict[str, Any]] | dict[str, dict[str, Any]]
    """Array of outputs (list) or keyed outputs (dict if key_by specified)."""
    
    errors: list[dict[str, Any]]
    """Array of error objects with index, exception_type, message."""
    
    count: int
    """Total number of items processed."""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for context storage."""
        return {
            "outputs": self.outputs,
            "errors": self.errors,
            "count": self.count,
        }
```

### 4.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Array Resolution                                                 │
│    for_each.source: "finder.output.kpis"                           │
│    → context.resolve_array_reference()                             │
│    → [kpi1, kpi2, ..., kpiN]                                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Batching                                                         │
│    items: [kpi1...kpi50]                                           │
│    max_concurrent: 5                                               │
│    → batches: [[kpi1-kpi5], [kpi6-kpi10], ..., [kpi46-kpi50]]     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Per-Batch Parallel Execution (asyncio.gather)                   │
│    Batch 1: kpi1-kpi5                                              │
│    ┌───────────────────────────────────────────────────────┐      │
│    │ For each item:                                        │      │
│    │  - deepcopy(context)                                  │      │
│    │  - inject loop vars: {kpi: kpi1, kpi_index: 0}        │      │
│    │  - render template agent prompt                       │      │
│    │  - execute_agent()                                    │      │
│    │  - return (index, output, key)                        │      │
│    └───────────────────────────────────────────────────────┘      │
│    → [result1, result2, ..., result5]                              │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Output Aggregation                                               │
│    Successful: outputs.append(result.content)                      │
│    Failed: errors.append({index, exception_type, message})         │
│    If key_by: outputs = {key: content, ...}                        │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Failure Mode Application                                        │
│    fail_fast: Raise on first error (batch aborts)                  │
│    continue_on_error: Raise only if all failed                     │
│    all_or_nothing: Raise if any failed                             │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Context Storage                                                  │
│    context.store(for_each.name, {                                  │
│      outputs: [...] or {...},                                      │
│      errors: [...],                                                │
│      count: 50                                                     │
│    })                                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 API Contracts

#### YAML Syntax

```yaml
agents:
  - name: analyzers
    type: for_each
    description: Analyze each KPI in parallel
    source: finder.output.kpis
    as: kpi
    max_concurrent: 5
    failure_mode: continue_on_error
    key_by: kpi.kpi_id  # Optional
    agent:
      model: opus-4.5
      prompt: |
        Analyze KPI: {{ kpi.kpi_id }}
        Title: {{ kpi.title }}
        Index: {{ kpi_index }}
      output:
        success:
          type: boolean
        summary:
          type: string
    routes:
      - to: summarizer
```

#### Template Access Patterns

```yaml
# Index-based access
- name: reporter
  prompt: |
    Results: {{ analyzers.outputs | length }} / {{ analyzers.count }}
    {% for result in analyzers.outputs %}
    - {{ result.summary }}
    {% endfor %}
    
    Errors: {{ analyzers.errors | length }}
    {% if analyzers.errors %}
    Failed indices: {{ analyzers.errors | map(attribute='index') | list }}
    {% endif %}

# Key-based access (when key_by specified)
- name: reporter
  prompt: |
    KPI123 result: {{ analyzers.outputs["KPI123"].summary }}
```

## 5. Dependencies

### External Dependencies
- **Python 3.12+**: For type hints and modern async features
- **Pydantic v2**: Schema validation (already used)
- **Jinja2**: Template rendering with loop variable injection (already used)
- **asyncio**: Parallel execution and semaphore-based batching (already used)

### Internal Dependencies
- **Static parallel infrastructure**: Reuses 80% of code from `parallel-agent-execution.plan.md`
  - Context snapshot mechanism (`copy.deepcopy`)
  - `asyncio.gather()` execution pattern
  - Failure mode handling logic
  - Error aggregation patterns
  - Verbose logging utilities
- **WorkflowContext**: Array resolution and loop variable injection
- **TemplateRenderer**: Prompt rendering with loop variables
- **AgentExecutor**: Template agent execution
- **Schema validation**: ForEachDef integration into WorkflowConfig

### Prerequisite Work
- **REQUIRED**: Static parallel groups implementation (Epics 1-10 from `parallel-agent-execution.plan.md`)
- Verification that static parallel groups are stable and tested

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Memory exhaustion on large arrays (10k+ items)** | Medium | High | Implement batching with `max_concurrent` limit; warn if array > 1000 items |
| **Deep copy overhead on large contexts** | Medium | Medium | Document context size best practices; consider shallow copy for immutable data |
| **Unbounded parallelism misconfiguration** | Low | High | Enforce max_concurrent upper limit (100); clear documentation on defaults |
| **Template variable name collisions** | Low | Medium | Validate loop var names don't conflict with reserved names (workflow, context) |
| **Circular reference in source** | Low | Low | Runtime validation of source reference before execution |
| **Empty array edge cases** | Medium | Low | Explicit handling: skip execution, return empty results, log clearly |
| **Key extraction failures (key_by)** | Medium | Medium | Try/except around key extraction; fall back to index-based on error with warning |
| **Complexity for users** | Medium | Medium | Comprehensive examples; migration guide from sequential loops |
| **Debugging parallel failures** | Medium | Medium | Verbose logging per instance; clear error messages with indices |

## 7. Implementation Phases

### Phase 1: Schema & Validation (Foundation)
**Goal**: Define ForEachDef schema and integrate validation into workflow loading.

**Exit Criteria**:
- ForEachDef validated on load
- Validation errors for nested for_each, invalid source refs, template routes
- Unit tests pass for all validation rules

### Phase 2: Array Resolution (Runtime Foundation)
**Goal**: Implement context reference resolution for arrays.

**Exit Criteria**:
- resolve_array_reference() handles all reference types
- Proper error messages for non-array, missing refs
- Unit tests cover workflow inputs, agent outputs, parallel group outputs
- Empty array handling tested

### Phase 3: Template Instantiation & Loop Variables
**Goal**: Support loop variable injection and template rendering.

**Exit Criteria**:
- Loop vars accessible in templates
- Prompt rendering works with injected variables
- Unit tests for variable injection and template rendering

### Phase 4: For-Each Execution Engine (Core)
**Goal**: Implement batched parallel execution with failure modes.

**Exit Criteria**:
- Batching logic works correctly
- All failure modes function as specified
- Output aggregation (array and dict) works
- Integration tests pass for basic for_each workflows

### Phase 5: Key-Based Output Access
**Goal**: Support optional key_by for dict-based output access.

**Exit Criteria**:
- Key extraction from items works
- Dict output accessible in templates
- Unit tests for key_by scenarios
- Graceful fallback on extraction errors

### Phase 6: Error Handling & Logging
**Goal**: Comprehensive error messages and verbose logging.

**Exit Criteria**:
- Per-instance verbose logging
- Summary logging with counts and timing
- Clear error messages with indices
- User-facing documentation on debugging

### Phase 7: Documentation & Examples
**Goal**: User-facing docs and migration guides.

**Exit Criteria**:
- YAML syntax reference updated
- Examples added (KPI analysis, multi-source research)
- Migration guide from sequential loops
- API reference for template access patterns

## 8. Files Affected

### New Files

| File Path | Purpose |
|-----------|---------|
| `src/copilot_conductor/engine/for_each.py` | ForEachGroupOutput dataclass and helper functions |
| `tests/test_engine/test_for_each.py` | Unit tests for for-each execution engine |
| `tests/test_config/test_for_each_validation.py` | Unit tests for ForEachDef validation |
| `tests/test_integration/test_for_each_workflows.py` | Integration tests for complete for-each workflows |
| `examples/kpi-analysis-parallel.yaml` | Example: Convert sequential KPI analysis to parallel |
| `examples/multi-source-research.yaml` | Example: Research workflow with for-each over sources |
| `docs/features/for-each-agents.md` | User documentation for for-each feature |
| `docs/migration/sequential-to-parallel.md` | Migration guide from loops to for-each |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `src/copilot_conductor/config/schema.py` | Add `ForEachDef` class; update `WorkflowConfig` to support for_each in agents list |
| `src/copilot_conductor/engine/context.py` | Add `resolve_array_reference()` method; enhance loop variable injection |
| `src/copilot_conductor/engine/workflow.py` | Add `_execute_for_each_group()` method; update routing to handle for_each agents |
| `src/copilot_conductor/executor/template.py` | Add `render_with_loop_vars()` helper for loop variable injection |
| `src/copilot_conductor/cli/run.py` | Add verbose logging functions: `verbose_log_for_each_start`, `verbose_log_for_each_instance_complete`, `verbose_log_for_each_instance_failed`, `verbose_log_for_each_summary` |
| `src/copilot_conductor/config/validator.py` | Add validation for for_each agents (no nesting, valid source refs, no template routes) |
| `docs/yaml-reference.md` | Add for_each agent type documentation |
| `README.md` | Add dynamic parallelism to features list |
| `pyproject.toml` | Bump version to indicate new feature |

### Deleted Files

| File Path | Reason |
|-----------|--------|
| None | No files deleted (additive feature) |

## 9. Implementation Plan

### Epic 1: Schema Definition and Basic Validation

**Goal**: Define the ForEachDef schema and integrate it into the workflow configuration system with basic validation rules.

**Prerequisites**: None (builds on existing schema infrastructure)

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E1-T1 | IMPL | Add ForEachDef class to schema.py with all fields | `src/copilot_conductor/config/schema.py` | TO DO |
| E1-T2 | IMPL | Update WorkflowConfig to accept ForEachDef in agents Union type | `src/copilot_conductor/config/schema.py` | TO DO |
| E1-T3 | IMPL | Add field validator for source reference format | `src/copilot_conductor/config/schema.py` | TO DO |
| E1-T4 | IMPL | Add validator to ensure template agent has no routes | `src/copilot_conductor/config/schema.py` | TO DO |
| E1-T5 | TEST | Unit tests for ForEachDef validation (valid/invalid configs) | `tests/test_config/test_for_each_validation.py` | TO DO |
| E1-T6 | TEST | Unit tests for error messages on validation failures | `tests/test_config/test_for_each_validation.py` | TO DO |

**Acceptance Criteria**:
- [ ] ForEachDef schema validates all required and optional fields
- [ ] Invalid source references rejected with clear error messages
- [ ] Template agents with routes rejected at load time
- [ ] All unit tests pass with 100% coverage of validation logic

---

### Epic 2: Array Resolution in Context

**Goal**: Implement runtime array resolution from context references to support dynamic for_each sources.

**Prerequisites**: Epic 1

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E2-T1 | IMPL | Add resolve_array_reference() method to WorkflowContext | `src/copilot_conductor/engine/context.py` | TO DO |
| E2-T2 | IMPL | Support workflow.input.* array references | `src/copilot_conductor/engine/context.py` | TO DO |
| E2-T3 | IMPL | Support agent.output.* array references | `src/copilot_conductor/engine/context.py` | TO DO |
| E2-T4 | IMPL | Support parallel_group.outputs array references | `src/copilot_conductor/engine/context.py` | TO DO |
| E2-T5 | IMPL | Add validation that resolved value is array type | `src/copilot_conductor/engine/context.py` | TO DO |
| E2-T6 | IMPL | Handle empty array gracefully (no error) | `src/copilot_conductor/engine/context.py` | TO DO |
| E2-T7 | TEST | Unit tests for all reference types | `tests/test_engine/test_for_each.py` | TO DO |
| E2-T8 | TEST | Unit tests for error cases (missing ref, non-array, invalid path) | `tests/test_engine/test_for_each.py` | TO DO |
| E2-T9 | TEST | Unit test for empty array handling | `tests/test_engine/test_for_each.py` | TO DO |

**Acceptance Criteria**:
- [ ] resolve_array_reference() handles all source reference patterns
- [ ] Clear error messages for non-array types and missing references
- [ ] Empty arrays return gracefully without exceptions
- [ ] All unit tests pass with edge case coverage

---

### Epic 3: Template Rendering with Loop Variables

**Goal**: Support loop variable injection into agent templates for rendering prompts with item-specific data.

**Prerequisites**: Epic 2

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E3-T1 | IMPL | Add loop variable injection to context snapshots | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E3-T2 | IMPL | Support both item variable and index variable (`{{ kpi }}`, `{{ kpi_index }}`) | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E3-T3 | IMPL | Add _render_agent_template() helper for prompt rendering | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E3-T4 | IMPL | Validate loop variable names don't conflict with reserved names | `src/copilot_conductor/config/validator.py` | TO DO |
| E3-T5 | TEST | Unit tests for loop variable injection | `tests/test_engine/test_for_each.py` | TO DO |
| E3-T6 | TEST | Unit tests for template rendering with loop vars | `tests/test_engine/test_for_each.py` | TO DO |
| E3-T7 | TEST | Unit test for variable name conflict detection | `tests/test_config/test_for_each_validation.py` | TO DO |

**Acceptance Criteria**:
- [ ] Loop variables accessible in templates (both item and index)
- [ ] Prompts render correctly with injected variables
- [ ] Reserved name conflicts detected and rejected
- [ ] All unit tests pass with template rendering scenarios

---

### Epic 4: Batched Parallel Execution Engine

**Goal**: Implement the core for-each execution engine with batching and parallel execution using asyncio.

**Prerequisites**: Epic 3

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E4-T1 | IMPL | Add _execute_for_each_group() method to WorkflowEngine | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E4-T2 | IMPL | Implement batching logic based on max_concurrent | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E4-T3 | IMPL | Implement execute_instance() helper for single item execution | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E4-T4 | IMPL | Use asyncio.gather() for batch parallel execution | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E4-T5 | IMPL | Create ForEachGroupOutput dataclass | `src/copilot_conductor/engine/for_each.py` | TO DO |
| E4-T6 | IMPL | Implement output aggregation (array-based) | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E4-T7 | IMPL | Add for_each routing support in main execution loop | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E4-T8 | TEST | Unit tests for batching logic (various array sizes vs max_concurrent) | `tests/test_engine/test_for_each.py` | TO DO |
| E4-T9 | TEST | Integration test: Simple for_each workflow (2-3 items) | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E4-T10 | TEST | Integration test: Large array (50+ items) with batching | `tests/test_integration/test_for_each_workflows.py` | TO DO |

**Acceptance Criteria**:
- [ ] Batching correctly limits concurrent executions to max_concurrent
- [ ] All items in array processed across batches
- [ ] Outputs aggregated in correct order
- [ ] Integration tests pass for small and large arrays

---

### Epic 5: Failure Mode Implementation

**Goal**: Implement all three failure modes (fail_fast, continue_on_error, all_or_nothing) with proper error aggregation.

**Prerequisites**: Epic 4

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E5-T1 | IMPL | Implement fail_fast mode (raise on first exception) | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E5-T2 | IMPL | Implement continue_on_error mode (collect errors, continue) | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E5-T3 | IMPL | Implement all_or_nothing mode (collect all, fail if any error) | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E5-T4 | IMPL | Add error object structure (index, exception_type, message) | `src/copilot_conductor/engine/for_each.py` | TO DO |
| E5-T5 | IMPL | Store errors array in ForEachGroupOutput | `src/copilot_conductor/engine/for_each.py` | TO DO |
| E5-T6 | TEST | Integration test: fail_fast stops on first error | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E5-T7 | TEST | Integration test: continue_on_error collects partial results | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E5-T8 | TEST | Integration test: all_or_nothing fails if any error | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E5-T9 | TEST | Unit test: Error object structure validation | `tests/test_engine/test_for_each.py` | TO DO |

**Acceptance Criteria**:
- [ ] fail_fast raises immediately on first agent failure
- [ ] continue_on_error collects errors and succeeds if at least one instance succeeds
- [ ] all_or_nothing runs all instances but fails if any error
- [ ] Error array contains correct metadata (index, type, message)
- [ ] All integration tests pass for each failure mode

---

### Epic 6: Key-Based Output Access

**Goal**: Support optional key_by parameter for dict-based output access instead of array-based.

**Prerequisites**: Epic 5

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E6-T1 | IMPL | Add _extract_key_value() helper to extract key from item | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E6-T2 | IMPL | Support nested field paths in key_by (e.g., 'kpi.id') | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E6-T3 | IMPL | Store outputs as dict when key_by specified | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E6-T4 | IMPL | Handle key extraction failures gracefully (warn, fallback to index) | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E6-T5 | IMPL | Update ForEachGroupOutput to support dict outputs type | `src/copilot_conductor/engine/for_each.py` | TO DO |
| E6-T6 | TEST | Unit tests for key extraction from various item structures | `tests/test_engine/test_for_each.py` | TO DO |
| E6-T7 | TEST | Integration test: Key-based access in templates | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E6-T8 | TEST | Unit test: Key extraction failure handling | `tests/test_engine/test_for_each.py` | TO DO |

**Acceptance Criteria**:
- [ ] key_by successfully extracts keys from items
- [ ] Outputs stored as dict with extracted keys
- [ ] Templates can access outputs via key (e.g., `outputs["KPI123"]`)
- [ ] Graceful fallback on extraction errors with warnings
- [ ] All tests pass for key-based access scenarios

---

### Epic 7: Verbose Logging and Error Messages

**Goal**: Add comprehensive verbose logging for for-each execution and clear error messages for debugging.

**Prerequisites**: Epic 5

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E7-T1 | IMPL | Add verbose_log_for_each_start() function | `src/copilot_conductor/cli/run.py` | TO DO |
| E7-T2 | IMPL | Add verbose_log_for_each_instance_complete() function | `src/copilot_conductor/cli/run.py` | TO DO |
| E7-T3 | IMPL | Add verbose_log_for_each_instance_failed() function | `src/copilot_conductor/cli/run.py` | TO DO |
| E7-T4 | IMPL | Add verbose_log_for_each_summary() function (counts, timing) | `src/copilot_conductor/cli/run.py` | TO DO |
| E7-T5 | IMPL | Add warnings for large arrays (>1000 items) | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E7-T6 | IMPL | Enhance error messages with indices and suggestions | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E7-T7 | TEST | Manual test: Run with --verbose and verify logging output | Manual test | TO DO |

**Acceptance Criteria**:
- [ ] Verbose mode shows start of for-each with item count and max_concurrent
- [ ] Per-instance completion/failure logged with timing
- [ ] Summary shows success/failure counts and total time
- [ ] Large array warnings displayed appropriately
- [ ] Error messages include helpful debugging information

---

### Epic 8: Context Integration and Template Access

**Goal**: Ensure for-each outputs are accessible in downstream agent templates with all access patterns.

**Prerequisites**: Epic 6

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E8-T1 | IMPL | Update context.py to handle for-each outputs in build_for_agent() | `src/copilot_conductor/engine/context.py` | TO DO |
| E8-T2 | IMPL | Support index-based access pattern in templates | `src/copilot_conductor/engine/context.py` | TO DO |
| E8-T3 | IMPL | Support key-based access pattern in templates (when key_by used) | `src/copilot_conductor/engine/context.py` | TO DO |
| E8-T4 | IMPL | Support iteration over outputs array in templates | Already supported by Jinja2 | TO DO |
| E8-T5 | TEST | Integration test: Access outputs[0] in downstream agent | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E8-T6 | TEST | Integration test: Iterate over outputs in template | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E8-T7 | TEST | Integration test: Access outputs["key"] with key_by | `tests/test_integration/test_for_each_workflows.py` | TO DO |
| E8-T8 | TEST | Integration test: Access errors array in template | `tests/test_integration/test_for_each_workflows.py` | TO DO |

**Acceptance Criteria**:
- [ ] Downstream agents can access for_each outputs via all patterns
- [ ] Index-based, key-based, and iteration patterns work correctly
- [ ] Errors array accessible for error handling in templates
- [ ] All integration tests pass for template access patterns

---

### Epic 9: Advanced Validation

**Goal**: Implement comprehensive validation rules for for_each agents (no nesting, valid references, etc.).

**Prerequisites**: Epic 1

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E9-T1 | IMPL | Add validation to prevent nested for_each agents | `src/copilot_conductor/config/validator.py` | TO DO |
| E9-T2 | IMPL | Validate source reference exists in workflow | `src/copilot_conductor/config/validator.py` | TO DO |
| E9-T3 | IMPL | Validate loop variable names (no conflicts with 'workflow', 'context') | `src/copilot_conductor/config/validator.py` | TO DO |
| E9-T4 | IMPL | Update cycle detection to handle for_each agents | `src/copilot_conductor/config/validator.py` | TO DO |
| E9-T5 | TEST | Unit test: Nested for_each rejected | `tests/test_config/test_for_each_validation.py` | TO DO |
| E9-T6 | TEST | Unit test: Invalid source reference rejected | `tests/test_config/test_for_each_validation.py` | TO DO |
| E9-T7 | TEST | Unit test: Reserved loop variable names rejected | `tests/test_config/test_for_each_validation.py` | TO DO |
| E9-T8 | TEST | Unit test: Cycle detection with for_each agents | `tests/test_config/test_for_each_validation.py` | TO DO |

**Acceptance Criteria**:
- [ ] Nested for_each agents rejected at load time
- [ ] Invalid source references detected and reported
- [ ] Reserved loop variable names blocked
- [ ] Cycle detection works correctly with for_each in graph
- [ ] All validation tests pass with clear error messages

---

### Epic 10: Documentation and Examples

**Goal**: Create comprehensive user documentation, examples, and migration guides.

**Prerequisites**: Epics 1-8 (all implementation complete)

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E10-T1 | IMPL | Create for-each feature documentation | `docs/features/for-each-agents.md` | TO DO |
| E10-T2 | IMPL | Update YAML reference with for_each syntax | `docs/yaml-reference.md` | TO DO |
| E10-T3 | IMPL | Create KPI analysis parallel example | `examples/kpi-analysis-parallel.yaml` | TO DO |
| E10-T4 | IMPL | Create multi-source research example | `examples/multi-source-research.yaml` | TO DO |
| E10-T5 | IMPL | Create migration guide from sequential loops to for_each | `docs/migration/sequential-to-parallel.md` | TO DO |
| E10-T6 | IMPL | Update README.md features list | `README.md` | TO DO |
| E10-T7 | IMPL | Add troubleshooting section for common for_each issues | `docs/features/for-each-agents.md` | TO DO |
| E10-T8 | TEST | Manual test: Run all example workflows and verify output | Manual test | TO DO |

**Acceptance Criteria**:
- [ ] Feature documentation covers all YAML options and access patterns
- [ ] YAML reference includes complete for_each syntax specification
- [ ] At least 2 realistic example workflows provided
- [ ] Migration guide helps users convert sequential loops to for_each
- [ ] README.md accurately reflects new capabilities
- [ ] All examples run successfully and produce expected output

---

### Epic 11: Performance Testing and Optimization

**Goal**: Validate performance characteristics and optimize batching/memory usage if needed.

**Prerequisites**: Epic 10 (all features complete)

**Tasks**:

| Task ID | Type | Description | Files | Status |
|---------|------|-------------|-------|--------|
| E11-T1 | TEST | Performance test: 100-item array with max_concurrent=10 | `tests/test_performance/test_for_each_performance.py` | TO DO |
| E11-T2 | TEST | Performance test: 1000-item array with max_concurrent=50 | `tests/test_performance/test_for_each_performance.py` | TO DO |
| E11-T3 | TEST | Memory test: Monitor memory usage during large for_each | `tests/test_performance/test_for_each_performance.py` | TO DO |
| E11-T4 | IMPL | Optimize context deepcopy if performance issues identified | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E11-T5 | IMPL | Add memory usage warnings for large arrays | `src/copilot_conductor/engine/workflow.py` | TO DO |
| E11-T6 | TEST | Benchmark: Compare for_each vs sequential loop performance | `tests/test_performance/test_for_each_performance.py` | TO DO |

**Acceptance Criteria**:
- [ ] 100-item for_each completes within 2x time of 100 max_concurrent
- [ ] 1000-item for_each completes without memory errors
- [ ] Memory usage scales linearly with max_concurrent, not array size
- [ ] Performance comparable to static parallel for equivalent agent counts
- [ ] Warnings displayed for scenarios that may cause issues

---

## Notes

### Code Reuse from Static Parallel

Approximately 80% of the for-each implementation reuses infrastructure from static parallel groups:

- **Context snapshots**: Same `copy.deepcopy()` mechanism
- **Parallel execution**: Same `asyncio.gather()` pattern
- **Failure modes**: Same logic for fail_fast, continue_on_error, all_or_nothing
- **Error aggregation**: Same error object structure and handling
- **Verbose logging**: Similar logging patterns (adapted for instances vs agents)
- **Output storage**: Same context.store() mechanism with structured outputs

### Migration from Sequential Loops

The for_each feature is designed as a direct replacement for sequential loop patterns:

**Before (sequential)**:
```yaml
agents:
  - name: finder
    output:
      next_item: { type: object }
      all_complete: { type: boolean }
    routes:
      - to: $end
        when: "{{ output.all_complete }}"
      - to: processor

  - name: processor
    input: [finder.output]
    routes:
      - to: finder  # Loop back
```

**After (parallel)**:
```yaml
agents:
  - name: finder
    output:
      items: { type: array }  # Return ALL items
    routes:
      - to: processors

  - name: processors
    type: for_each
    source: finder.output.items
    as: item
    max_concurrent: 10
    agent:
      prompt: "Process {{ item }}"
```

### Future Enhancements

The following features are intentionally deferred to post-MVP based on user feedback:

1. **Template references**: `template: existing_agent_name` to reuse agent definitions
2. **Template overrides**: `template: base, overrides: {model: opus-4.5}` for customization
3. **Streaming results**: Callbacks/webhooks as each instance completes
4. **Retry failed items**: Mechanism to re-run only items in errors array
5. **Dynamic max_concurrent**: Adjust concurrency based on runtime conditions
6. **Resource pooling**: Share rate limits across instances
7. **Progress tracking**: Real-time progress reporting for long-running for_each

These can be added incrementally based on proven use cases without breaking backward compatibility.

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-31  
**Status**: Ready for Implementation
