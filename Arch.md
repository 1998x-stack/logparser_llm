# LogParser-LLM 代码架构设计

## 一、项目结构

```
logparser_llm/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   └── config.yaml          # 配置文件
├── logparser_llm/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── parser.py        # 主解析引擎
│   │   ├── prefix_tree.py   # 前缀树实现
│   │   ├── llm_extractor.py # LLM模板提取器
│   │   └── merger.py        # 模板合并器
│   ├── models/
│   │   ├── __init__.py
│   │   ├── log_entry.py     # 日志条目数据模型
│   │   ├── template.py      # 模板数据模型
│   │   └── cluster.py       # 聚类数据模型
│   ├── preprocessor/
│   │   ├── __init__.py
│   │   └── cleaner.py       # 日志预处理
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── template_pool.py # 模板池
│   │   └── cache.py         # 缓存管理
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py        # OpenAI客户端
│   │   └── prompts.py       # Prompt模板
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py        # 日志工具
│   │   └── metrics.py       # 评估指标
│   └── api/
│       ├── __init__.py
│       └── rest_api.py      # REST API接口
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_prefix_tree.py
│   └── test_llm_extractor.py
├── examples/
│   ├── basic_usage.py
│   ├── batch_processing.py
│   └── custom_config.py
└── data/
    ├── sample_logs/
    └── benchmarks/
```

## 二、核心设计思路

### 2.1 设计原则

1. **模块化**: 每个组件职责单一,易于测试和维护
2. **可扩展**: 支持不同LLM Provider (OpenAI, Azure, 本地模型)
3. **高性能**: 异步处理,批量操作,智能缓存
4. **容错性**: 异常处理,重试机制,降级策略
5. **可观测**: 详细日志,性能监控,统计指标

### 2.2 数据流

```
原始日志
  ↓
[预处理器] → 清洗、标准化
  ↓
[前缀树] → 语法聚类
  ↓
[缓存检查] → 命中?返回模板 : 继续
  ↓
[LLM提取器] → 调用OpenAI API
  ↓
[模板池] → 存储新模板
  ↓
[合并器] → 自动合并相似模板
  ↓
结构化输出
```

### 2.3 关键算法

#### 前缀树聚类算法
- 基于Token分割
- 公共前缀识别
- 相似度计算

#### LLM提取策略
- 零样本提取
- 少样本ICL
- 自适应批处理

#### 模板合并算法
- 编辑距离计算
- 语义相似度
- 阈值控制

## 三、接口设计

### 3.1 主API接口

```python
class LogParserLLM:
    def __init__(self, config: Config)
    def parse(self, log: str) -> ParsedLog
    def parse_batch(self, logs: List[str]) -> List[ParsedLog]
    def parse_file(self, file_path: str) -> DataFrame
    def get_statistics(self) -> Dict
    def save_templates(self, path: str)
    def load_templates(self, path: str)
```

### 3.2 配置接口

```python
class Config:
    # LLM配置
    llm_provider: str = "openai"
    api_key: str
    model_name: str = "gpt-4"
    temperature: float = 0.0
    
    # 解析配置
    use_cache: bool = True
    batch_size: int = 10
    similarity_threshold: float = 0.85
    
    # 前缀树配置
    max_depth: int = 5
    min_cluster_size: int = 3
```

### 3.3 输出格式

```python
class ParsedLog:
    original: str           # 原始日志
    template: str          # 提取的模板
    template_id: str       # 模板ID
    variables: Dict        # 提取的变量
    timestamp: datetime    # 时间戳
    log_level: str        # 日志级别
    confidence: float     # 置信度
```

## 四、技术选型

### 4.1 核心依赖

```
openai>=1.0.0              # OpenAI官方SDK
pydantic>=2.0.0            # 数据验证
pyyaml>=6.0               # 配置文件
redis>=4.5.0              # 可选缓存
pandas>=2.0.0             # 数据处理
numpy>=1.24.0             # 数值计算
python-dotenv>=1.0.0      # 环境变量
aiohttp>=3.8.0            # 异步HTTP
tqdm>=4.65.0              # 进度条
```

### 4.2 LLM集成方案

**方案1: OpenAI官方SDK** (推荐)
```python
from openai import OpenAI
client = OpenAI(api_key=api_key)
response = client.chat.completions.create(...)
```

**方案2: Azure OpenAI**
```python
from openai import AzureOpenAI
client = AzureOpenAI(...)
```

**方案3: 本地模型** (通过OpenAI兼容接口)
```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)
```

## 五、性能优化策略

### 5.1 缓存策略

```
三级缓存:
1. 内存缓存 (LRU Cache) - 最热数据
2. Redis缓存 (可选) - 持久化
3. 前缀树缓存 - 结构化存储
```

### 5.2 批处理优化

- **智能批量**: 相似日志一起处理
- **并行处理**: 多线程/协程
- **流式处理**: 大文件增量处理

### 5.3 成本控制

- **模型降级**: 简单日志用便宜模型
- **调用限制**: 设置最大调用次数
- **相似度过滤**: 避免重复调用

## 六、错误处理

### 6.1 异常类型

```python
class LogParserException(Exception): pass
class LLMAPIError(LogParserException): pass
class TemplateExtractionError(LogParserException): pass
class InvalidLogFormat(LogParserException): pass
```

### 6.2 重试策略

- **指数退避**: 1s, 2s, 4s, 8s
- **最大重试**: 3次
- **降级处理**: 失败后使用规则方法

## 七、监控指标

### 7.1 性能指标

- 解析吞吐量 (logs/s)
- 平均响应时间
- LLM调用次数
- 缓存命中率

### 7.2 质量指标

- 模板准确率
- 分组准确率
- F1分数
- 覆盖率

### 7.3 成本指标

- API调用成本
- Token使用量
- 内存使用
- 存储占用

## 八、扩展点

### 8.1 自定义Preprocessor

```python
class CustomPreprocessor(BasePreprocessor):
    def process(self, log: str) -> str:
        # 自定义清洗逻辑
        pass
```

### 8.2 自定义LLM Provider

```python
class CustomLLMProvider(BaseLLMProvider):
    def extract_template(self, log: str) -> str:
        # 自定义LLM调用
        pass
```

### 8.3 插件系统

```python
class ParserPlugin:
    def on_before_parse(self, log: str): pass
    def on_after_parse(self, result: ParsedLog): pass
    def on_error(self, error: Exception): pass
```

## 九、部署方案

### 9.1 单机部署

```bash
pip install logparser-llm
logparser-llm serve --config config.yaml
```

### 9.2 容器部署

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "-m", "logparser_llm.api.rest_api"]
```

### 9.3 分布式部署

- **任务队列**: Celery + Redis
- **负载均衡**: Nginx
- **横向扩展**: Kubernetes

## 十、测试策略

### 10.1 单元测试

- 每个组件独立测试
- Mock LLM API调用
- 覆盖率 > 80%

### 10.2 集成测试

- 端到端流程测试
- 使用真实日志数据
- 性能基准测试

### 10.3 压力测试

- 大规模日志处理
- 并发请求测试
- 内存泄漏检测

## 十一、最佳实践

### 11.1 配置管理

```yaml
# config.yaml
llm:
  provider: openai
  model: gpt-4-turbo-preview
  api_key: ${OPENAI_API_KEY}  # 从环境变量读取
  
parsing:
  cache_enabled: true
  batch_size: 20
  similarity_threshold: 0.85
  
performance:
  max_workers: 4
  timeout: 30
  max_retries: 3
```

### 11.2 日志记录

```python
import logging

logger = logging.getLogger("logparser_llm")
logger.info("Parsed %d logs, LLM calls: %d", total, llm_calls)
```

### 11.3 指标上报

```python
from prometheus_client import Counter, Histogram

parse_counter = Counter("logs_parsed_total")
parse_duration = Histogram("parse_duration_seconds")
```

## 十二、下一步开发计划

### Phase 1: 核心功能 (Week 1-2)
- ✅ 基础架构搭建
- ✅ 前缀树实现
- ✅ OpenAI集成
- ✅ 基本解析功能

### Phase 2: 优化增强 (Week 3-4)
- 缓存系统
- 批处理优化
- 模板合并
- 错误处理

### Phase 3: 工程化 (Week 5-6)
- REST API
- 监控指标
- 文档完善
- 性能测试

### Phase 4: 高级特性 (Week 7-8)
- 多Provider支持
- 插件系统
- 可视化界面
- 分布式部署

---

这个架构设计确保了:
- ✅ 高性能和可扩展性
- ✅ 易于维护和测试
- ✅ 生产就绪
- ✅ 符合工业标准