# 测试文档

## 测试框架

本项目使用 `pytest` 作为测试框架，包含以下测试类型：

- **单元测试** (`@pytest.mark.unit`): 测试独立的函数和模块
- **集成测试** (`@pytest.mark.integration`): 测试组件之间的交互
- **API测试** (`@pytest.mark.api`): 测试API端点

## 安装测试依赖

```bash
pip install -r requirements.txt
```

## 运行测试

### 运行所有测试

```bash
pytest
```

### 运行特定类型的测试

```bash
# 只运行单元测试
pytest -m unit

# 只运行API测试
pytest -m api

# 只运行集成测试
pytest -m integration
```

### 查看测试覆盖率

```bash
# 终端输出
pytest --cov=. --cov-report=term-missing

# 生成HTML报告
pytest --cov=. --cov-report=html
# 然后打开 htmlcov/index.html
```
