"""
RAG系统使用示例
演示如何处理海量日志的摄入和检索
"""

import asyncio
from datetime import datetime, timedelta
from backend.app.rag import (
    HybridIngestionManager,
    BatchIngestionProcessor,
    StreamingIngestionProcessor,
    TelemetryLog,
    LogPriority,
    HybridRetriever,
    ContextualRetriever,
    RetrievalQuery,
    IngestionConfig,
    RetrievalConfig
)
import asyncpg
from boto3 import client as boto_client


async def setup_database():
    """设置数据库连接池"""
    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        database="traffic_sentinel",
        user="postgres",
        password="your_password",
        min_size=10,
        max_size=20
    )
    return pool


async def setup_bedrock():
    """设置Bedrock客户端"""
    return boto_client('bedrock-runtime', region_name='us-east-1')


async def example_ingestion():
    """示例：数据摄入"""
    print("=" * 60)
    print("示例：海量日志摄入（延迟优化）")
    print("=" * 60)
    
    # 1. 初始化配置
    config = IngestionConfig(
        batch_size=1000,
        batch_timeout=5.0,
        max_concurrent_batches=10
    )
    
    # 2. 设置数据库和Bedrock
    db_pool = await setup_database()
    bedrock_client = await setup_bedrock()
    
    # 3. 创建处理器
    batch_processor = BatchIngestionProcessor(
        batch_size=config.batch_size,
        batch_timeout=config.batch_timeout,
        max_concurrent_batches=config.max_concurrent_batches,
        db_pool=db_pool,
        bedrock_client=bedrock_client
    )
    
    stream_processor = StreamingIngestionProcessor(
        db_pool=db_pool,
        bedrock_client=bedrock_client
    )
    
    ingestion_manager = HybridIngestionManager(
        batch_processor=batch_processor,
        stream_processor=stream_processor
    )
    
    # 4. 启动处理器
    await ingestion_manager.start()
    
    # 5. 模拟海量日志摄入
    print("\n开始摄入日志...")
    start_time = datetime.now()
    
    # 模拟10000条日志
    for i in range(10000):
        # 根据日志特征设置优先级
        if i % 100 == 0:
            # 每100条有一条关键日志（高延迟或错误）
            log = TelemetryLog(
                timestamp=datetime.now(),
                source_ip=f"192.168.1.{i % 255}",
                dest_ip="10.0.0.1",
                protocol="HTTP",
                status_code=500 if i % 200 == 0 else 200,
                bytes_sent=1024,
                bytes_received=2048,
                latency_ms=1500.0,  # 高延迟
                path="/api/endpoint",
                priority=LogPriority.CRITICAL
            )
        else:
            # 普通日志
            log = TelemetryLog(
                timestamp=datetime.now(),
                source_ip=f"192.168.1.{i % 255}",
                dest_ip="10.0.0.1",
                protocol="HTTP",
                status_code=200,
                bytes_sent=512,
                bytes_received=1024,
                latency_ms=50.0,
                path="/api/endpoint",
                priority=LogPriority.NORMAL
            )
        
        await ingestion_manager.ingest(log)
        
        if (i + 1) % 1000 == 0:
            print(f"已摄入 {i + 1} 条日志...")
    
    # 6. 等待所有批次处理完成
    await asyncio.sleep(10)  # 等待批处理完成
    
    # 7. 停止处理器
    await ingestion_manager.stop()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n摄入完成！")
    print(f"总耗时: {elapsed:.2f}秒")
    print(f"吞吐量: {10000 / elapsed:.2f} 条/秒")
    
    await db_pool.close()


async def example_retrieval():
    """示例：检索（相关性优化）"""
    print("\n" + "=" * 60)
    print("示例：日志检索（相关性优化）")
    print("=" * 60)
    
    # 1. 初始化配置
    config = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        default_top_k=10,
        enable_reranking=True
    )
    
    # 2. 设置数据库和Bedrock
    db_pool = await setup_database()
    bedrock_client = await setup_bedrock()
    
    # 3. 创建检索器
    base_retriever = HybridRetriever(
        db_pool=db_pool,
        bedrock_client=bedrock_client,
        vector_weight=config.vector_weight,
        keyword_weight=config.keyword_weight
    )
    
    contextual_retriever = ContextualRetriever(
        base_retriever=base_retriever,
        time_decay_factor=config.time_decay_factor
    )
    
    # 4. 执行检索查询
    print("\n执行检索查询...")
    
    # 查询1：查找高延迟请求
    query1 = RetrievalQuery(
        query_text="高延迟的HTTP请求，响应时间超过1秒",
        time_range=(
            datetime.now() - timedelta(hours=24),
            datetime.now()
        ),
        min_similarity=0.7,
        top_k=10
    )
    
    results1 = await contextual_retriever.retrieve_with_context(query1)
    print(f"\n查询1结果: 找到 {len(results1)} 条相关日志")
    for i, result in enumerate(results1[:5], 1):
        print(f"  {i}. 延迟: {result.latency_ms}ms, "
              f"相似度: {result.hybrid_score:.3f}, "
              f"时间: {result.timestamp}")
    
    # 查询2：查找特定IP的错误请求
    query2 = RetrievalQuery(
        query_text="来自192.168.1.100的错误请求",
        source_ip="192.168.1.100",
        status_code=500,
        time_range=(
            datetime.now() - timedelta(hours=1),
            datetime.now()
        ),
        min_similarity=0.6,
        top_k=5
    )
    
    results2 = await contextual_retriever.retrieve_with_context(query2)
    print(f"\n查询2结果: 找到 {len(results2)} 条相关日志")
    for i, result in enumerate(results2[:5], 1):
        print(f"  {i}. 状态码: {result.status_code}, "
              f"源IP: {result.source_ip}, "
              f"相似度: {result.hybrid_score:.3f}")
    
    # 查询3：带上下文的连续查询
    print("\n执行带上下文的连续查询...")
    conversation_history = [
        "查找高延迟请求",
        "这些请求来自哪些IP？"
    ]
    
    query3 = RetrievalQuery(
        query_text="这些高延迟请求的来源IP分布",
        time_range=(
            datetime.now() - timedelta(hours=24),
            datetime.now()
        ),
        min_similarity=0.7,
        top_k=10
    )
    
    results3 = await contextual_retriever.retrieve_with_context(
        query3,
        conversation_history=conversation_history
    )
    print(f"查询3结果: 找到 {len(results3)} 条相关日志")
    
    # 统计IP分布
    ip_counts = {}
    for result in results3:
        ip_counts[result.source_ip] = ip_counts.get(result.source_ip, 0) + 1
    
    print("\n来源IP分布:")
    for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {ip}: {count} 条")
    
    await db_pool.close()


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RAG系统示例：海量网络日志处理")
    print("=" * 60)
    
    # 注意：实际运行时需要先设置数据库和Bedrock
    # 这里仅展示使用方式
    
    # await example_ingestion()
    # await example_retrieval()
    
    print("\n提示：取消注释上面的函数调用来运行示例")


if __name__ == "__main__":
    asyncio.run(main())



