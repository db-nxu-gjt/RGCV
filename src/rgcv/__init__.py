"""RGCV 复现代码库(Retrieval–Generation–Correction–Verification)。

对应 y.docx(《面向企业级数据库的 Text-to-SQL:结构感知模式检索、DBMS 协同
修复与语义级验证》)第 4 节方法设计与第 5 节实验协议的最小可行复现。

注意:本文档为 registered report 风格研究方案,原实验为设计目标而非已完成的
结论;本代码库在无 TRAE LLM API 通道的约束下,以"规则式生成器 + 受控错误
注入"作为 LLM 代理,复现框架机制与实验协议结构,并在报告中明确标注偏差。
"""

__version__ = "0.1.0"
