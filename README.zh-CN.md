# 沪金波动率预测与期权策略研究

[English README](README.md) · [完整研究报告](docs/research_report_cn.pdf) · [最终策略说明书](docs/strategy_manual_cn.pdf)

这是一个从数据、预测模型到可执行回测的完整量化研究项目：预测沪金未来 5、20、40 个交易日的实现波动率，再把预测与可交易期权 IV 比较，构建波动率风险溢价策略。

## 核心结果

- 主模型：HAR-X + 宏观 + GVZ + SLV IV + US EPU，使用 log-RV MSE 训练。
- 稳健模型：HAR-X + 宏观 + US EPU，使用 QLIKE 训练，完全不使用 IV，用来检查“自证”问题。
- 主模型相对 Persistence 的样本外 R-squared：5/20/40 日分别为 51.0%、53.1%、55.4%。
- 方向准确率：73.9%、76.4%、76.3%。
- 最终策略：10% 单笔风险预算、每 2 个交易日 Delta 对冲、跨式相对价差不超过 25%。
- 2021-07-01 至 2026-07-27 研究回测：净利润 509,760 元，Sharpe 1.085，最大回撤 -0.91%。

需要诚实强调：这是稀疏策略，五年日历里只有 12 笔完整交易，且 2021–2023 没有符合全部信号和真实报价过滤条件的交易。因此回测 Sharpe 的统计不确定性仍然很大，不能当成实盘业绩。

## 回测现实性

- 期权买入按 ask、卖出按 bid；
- 期货对冲买入按 ask1、卖出按 bid1；
- 对冲数量不超过设定的可见深度容量；
- 计入期权/期货手续费、行权费和观测到的买卖价差成本；
- 所有训练样本必须在目标窗口成熟后才能进入训练，避免未来信息泄漏。

## 快速运行

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-lock.txt
    python scripts/33_predict_final_model.py --json
    python scripts/35_verify_final_prediction_model.py
    python scripts/34_serve_final_model_dashboard.py

然后打开 http://127.0.0.1:8765。

原始会员数据、真实密钥、逐笔行情和原始报价均未上传。公开仓库只包含代码、测试、已训练的小型模型、合成样例和衍生结果摘要。详细方法、局限性与完整项目过程请查看页首的研究报告。

