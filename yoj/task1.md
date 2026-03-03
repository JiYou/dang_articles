# Task 1: ETF轮动策略优化与组合

## 背景

我们已完成两个ETF轮动策略的回测（前复权数据）：

| 策略 | 年化 | 回撤 | Sharpe | Calmar |
|------|------|------|--------|--------|
| 中信证券+纳指ETF轮动 (3200/2900) | 32.12% | 24.01% | — | 1.34 |
| 全ETF MACD轮动 M(6,17,5) 370只ETF | 27.06% | 7.97% | 2.16 | 3.39 |

**数据说明**：
- 所有ETF数据使用**前复权**（qfq），存储在 `market_data_qfq/` 目录
- 原始不复权数据在 `market_data/`，已弃用（会产生幽灵收益）
- 前复权数据通过腾讯财经API下载，共1442只ETF
- CSV列顺序：`date,open,high,low,close,volume`（volume单位是手，1手=100股）
- 已通过流动性过滤（最近60日平均日成交额 >= 1亿元），370只通过
- ETF名称来源：`etf_list_cache.json`（格式：`{"code":"159998","name":"计算机ETF"}`）

**策略代码说明**：
- 所有策略代码位于 `/ceph/dang_articles/yoj/stock_data_bfq/` 目录
- 均为 C++17，使用多线程参数优化
- `all_etf_rotation.cpp`（950行）：全ETF MACD轮动，参数网格约48600个组合
- `etf_rotation.cpp`（463行）：中信证券+纳指ETF轮动
- `cyclical_etf_rotation_wf.cpp`（985行）：14只周期ETF的walk-forward验证（参考模板）

现在要做三件事：过滤非权益类ETF、Walk-forward验证、两策略组合。

---

## 任务一：过滤非权益类ETF

**目标**：从370只通过流动性过滤的ETF中，去掉债券/货币/商品类ETF，只保留权益类ETF。

**当前ETF分类统计**（基于 `etf_list_cache.json` 中1442只ETF的名称分析）：

| 类别 | 数量 | 说明 |
|------|------|------|
| 债券类 | 52 | 名称含：债、利率、信用 |
| 货币类 | 19 | 名称含：货币ETF（注意"现金流ETF"是权益类！） |
| 商品类 | ~10 | 黄金ETF（不含"黄金股"）、豆粕ETF、能源化工ETF |
| 权益类 | ~1361 | 其余全部 |

**排除关键词（精确规则）**：

```
排除条件（满足任意一条即排除）：
1. 名称包含 "货币"
2. 名称包含 "债"（覆盖：国债、企债、信用债、可转债、政金债、科创债、地债）
3. 名称包含 "利率"
4. 名称包含 "豆粕"
5. 名称包含 "黄金" 且 不包含 "股"
   → 排除：黄金ETF博时(159937)、黄金ETF(518880)等（商品类）
   → 保留：黄金股ETF(159562)、黄金股票ETF(159321)等（权益类）
6. 名称包含 "能源化工"
   → 排除：能源化工ETF建信(159981)（跟踪商品期货指数）
   → 注意："能源ETF"(159930)是权益类，不排除

不排除的易混淆名称：
- "现金流ETF" / "自由现金流ETF" → 权益类（跟踪自由现金流指数，选股因子ETF）
- "有色ETF" / "有色金属ETF" → 权益类（跟踪有色金属行业股票）
- "能源ETF" → 权益类（跟踪能源行业股票）
- "增强ETF" → 权益类（指数增强基金）
- "基建ETF" → 权益类（跟踪基建行业股票）
```

**实现位置**：在 `all_etf_rotation.cpp` 的流动性过滤代码（line 665-680）之后，新增名称过滤逻辑。需要使用已加载的ETF名称（`etf.name` 字段）。

**注意**：C++ 中处理中文字符串需要用 `string::find()` 查找 UTF-8 子串，不能用单字符比较。例如：
```cpp
bool is_non_equity = false;
if (name.find("货币") != string::npos) is_non_equity = true;
if (name.find("债") != string::npos) is_non_equity = true;
if (name.find("利率") != string::npos) is_non_equity = true;
if (name.find("豆粕") != string::npos) is_non_equity = true;
if (name.find("能源化工") != string::npos) is_non_equity = true;
if (name.find("黄金") != string::npos && name.find("股") == string::npos) is_non_equity = true;
```

**预期产出**：
- 更新后的 `all_etf_rotation.cpp`，包含非权益类过滤逻辑
- 终端输出：被过滤掉的ETF列表（带原因）+ 过滤后ETF数量
- 重跑策略后的结果对比表（370只 vs 过滤后）

---

## 任务二：Walk-forward验证（MACD轮动策略）

**目标**：对过滤后的权益类ETF池做滚动窗口验证，确认 M(6,17,5) 最优参数不是过拟合。

**参考模板**：`cyclical_etf_rotation_wf.cpp` 的核心结构：

1. **滚动窗口**：2年训练 + 1年测试，每次前移1年
   - 例：训练[2016-2018), 测试[2018-2019) → 训练[2017-2019), 测试[2019-2020) → ...
2. **训练期**：在窗口内跑完整参数网格（~48600组合），按 combined_score 排序取最优
3. **测试期**：用训练期最优参数在测试期外推，记录年化/回撤/Sharpe/Calmar
4. **MACD warmup**：`build_etfs_for_window()` 加载从最早数据到窗口结束的所有数据做MACD计算，但 `run_backtest_windowed()` 只在窗口内交易（避免前视偏差）
5. **鲁棒性检查**：同时测试训练期 Top-5 参数在测试期的中位数表现
6. **固定参数逐年测试**：用全样本最优参数逐年跑，观察年度稳定性

**关键实现差异**（vs 参考模板）：
- 参考模板用14只固定ETF + `market_data/`（不复权），新代码需用 `market_data_qfq/`（前复权）+ 动态发现ETF
- 参考模板ETF池固定，新代码的ETF池在每个窗口可能不同（因为流动性过滤基于最近60天数据）
- 新代码需要包含任务一的非权益类过滤逻辑
- `run_backtest_windowed()` 接受 `global_start_idx` 和 `global_end_idx` 参数控制交易窗口范围

**性能考量**：
- 参数网格 ~48600 × 370只ETF × 多个窗口，计算量巨大
- 可考虑：缩减网格（先用粗网格定位区域，再细搜）或减少 buy_signal/sell_signal 的候选值
- 参考模板在14只ETF上跑一个窗口的网格已用多线程加速

**预期产出**：
- `all_etf_rotation_wf.cpp` 代码（在 `stock_data_bfq/` 目录下）
- 输出：各窗口训练最优参数 + 测试表现 + Top-5 中位数
- 汇总统计：测试年化中位数/均值、正收益窗口比例、最大测试回撤
- 与全样本结果（27.06%年化, 7.97%回撤）的对比

---

## 任务三：两策略组合

**目标**：将中信+纳指宏观择时策略与MACD轮动行业选择策略组合成一个完整策略。

**组合逻辑**：

```
每个交易日，用昨日上证指数收盘价判断：

1. 上证 < 2900（低位抄底区）：
   - 清空MACD轮动持仓
   - 全仓买入中信证券(600030)
   
2. 上证 > 3200（高位避险区）：
   - 清空中信证券和MACD轮动持仓
   - 全仓买入纳指ETF(513100)

3. 上证在 2900~3200（正常区间）：
   - 如果从区间1或2切换过来，先清空旧持仓
   - 执行MACD轮动策略，在权益类ETF池中按月线MACD信号轮动
   
切换时的持仓处理：
- 从区间1→3 或 3→1：中信卖出 → 等待MACD信号 / MACD持仓卖出 → 买入中信
- 从区间2→3 或 3→2：纳指卖出 → 等待MACD信号 / MACD持仓卖出 → 买入纳指
- 从区间1→2 或 2→1：直接切换（中信↔纳指）
```

**需要额外加载的数据**：
- **上证指数**：`market_data/sh000001.csv`（不复权，指数不需要复权）
  - 格式：`date,open,high,low,close,volume`
  - 用 `load_market_csv()` 加载（与 `etf_rotation.cpp` 中相同）
- **中信证券**：`stock_data_bfq/600030.csv`（不复权股票数据）
  - 格式：`date,股票代码,open,close,high,low,volume,...`
  - 注意列顺序不同！close在open后面（第4列），不是第5列
  - 用 `load_stock_csv()` 加载

**实现复杂度**：
- 需要同时维护三种状态：HOLD_CITIC / HOLD_NASDAQ / HOLD_MACD_ROTATION
- MACD轮动状态下需要维护多仓位（`vector<Position>`）
- 状态切换时需要在当日开盘价卖出旧持仓、买入新持仓

**预期产出**：
- `combined_strategy.cpp` 组合策略代码
- 回测结果：年化、回撤、Sharpe、Calmar、交易次数
- 三策略对比表：组合 vs 中信+纳指单策略 vs MACD轮动单策略
- 交易明细（含状态切换记录）

---

## 依赖关系

```
任务一（过滤非权益类ETF）
    ↓
任务二（Walk-forward验证）← 用过滤后的ETF池
    ↓
任务三（两策略组合）← 用验证后的最优参数 + 过滤后的ETF池
```

## 相关文件

### 数据文件
| 路径 | 说明 | 格式 |
|------|------|------|
| `market_data_qfq/` | 1442只ETF前复权日线数据 | `date,open,high,low,close,volume` (volume=手) |
| `market_data/sh000001.csv` | 上证指数（任务三需要） | `date,open,high,low,close,volume` |
| `stock_data_bfq/600030.csv` | 中信证券（任务三需要） | `date,code,open,close,high,low,volume,...` |
| `etf_list_cache.json` | ETF名称映射 | `[{"code":"159998","name":"计算机ETF"},...]` |

### 策略代码（均在 `stock_data_bfq/` 目录下）
| 文件 | 行数 | 说明 |
|------|------|------|
| `all_etf_rotation.cpp` | 950 | 全ETF MACD轮动（任务一在此修改） |
| `etf_rotation.cpp` | 463 | 中信+纳指轮动（任务三参考） |
| `cyclical_etf_rotation_wf.cpp` | 985 | Walk-forward模板（任务二参考） |

### 编译命令
```bash
# 所有文件在 /ceph/dang_articles/yoj/stock_data_bfq/ 目录下编译
g++ -O3 -std=c++17 -o all_etf_rotation all_etf_rotation.cpp -lpthread
g++ -O3 -std=c++17 -o all_etf_rotation_wf all_etf_rotation_wf.cpp -lpthread
g++ -O3 -std=c++17 -o combined_strategy combined_strategy.cpp -lpthread
g++ -O3 -std=c++17 -o etf_rotation etf_rotation.cpp
```

### 数据路径约定
- 所有路径相对于 `/ceph/dang_articles/yoj/`
- ETF CSV文件名格式：`sh513100.csv` 或 `sz159941.csv`（带交易所前缀）
- `etf_list_cache.json` 中的code不带前缀（如 `"159998"`），代码中需要匹配时要strip前缀
