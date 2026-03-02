/**
 * 周期股金字塔建仓策略回测 — C++ 版本
 * =====================================================
 *
 * 策略逻辑（来自 MR Dang 投资方案 + 金字塔建仓）：
 *   - 第1层: 初始建仓 (总资金的 LAYER1_PCT)
 *   - 第2层: 从建仓价跌 DROP_TRIGGER_PCT，加仓 (LAYER2_PCT)
 *   - 第3层: 从建仓价跌 2×DROP_TRIGGER_PCT，加仓 (LAYER3_PCT)
 *   - 第4层: 从建仓价跌 3×DROP_TRIGGER_PCT，满仓 (LAYER4_PCT)
 *   - 止盈: 从均价涨 TAKE_PROFIT_PCT → 全部卖出
 *   - 无止损: 跌到补仓点就补仓，没钱就持仓等待
 *   - 手续费: 每笔交易固定 5 元
 *
 * 数据: stock_data_bfq/ 下的不复权 CSV
 * CSV 格式: date,股票代码,open,close,high,low,volume,成交额,振幅,涨跌幅,涨跌额,换手率
 *
 * 编译: g++ -std=c++17 -O2 -o cyclical_backtest cyclical_backtest.cpp
 * 用法:
 *   ./cyclical_backtest                                    # 默认: 紫金矿业
 *   ./cyclical_backtest --stock 601919                     # 指定股票
 *   ./cyclical_backtest --stock 601899 --start 2020-01-01  # 指定起止
 *   ./cyclical_backtest --all                              # 回测默认9只龙头
 *   ./cyclical_backtest --scan --top 20                    # 全量扫描 Top20
 *   ./cyclical_backtest --drop 0.08 --profit 0.25          # 调参数
 *   ./cyclical_backtest --list                             # 列出本地数据
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// 数据结构
// ============================================================================

struct Bar {
    std::string date;
    double open;
    double close;
    double high;
    double low;
    double volume;
};

struct TradeRecord {
    std::string date;
    double pnl;        // 净利（含手续费）
    double pnl_pct;    // 收益率 %
};

struct BacktestResult {
    std::string stock_code;
    std::string stock_name;
    double total_return;     // %
    double annual_return;    // %
    double max_drawdown;     // %
    int total_trades;        // 完整交易轮次
    int won;
    int lost;
    double win_rate;         // %
    std::vector<TradeRecord> trade_log;
};

// ============================================================================
// 默认股票池
// ============================================================================

static const std::map<std::string, std::string> DEFAULT_STOCKS = {
    {"601899", "紫金矿业"},
    {"601168", "西部矿业"},
    {"600362", "江西铜业"},
    {"600219", "南山铝业"},
    {"000933", "神火股份"},
    {"600989", "宝丰能源"},
    {"000792", "盐湖股份"},
    {"601919", "中远海控"},
    {"600938", "中国海油"},
};

// 全量股票名称映射（从 all_stock.csv 加载）
static std::map<std::string, std::string> ALL_STOCK_NAMES;

static void load_stock_names() {
    std::ifstream fin("all_stock.csv");
    if (!fin.is_open()) return;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        // 跳过 BOM
        if (line.size() >= 3 &&
            (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB &&
            (unsigned char)line[2] == 0xBF) {
            line = line.substr(3);
        }
        auto pos = line.find(',');
        if (pos == std::string::npos) continue;
        std::string code = line.substr(0, pos);
        std::string name = line.substr(pos + 1);
        // 去除名称中的空格
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
        ALL_STOCK_NAMES[code] = name;
    }
}

static std::string get_stock_name(const std::string& code) {
    auto it = ALL_STOCK_NAMES.find(code);
    if (it != ALL_STOCK_NAMES.end()) return it->second;
    auto it2 = DEFAULT_STOCKS.find(code);
    if (it2 != DEFAULT_STOCKS.end()) return it2->second;
    return code;
}

// ============================================================================
// 策略参数
// ============================================================================

struct StrategyParams {
    double layer1_pct      = 0.10;   // 第1层: 10%
    double layer2_pct      = 0.15;   // 第2层: 15%
    double layer3_pct      = 0.25;   // 第3层: 25%
    double layer4_pct      = 0.30;   // 第4层: 30%
    double drop_trigger_pct = 0.10;  // 每跌 10% 加一层
    double take_profit_pct  = 0.30;  // 盈利 30% 止盈
    double commission       = 5.0;   // 每笔固定手续费 5 元
    // 无止损，无冷却期
};

// ============================================================================
// 数据加载
// ============================================================================

static std::string get_data_dir() {
    // 获取可执行文件所在目录 or 当前目录
    // 这里用当前工作目录下的 stock_data_bfq/
    return "stock_data_bfq";
}

static std::vector<Bar> load_csv(const std::string& stock_code,
                                 const std::string& start_date = "",
                                 const std::string& end_date = "") {
    std::string path = get_data_dir() + "/" + stock_code + ".csv";
    std::ifstream fin(path);
    if (!fin.is_open()) {
        return {};
    }

    std::vector<Bar> bars;
    std::string line;
    // 跳过 header
    if (!std::getline(fin, line)) return {};

    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        // CSV: date,股票代码,open,close,high,low,volume,...
        Bar bar;
        std::istringstream ss(line);
        std::string token;

        // date
        if (!std::getline(ss, bar.date, ',')) continue;
        // 股票代码 (skip)
        if (!std::getline(ss, token, ',')) continue;
        // open
        if (!std::getline(ss, token, ',')) continue;
        bar.open = std::stod(token);
        // close
        if (!std::getline(ss, token, ',')) continue;
        bar.close = std::stod(token);
        // high
        if (!std::getline(ss, token, ',')) continue;
        bar.high = std::stod(token);
        // low
        if (!std::getline(ss, token, ',')) continue;
        bar.low = std::stod(token);
        // volume
        if (!std::getline(ss, token, ',')) continue;
        bar.volume = std::stod(token);

        // 日期筛选
        if (!start_date.empty() && bar.date < start_date) continue;
        if (!end_date.empty() && bar.date > end_date) continue;

        if (bar.open <= 0 || bar.close <= 0 || bar.high <= 0 || bar.low <= 0)
            continue;

        bars.push_back(bar);
    }

    // 按日期排序（CSV 通常已排序，但以防万一）
    std::sort(bars.begin(), bars.end(),
              [](const Bar& a, const Bar& b) { return a.date < b.date; });

    return bars;
}

static std::vector<std::string> list_local_stocks() {
    std::string dir = get_data_dir();
    std::vector<std::string> codes;

    if (!fs::exists(dir)) return codes;

    for (auto& entry : fs::directory_iterator(dir)) {
        std::string fname = entry.path().filename().string();
        if (fname.size() > 4 && fname.substr(fname.size() - 4) == ".csv") {
            codes.push_back(fname.substr(0, fname.size() - 4));
        }
    }
    std::sort(codes.begin(), codes.end());
    return codes;
}

// ============================================================================
// 金字塔建仓策略（无止损版本）
// ============================================================================

static std::optional<BacktestResult> run_single_backtest(
    const std::string& stock_code,
    const std::string& stock_name,
    const std::string& start_date,
    const std::string& end_date,
    double initial_cash,
    const StrategyParams& params,
    bool printlog) {

    auto bars = load_csv(stock_code, start_date, end_date);
    if (bars.empty()) {
        if (printlog)
            std::cerr << "  [错误] " << stock_code << " 无有效数据\n";
        return std::nullopt;
    }

    if (printlog) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "  回测: " << stock_name << " (" << stock_code << ")\n";
        if (!start_date.empty() || !end_date.empty())
            std::cout << "  时段: " << (start_date.empty() ? "最早" : start_date)
                      << " ~ " << (end_date.empty() ? "最新" : end_date) << "\n";
        std::cout << "  初始资金: " << std::fixed << std::setprecision(0)
                  << initial_cash << " 元\n";
        std::cout << "  数据类型: 不复权\n";
        std::cout << "  手续费: 每笔 " << params.commission << " 元\n";
        std::cout << "  参数: 补仓跌幅=" << (int)(params.drop_trigger_pct * 100)
                  << "% 止盈=" << (int)(params.take_profit_pct * 100) << "% 无止损\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "  数据: " << bars.size() << " 个交易日 ("
                  << bars.front().date << " ~ " << bars.back().date << ")\n";
    }

    // ---- 回测状态 ----
    double cash = initial_cash;
    int position_shares = 0;            // 持仓股数
    double first_entry_price = 0;       // 首次建仓价
    int current_layer = 0;              // 当前层 0=空仓, 1-4
    double total_invested = 0;          // 累计投入金额（含手续费）
    int total_shares_bought = 0;        // 累计买入股数
    double avg_cost = 0;                // 持仓均价

    // 统计
    std::vector<TradeRecord> trade_log;
    int round_count = 0;               // 完整交易轮次
    int won = 0, lost = 0;
    double peak_value = initial_cash;
    double max_drawdown = 0;

    // ---- 辅助函数 ----
    auto calc_shares = [&](double pct) -> int {
        double total_value = cash + position_shares * bars[0].close; // 近似
        double target_value = total_value * pct;
        double price = bars[0].close; // 占位，实际在循环里用
        if (price <= 0) return 0;
        int shares = (int)(target_value / price);
        shares = (shares / 100) * 100;  // A 股最少 100 股
        return std::max(shares, 0);
    };
    (void)calc_shares; // suppress unused warning, we'll inline the logic

    // ---- 主循环 ----
    int n = (int)bars.size();

    for (int i = 0; i < n; ++i) {
        double price = bars[i].close;

        // 更新净值和回撤
        double current_value = cash + position_shares * price;
        if (current_value > peak_value)
            peak_value = current_value;
        double dd = (peak_value - current_value) / peak_value * 100.0;
        if (dd > max_drawdown)
            max_drawdown = dd;

        // ==================== 持仓中 ====================
        if (position_shares > 0 && current_layer > 0) {

            // --- 止盈检查 ---
            if (avg_cost > 0) {
                double gain_pct = (price - avg_cost) / avg_cost;

                if (gain_pct >= params.take_profit_pct) {
                    // 全部卖出
                    double sell_income = position_shares * price - params.commission;
                    double pnl = sell_income - total_invested;
                    double pnl_pct = (total_invested > 0)
                                     ? (pnl / total_invested * 100.0) : 0.0;

                    if (printlog) {
                        std::cout << "  " << bars[i].date
                                  << " 🎯 止盈! 均价:" << std::fixed
                                  << std::setprecision(2) << avg_cost
                                  << " 现价:" << price
                                  << " 盈利:" << std::setprecision(1)
                                  << (gain_pct * 100) << "%"
                                  << " 净利:" << std::setprecision(2) << pnl
                                  << "\n";
                    }

                    cash += sell_income;
                    trade_log.push_back({bars[i].date, pnl, pnl_pct});
                    round_count++;
                    if (pnl >= 0) won++; else lost++;

                    // 重置状态
                    position_shares = 0;
                    first_entry_price = 0;
                    current_layer = 0;
                    avg_cost = 0;
                    total_invested = 0;
                    total_shares_bought = 0;
                    continue;
                }
            }

            // --- 补仓检查（无止损，到点就补，没钱就持仓）---
            if (first_entry_price > 0) {
                double drop_from_entry =
                    (price - first_entry_price) / first_entry_price;

                auto try_add_layer = [&](int from_layer, double threshold_mult,
                                         double layer_pct, int next_layer) {
                    if (current_layer != from_layer) return;
                    if (drop_from_entry > -params.drop_trigger_pct * threshold_mult)
                        return;

                    double total_value = cash + position_shares * price;
                    double target_value = total_value * layer_pct;
                    if (price <= 0) return;
                    int shares = (int)(target_value / price);
                    shares = (shares / 100) * 100;

                    if (shares < 100) return;

                    double cost = shares * price + params.commission;
                    if (cost > cash) {
                        // 没钱了，能买多少买多少
                        shares = (int)((cash - params.commission) / price);
                        shares = (shares / 100) * 100;
                        if (shares < 100) {
                            // 连 1 手都买不起，持仓等待
                            if (printlog) {
                                std::cout << "  " << bars[i].date
                                          << " ⚠️ 第" << next_layer
                                          << "层信号触发 跌幅:"
                                          << std::setprecision(1)
                                          << (drop_from_entry * 100) << "%"
                                          << " 但现金不足，继续持仓\n";
                            }
                            // 标记层级已触发，避免重复输出
                            current_layer = next_layer;
                            return;
                        }
                        cost = shares * price + params.commission;
                    }

                    cash -= cost;
                    position_shares += shares;
                    total_invested += cost;
                    total_shares_bought += shares;
                    avg_cost = total_invested / total_shares_bought;
                    current_layer = next_layer;

                    if (printlog) {
                        std::cout << "  " << bars[i].date
                                  << " ⚠️ 补仓第" << next_layer << "层"
                                  << " 跌幅:" << std::setprecision(1)
                                  << (drop_from_entry * 100) << "%"
                                  << " 加仓" << shares << "股"
                                  << " 均价:" << std::setprecision(2) << avg_cost
                                  << "\n";
                    }
                };

                try_add_layer(1, 1.0, params.layer2_pct, 2);
                try_add_layer(2, 2.0, params.layer3_pct, 3);
                try_add_layer(3, 3.0, params.layer4_pct, 4);
            }
        }

        // ==================== 空仓 → 寻找建仓机会 ====================
        else if (position_shares == 0 && current_layer == 0) {
            if (i < 59) continue;  // 数据不够 60 根

            // 计算 60 日高点和低点
            double high_60 = -1e18, low_60 = 1e18, low_5 = 1e18;
            for (int j = i - 59; j <= i; ++j) {
                if (bars[j].high > high_60) high_60 = bars[j].high;
                if (bars[j].low < low_60) low_60 = bars[j].low;
            }
            for (int j = std::max(0, i - 4); j <= i; ++j) {
                if (bars[j].low < low_5) low_5 = bars[j].low;
            }

            double drawdown = (high_60 > 0) ? (price - high_60) / high_60 : 0;
            bool near_low = low_5 <= low_60 * 1.03;

            if (drawdown <= -0.15 && near_low) {
                double total_value = cash; // 空仓时 total_value = cash
                double target_value = total_value * params.layer1_pct;
                int shares = (int)(target_value / price);
                shares = (shares / 100) * 100;

                if (shares >= 100) {
                    double cost = shares * price + params.commission;
                    if (cost > cash) {
                        shares = (int)((cash - params.commission) / price);
                        shares = (shares / 100) * 100;
                        cost = shares * price + params.commission;
                    }
                    if (shares >= 100 && cost <= cash) {
                        cash -= cost;
                        position_shares = shares;
                        first_entry_price = price;
                        current_layer = 1;
                        total_invested = cost;
                        total_shares_bought = shares;
                        avg_cost = cost / shares;

                        if (printlog) {
                            std::cout << "  " << bars[i].date
                                      << " 📈 首次建仓 价格:"
                                      << std::setprecision(2) << price
                                      << " 60日回撤:"
                                      << std::setprecision(1)
                                      << (drawdown * 100) << "%"
                                      << " 买入" << shares << "股\n";
                        }
                    }
                }
            }
        }
    }

    // ---- 回测结束，如果还有持仓，按最后收盘价计算浮动盈亏 ----
    double end_value = cash;
    if (position_shares > 0 && n > 0) {
        end_value += position_shares * bars[n - 1].close;
    }

    double total_return = (end_value - initial_cash) / initial_cash * 100.0;

    // 年化收益（简化计算）
    double years = 0;
    if (n > 1) {
        // 用交易日数 / 250 估算年数
        years = (double)n / 250.0;
    }
    double annual_return = 0;
    if (years > 0) {
        double ratio = end_value / initial_cash;
        if (ratio > 0)
            annual_return = (std::pow(ratio, 1.0 / years) - 1.0) * 100.0;
    }

    if (printlog) {
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "  回测结果: " << stock_name << "\n";
        std::cout << std::string(60, '-') << "\n";
        std::cout << std::fixed;
        std::cout << "  期初资金:    " << std::setw(12) << std::setprecision(2)
                  << initial_cash << " 元\n";
        std::cout << "  期末资金:    " << std::setw(12) << std::setprecision(2)
                  << end_value << " 元\n";

        if (position_shares > 0) {
            double float_pnl = position_shares * bars[n - 1].close - total_invested;
            std::cout << "  [持仓中]     持有 " << position_shares << " 股"
                      << " 均价:" << std::setprecision(2) << avg_cost
                      << " 现价:" << bars[n - 1].close
                      << " 浮盈:" << std::setprecision(2) << float_pnl << "\n";
        }

        std::cout << "  总收益率:    " << std::setw(11) << std::setprecision(2)
                  << total_return << "%\n";
        std::cout << "  年化收益:    " << std::setw(11) << std::setprecision(2)
                  << annual_return << "%\n";
        std::cout << "  最大回撤:    " << std::setw(11) << std::setprecision(2)
                  << max_drawdown << "%\n";

        std::cout << "  完整交易轮次:" << std::setw(11) << round_count << "\n";
        std::cout << "  盈利次数:    " << std::setw(11) << won << "\n";
        std::cout << "  亏损次数:    " << std::setw(11) << lost << "\n";
        double wr = (round_count > 0) ? (double)won / round_count * 100 : 0;
        std::cout << "  胜率:        " << std::setw(10) << std::setprecision(1)
                  << wr << "%\n";

        // 交易明细
        if (!trade_log.empty()) {
            std::cout << "\n  交易明细:\n";
            for (auto& t : trade_log) {
                const char* emoji = (t.pnl >= 0) ? "✅" : "❌";
                std::cout << "    " << emoji << " " << t.date
                          << "  净利:" << std::setw(10) << std::setprecision(2)
                          << std::showpos << t.pnl << std::noshowpos
                          << "  收益率:" << std::setw(6) << std::setprecision(1)
                          << std::showpos << t.pnl_pct << std::noshowpos
                          << "%\n";
            }
        }
        std::cout << std::string(60, '=') << "\n\n";
    }

    BacktestResult result;
    result.stock_code = stock_code;
    result.stock_name = stock_name;
    result.total_return = total_return;
    result.annual_return = annual_return;
    result.max_drawdown = max_drawdown;
    result.total_trades = round_count;
    result.won = won;
    result.lost = lost;
    result.win_rate = (round_count > 0) ? (double)won / round_count * 100 : 0;
    result.trade_log = std::move(trade_log);

    return result;
}

// ============================================================================
// 批量回测
// ============================================================================

static void run_all_default(const std::string& start_date,
                            const std::string& end_date,
                            double initial_cash,
                            const StrategyParams& params,
                            bool printlog) {
    std::vector<BacktestResult> results;

    for (auto& [code, name] : DEFAULT_STOCKS) {
        auto r = run_single_backtest(code, name, start_date, end_date,
                                     initial_cash, params, printlog);
        if (r.has_value())
            results.push_back(r.value());
    }

    if (results.empty()) return;

    // 汇总表
    std::cout << "\n" << std::string(60, '#') << "\n";
    std::cout << "  逐只回测汇总\n";
    std::cout << std::string(60, '#') << "\n";
    std::cout << std::fixed;
    std::cout << "  " << std::left << std::setw(14) << "股票"
              << std::right << std::setw(9) << "总收益%"
              << std::setw(9) << "年化%"
              << std::setw(9) << "回撤%"
              << std::setw(7) << "交易数"
              << std::setw(7) << "胜率%"
              << "\n";
    std::cout << "  " << std::string(55, '-') << "\n";

    for (auto& r : results) {
        std::cout << "  " << std::left << std::setw(14) << r.stock_name
                  << std::right
                  << std::setw(8) << std::setprecision(1)
                  << std::showpos << r.total_return << std::noshowpos
                  << std::setw(9) << std::setprecision(1)
                  << std::showpos << r.annual_return << std::noshowpos
                  << std::setw(9) << std::setprecision(1) << r.max_drawdown
                  << std::setw(7) << r.total_trades
                  << std::setw(7) << std::setprecision(1) << r.win_rate
                  << "\n";
    }
    std::cout << std::string(60, '#') << "\n\n";
}

static void run_scan(const std::string& start_date,
                     const std::string& end_date,
                     double initial_cash,
                     const StrategyParams& params,
                     int top_n) {
    auto all_codes = list_local_stocks();

    std::cout << "\n" << std::string(60, '#') << "\n";
    std::cout << "  全量扫描回测: 共 " << all_codes.size() << " 只股票\n";
    if (!start_date.empty() || !end_date.empty())
        std::cout << "  时段: " << (start_date.empty() ? "最早" : start_date)
                  << " ~ " << (end_date.empty() ? "最新" : end_date) << "\n";
    std::cout << std::fixed << std::setprecision(0)
              << "  初始资金: " << initial_cash << " 元\n";
    std::cout << std::string(60, '#') << "\n\n";

    std::vector<BacktestResult> results;
    int skipped = 0;

    for (size_t i = 0; i < all_codes.size(); ++i) {
        if ((i + 1) % 500 == 0 || i == 0) {
            std::cout << "  进度: " << (i + 1) << "/" << all_codes.size()
                      << " ...\n";
        }

        std::string name = get_stock_name(all_codes[i]);

        auto r = run_single_backtest(all_codes[i], name, start_date, end_date,
                                     initial_cash, params, false);
        if (r.has_value())
            results.push_back(r.value());
        else
            skipped++;
    }

    // 按总收益率排序
    std::sort(results.begin(), results.end(),
              [](const BacktestResult& a, const BacktestResult& b) {
                  return a.total_return > b.total_return;
              });

    // 显示结果
    size_t display_n = (top_n > 0) ? std::min((size_t)top_n, results.size())
                                   : results.size();

    std::cout << "\n" << std::string(70, '#') << "\n";
    if (top_n > 0)
        std::cout << "  扫描结果 Top " << top_n << " (共 " << results.size()
                  << " 只有效)\n";
    else
        std::cout << "  扫描结果 (共 " << results.size() << " 只有效, "
                  << skipped << " 只跳过)\n";
    std::cout << std::string(70, '#') << "\n";

    std::cout << std::fixed;
    std::cout << "  " << std::left << std::setw(14) << "股票"
              << std::setw(8) << "代码"
              << std::right << std::setw(9) << "总收益%"
              << std::setw(9) << "年化%"
              << std::setw(9) << "回撤%"
              << std::setw(7) << "交易数"
              << std::setw(7) << "胜率%"
              << "\n";
    std::cout << "  " << std::string(63, '-') << "\n";

    for (size_t i = 0; i < display_n; ++i) {
        auto& r = results[i];
        std::cout << "  " << std::left << std::setw(14) << r.stock_name
                  << std::setw(8) << r.stock_code
                  << std::right
                  << std::setw(8) << std::setprecision(1)
                  << std::showpos << r.total_return << std::noshowpos
                  << std::setw(9) << std::setprecision(1)
                  << std::showpos << r.annual_return << std::noshowpos
                  << std::setw(9) << std::setprecision(1) << r.max_drawdown
                  << std::setw(7) << r.total_trades
                  << std::setw(7) << std::setprecision(1) << r.win_rate
                  << "\n";
    }
    std::cout << std::string(70, '#') << "\n";

    // 统计摘要
    if (!results.empty()) {
        double avg_ret = 0;
        for (auto& r : results) avg_ret += r.total_return;
        avg_ret /= results.size();

        int positive = 0, negative = 0, zero = 0;
        for (auto& r : results) {
            if (r.total_return > 0) positive++;
            else if (r.total_return < 0) negative++;
            else if (r.total_trades == 0) zero++;
        }

        std::cout << "\n  📊 统计摘要:\n";
        std::cout << "     平均收益率: " << std::showpos << std::setprecision(2)
                  << avg_ret << std::noshowpos << "%\n";
        std::cout << "     盈利股票: " << positive << " 只"
                  << "  亏损股票: " << negative << " 只"
                  << "  无交易: " << zero << " 只\n";
        std::cout << "     最佳: " << results.front().stock_name
                  << "(" << results.front().stock_code << ") "
                  << std::showpos << std::setprecision(1)
                  << results.front().total_return << std::noshowpos << "%\n";
        std::cout << "     最差: " << results.back().stock_name
                  << "(" << results.back().stock_code << ") "
                  << std::showpos << std::setprecision(1)
                  << results.back().total_return << std::noshowpos << "%\n\n";
    }
}

// ============================================================================
// CLI
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << R"(
周期股金字塔建仓策略回测 (C++ 版, 无止损, 固定手续费5元/笔)

用法:
  )" << prog << R"(                                      # 默认: 紫金矿业
  )" << prog << R"( --stock 601919                       # 指定股票
  )" << prog << R"( --stock 601919 --start 2020-01-01    # 指定起止
  )" << prog << R"( --all                                # 逐只回测默认9只龙头股
  )" << prog << R"( --scan --top 20                      # 全量扫描 Top20
  )" << prog << R"( --drop 0.08 --profit 0.25            # 调参数
  )" << prog << R"( --list                               # 列出本地数据
  )" << prog << R"( --quiet                              # 安静模式

策略说明:
  金字塔建仓: 第1层(10%) → 跌10%加第2层(15%) → 再跌10%加第3层(25%) → 再跌10%满仓(30%)
  止盈: 从均价涨30%全部卖出
  无止损: 到点补仓，没钱就持仓等待
  手续费: 每笔固定 5 元
)";
}

int main(int argc, char* argv[]) {
    // 加载股票名称映射
    load_stock_names();

    // 解析命令行
    std::string stock_code = "601899";
    std::string start_date, end_date;
    double initial_cash = 100000;
    StrategyParams params;
    bool do_all = false;
    bool do_scan = false;
    bool do_list = false;
    bool quiet = false;
    int top_n = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--stock" && i + 1 < argc) {
            stock_code = argv[++i];
        } else if (arg == "--start" && i + 1 < argc) {
            start_date = argv[++i];
        } else if (arg == "--end" && i + 1 < argc) {
            end_date = argv[++i];
        } else if (arg == "--cash" && i + 1 < argc) {
            initial_cash = std::stod(argv[++i]);
        } else if (arg == "--drop" && i + 1 < argc) {
            params.drop_trigger_pct = std::stod(argv[++i]);
        } else if (arg == "--profit" && i + 1 < argc) {
            params.take_profit_pct = std::stod(argv[++i]);
        } else if (arg == "--commission" && i + 1 < argc) {
            params.commission = std::stod(argv[++i]);
        } else if (arg == "--all") {
            do_all = true;
        } else if (arg == "--scan") {
            do_scan = true;
        } else if (arg == "--top" && i + 1 < argc) {
            top_n = std::stoi(argv[++i]);
        } else if (arg == "--list") {
            do_list = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else {
            std::cerr << "未知参数: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (do_list) {
        auto codes = list_local_stocks();
        std::cout << "\n本地数据目录: " << get_data_dir() << "\n";
        std::cout << "共 " << codes.size() << " 只股票:\n\n";
        for (auto& code : codes) {
            std::string sname = get_stock_name(code);
            std::cout << "  " << code;
            if (sname != code)
                std::cout << "  " << sname;
            std::cout << "\n";
        }
        std::cout << "\n";
    } else if (do_scan) {
        run_scan(start_date, end_date, initial_cash, params, top_n);
    } else if (do_all) {
        run_all_default(start_date, end_date, initial_cash, params, !quiet);
    } else {
        std::string name = get_stock_name(stock_code);
        run_single_backtest(stock_code, name, start_date, end_date,
                            initial_cash, params, !quiet);
    }

    return 0;
}
