// ETF轮动策略回测 (V2 - 对齐聚宽逻辑)
// 策略逻辑（来自聚宽）：
//   - 每天开盘运行，用昨日上证收盘价判断：
//   - 上证指数 < 2900：卖出纳指ETF，全仓买入中信证券(600030)
//   - 上证指数 > 3200：卖出中信证券，全仓买入纳指ETF
//   - 2900 <= 上证 <= 3200：维持当前持仓不动
//   - 买卖均以当日开盘价成交（模拟开盘执行）
//
// 与V1的差异：
//   1. 阈值 3200/2900（聚宽用的），而非3400/2900
//   2. 用昨日收盘价判断，今日开盘价成交（聚宽 run_daily(trade, 'open')）
//   3. 增加 159941（广发纳指100ETF）测试
//   4. 同时跑 3200/2900 和 3400/2900 两组阈值对比
//
// 数据源：
//   上证指数: market_data/sh000001.csv  (date,open,high,low,close,volume)
//   中信证券: stock_data_bfq/600030.csv (date,股票代码,open,close,high,low,volume,...)
//   纳指ETF:  market_data/sh513100.csv  (date,open,high,low,close,volume)
//   159941:   market_data/sz159941.csv  (date,open,high,low,close,volume)
//   513180:   market_data/sh513180.csv  (date,open,high,low,close,volume)
//
// 编译: g++ -O3 -std=c++17 -o etf_rotation etf_rotation.cpp

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct DailyBar {
    std::string date;
    double open, high, low, close, volume;
};

struct Trade {
    std::string asset;       // "中信证券" or "纳指ETF"
    std::string buy_date;
    std::string sell_date;
    double buy_price;
    double sell_price;
    double return_pct;
};

struct BacktestResult {
    double initial_capital;
    double final_capital;
    double cumulative_return_pct;
    double annualized_return_pct;
    double max_drawdown_pct;
    std::string max_drawdown_peak_date;
    std::string max_drawdown_trough_date;
    int total_trades;
    int winning_trades;
    double win_rate;
    double total_days;
    double total_years;
    std::string start_date;
    std::string end_date;
    std::vector<Trade> trades;
};

// Load market_data format CSV: date,open,high,low,close,volume
bool load_market_csv(const std::string& path, std::vector<DailyBar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }
    std::string line;
    std::getline(file, line); // skip header
    bars.clear();
    bars.reserve(4096);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, s_open, s_high, s_low, s_close, s_vol;
        std::getline(ss, date, ',');
        std::getline(ss, s_open, ',');
        std::getline(ss, s_high, ',');
        std::getline(ss, s_low, ',');
        std::getline(ss, s_close, ',');
        std::getline(ss, s_vol, ',');
        try {
            DailyBar b;
            b.date = date;
            b.open = std::stod(s_open);
            b.high = std::stod(s_high);
            b.low = std::stod(s_low);
            b.close = std::stod(s_close);
            b.volume = std::stod(s_vol);
            if (b.close > 0) bars.push_back(b);
        } catch (...) {}
    }
    std::sort(bars.begin(), bars.end(),
        [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });
    return !bars.empty();
}

// Load stock_data_bfq format CSV: date,股票代码,open,close,high,low,volume,...
bool load_stock_csv(const std::string& path, std::vector<DailyBar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }
    std::string line;
    std::getline(file, line); // skip header
    bars.clear();
    bars.reserve(4096);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, code, s_open, s_close, s_high, s_low, s_vol;
        std::getline(ss, date, ',');
        std::getline(ss, code, ',');
        std::getline(ss, s_open, ',');
        std::getline(ss, s_close, ',');
        std::getline(ss, s_high, ',');
        std::getline(ss, s_low, ',');
        std::getline(ss, s_vol, ',');
        try {
            DailyBar b;
            b.date = date;
            b.open = std::stod(s_open);
            b.close = std::stod(s_close);
            b.high = std::stod(s_high);
            b.low = std::stod(s_low);
            b.volume = std::stod(s_vol);
            if (b.close > 0) bars.push_back(b);
        } catch (...) {}
    }
    std::sort(bars.begin(), bars.end(),
        [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });
    return !bars.empty();
}

// Build date -> price maps
std::map<std::string, double> build_close_map(const std::vector<DailyBar>& bars) {
    std::map<std::string, double> m;
    for (const auto& b : bars) m[b.date] = b.close;
    return m;
}
std::map<std::string, double> build_open_map(const std::vector<DailyBar>& bars) {
    std::map<std::string, double> m;
    for (const auto& b : bars) m[b.date] = b.open;
    return m;
}

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    return (std::pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

int date_to_days(const std::string& d) {
    if (d.size() < 10) return 0;
    int y = std::stoi(d.substr(0, 4));
    int m = std::stoi(d.substr(5, 2));
    int day = std::stoi(d.substr(8, 2));
    if (m <= 2) { y--; m += 12; }
    return 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + day;
}

// 聚宽逻辑：
//   每天开盘运行 trade()
//   sh_price = 昨日收盘价
//   if sh_price > high_threshold: 卖中信 买纳指 (当日开盘价成交)
//   elif sh_price < low_threshold: 卖纳指 买中信 (当日开盘价成交)
//   else: 不操作
BacktestResult run_rotation(
    const std::vector<DailyBar>& index_bars,
    const std::vector<DailyBar>& citic_bars,
    const std::vector<DailyBar>& nasdaq_bars,
    const std::string& nasdaq_name,
    double threshold_low,
    double threshold_high
) {
    BacktestResult result{};
    result.initial_capital = 1000000.0;

    auto citic_close = build_close_map(citic_bars);
    auto citic_open = build_open_map(citic_bars);
    auto nasdaq_close = build_close_map(nasdaq_bars);
    auto nasdaq_open = build_open_map(nasdaq_bars);
    auto index_close = build_close_map(index_bars);

    // Build ordered list of trading dates where all 3 have data
    struct DayData {
        std::string date;
        double idx_close;  // today's index close (used as "yesterday" for next day)
        double citic_open, citic_close;
        double nasdaq_open, nasdaq_close;
    };
    std::vector<DayData> days;
    for (const auto& bar : index_bars) {
        const auto& d = bar.date;
        if (citic_close.count(d) && citic_open.count(d) &&
            nasdaq_close.count(d) && nasdaq_open.count(d)) {
            days.push_back({d, bar.close,
                           citic_open[d], citic_close[d],
                           nasdaq_open[d], nasdaq_close[d]});
        }
    }

    if (days.size() < 2) {
        std::cerr << "Not enough overlapping data!\n";
        return result;
    }

    // State
    enum State { CASH, HOLD_CITIC, HOLD_NASDAQ };
    State state = CASH;
    double capital = result.initial_capital;
    double shares = 0;
    double buy_price = 0;
    std::string buy_date;

    double peak_capital = capital;
    double max_dd = 0;
    std::string peak_date, dd_peak_date, dd_trough_date;

    result.start_date = days.front().date;
    result.end_date = days.back().date;

    // Day 0: just record close for "yesterday's close" logic
    // From day 1 onwards: use days[i-1].idx_close as "昨日收盘价"
    for (size_t i = 0; i < days.size(); i++) {
        const auto& today = days[i];

        // Track portfolio value at today's close for drawdown
        double current_value = capital;
        if (state == HOLD_CITIC) {
            current_value = shares * today.citic_close;
        } else if (state == HOLD_NASDAQ) {
            current_value = shares * today.nasdaq_close;
        }

        if (current_value > peak_capital) {
            peak_capital = current_value;
            peak_date = today.date;
        }
        double dd = (peak_capital - current_value) / peak_capital;
        if (dd > max_dd) {
            max_dd = dd;
            dd_peak_date = peak_date;
            dd_trough_date = today.date;
        }

        // Skip day 0 — no "yesterday" yet
        if (i == 0) continue;

        double yesterday_idx_close = days[i - 1].idx_close;

        // 聚宽逻辑: check yesterday's close, trade at today's open
        if (yesterday_idx_close > threshold_high) {
            // Should hold 纳指ETF
            if (state == HOLD_CITIC) {
                // Sell 中信 at today's open
                double sell_price = today.citic_open;
                capital = shares * sell_price;
                double ret = (sell_price - buy_price) / buy_price;
                result.trades.push_back({"中信证券", buy_date, today.date, buy_price, sell_price, ret * 100.0});
                if (ret > 0) result.winning_trades++;

                // Buy 纳指 at today's open
                buy_price = today.nasdaq_open;
                shares = capital / buy_price;
                buy_date = today.date;
                state = HOLD_NASDAQ;
            } else if (state == CASH) {
                // Initial buy: 纳指
                buy_price = today.nasdaq_open;
                shares = capital / buy_price;
                buy_date = today.date;
                state = HOLD_NASDAQ;
            }
        } else if (yesterday_idx_close < threshold_low) {
            // Should hold 中信证券
            if (state == HOLD_NASDAQ) {
                // Sell 纳指 at today's open
                double sell_price = today.nasdaq_open;
                capital = shares * sell_price;
                double ret = (sell_price - buy_price) / buy_price;
                result.trades.push_back({nasdaq_name, buy_date, today.date, buy_price, sell_price, ret * 100.0});
                if (ret > 0) result.winning_trades++;

                // Buy 中信 at today's open
                buy_price = today.citic_open;
                shares = capital / buy_price;
                buy_date = today.date;
                state = HOLD_CITIC;
            } else if (state == CASH) {
                // Initial buy: 中信
                buy_price = today.citic_open;
                shares = capital / buy_price;
                buy_date = today.date;
                state = HOLD_CITIC;
            }
        }
        // 2900 <= idx <= 3200: no action
    }

    // Close final position at last date's close
    const auto& last = days.back();
    if (state == HOLD_CITIC) {
        double sell_price = last.citic_close;
        capital = shares * sell_price;
        double ret = (sell_price - buy_price) / buy_price;
        result.trades.push_back({"中信证券", buy_date, last.date, buy_price, sell_price, ret * 100.0});
        if (ret > 0) result.winning_trades++;
    } else if (state == HOLD_NASDAQ) {
        double sell_price = last.nasdaq_close;
        capital = shares * sell_price;
        double ret = (sell_price - buy_price) / buy_price;
        result.trades.push_back({nasdaq_name, buy_date, last.date, buy_price, sell_price, ret * 100.0});
        if (ret > 0) result.winning_trades++;
    }

    result.final_capital = capital;
    result.total_trades = (int)result.trades.size();
    result.cumulative_return_pct = (capital / result.initial_capital - 1.0) * 100.0;

    int total_days = date_to_days(result.end_date) - date_to_days(result.start_date);
    result.total_days = total_days;
    result.total_years = total_days / 365.25;
    result.annualized_return_pct = compute_annualized(capital / result.initial_capital, result.total_years);
    result.max_drawdown_pct = max_dd * 100.0;
    result.max_drawdown_peak_date = dd_peak_date;
    result.max_drawdown_trough_date = dd_trough_date;
    result.win_rate = result.total_trades > 0 ? (double)result.winning_trades / result.total_trades * 100.0 : 0;

    return result;
}

void print_result(const BacktestResult& r, const std::string& label) {
    printf("\n");
    printf("========================================\n");
    printf("  %s\n", label.c_str());
    printf("========================================\n");
    printf("回测区间:     %s ~ %s (%.1f年)\n", r.start_date.c_str(), r.end_date.c_str(), r.total_years);
    printf("初始资金:     ¥%.0f\n", r.initial_capital);
    printf("最终资金:     ¥%.0f\n", r.final_capital);
    printf("累计收益:     %.2f%%\n", r.cumulative_return_pct);
    printf("年化收益:     %.2f%%\n", r.annualized_return_pct);
    printf("最大回撤:     %.2f%%\n", r.max_drawdown_pct);
    printf("  回撤区间:   %s(峰) ~ %s(谷)\n", r.max_drawdown_peak_date.c_str(), r.max_drawdown_trough_date.c_str());
    printf("总交易次数:   %d\n", r.total_trades);
    printf("盈利交易:     %d\n", r.winning_trades);
    printf("胜率:         %.1f%%\n", r.win_rate);
    if (r.total_years > 0) {
        double calmar = r.max_drawdown_pct > 0 ? r.annualized_return_pct / r.max_drawdown_pct : 0;
        printf("Calmar比率:   %.2f\n", calmar);
    }
    printf("\n--- 交易明细 ---\n");
    printf("%-4s %-12s %-12s %-12s %10s %10s %10s\n",
           "序号", "资产", "买入日期", "卖出日期", "买入价", "卖出价", "收益%");
    printf("---------------------------------------------------------------------\n");
    for (int i = 0; i < (int)r.trades.size(); i++) {
        const auto& t = r.trades[i];
        printf("%-4d %-12s %-12s %-12s %10.3f %10.3f %10.2f%%\n",
               i + 1, t.asset.c_str(), t.buy_date.c_str(), t.sell_date.c_str(),
               t.buy_price, t.sell_price, t.return_pct);
    }
}

void print_buy_hold(const std::vector<DailyBar>& bars, const std::string& name,
                    const std::string& start_date, const std::string& end_date) {
    double start_price = 0, end_price = 0;
    for (const auto& b : bars) {
        if (b.date >= start_date && start_price == 0) start_price = b.close;
        if (b.date <= end_date) end_price = b.close;
    }
    if (start_price <= 0) return;
    int days = date_to_days(end_date) - date_to_days(start_date);
    double years = days / 365.25;
    double ratio = end_price / start_price;
    printf("  %-20s  累计: %+.2f%%  年化: %+.2f%%\n", name.c_str(),
           (ratio - 1.0) * 100.0, compute_annualized(ratio, years));
}

int main() {
    const std::string data_dir = "../market_data/";
    const std::string qfq_dir = "../market_data_qfq/";
    const std::string stock_dir = "./";

    std::vector<DailyBar> index_bars, citic_bars, nasdaq513100, nasdaq159941, sh513180;
    // 前复权版本的ETF数据（修正了拆股）
    std::vector<DailyBar> nasdaq513100_qfq, nasdaq159941_qfq;

    printf("加载数据...\n");
    if (!load_market_csv(data_dir + "sh000001.csv", index_bars)) { std::cerr << "Failed: 上证指数\n"; return 1; }
    printf("  上证指数: %zu bars (%s ~ %s)\n", index_bars.size(), index_bars.front().date.c_str(), index_bars.back().date.c_str());

    if (!load_stock_csv(stock_dir + "600030.csv", citic_bars)) { std::cerr << "Failed: 中信证券\n"; return 1; }
    printf("  中信证券(不复权): %zu bars (%s ~ %s)\n", citic_bars.size(), citic_bars.front().date.c_str(), citic_bars.back().date.c_str());

    // 前复权ETF数据（拆股修正后）
    if (!load_market_csv(qfq_dir + "sh513100.csv", nasdaq513100_qfq)) { std::cerr << "Failed: 513100(qfq)\n"; return 1; }
    printf("  513100纳指ETF(前复权): %zu bars (%s ~ %s)\n", nasdaq513100_qfq.size(), nasdaq513100_qfq.front().date.c_str(), nasdaq513100_qfq.back().date.c_str());

    if (!load_market_csv(qfq_dir + "sz159941.csv", nasdaq159941_qfq)) { std::cerr << "Failed: 159941(qfq)\n"; return 1; }
    printf("  159941纳指100ETF(前复权): %zu bars (%s ~ %s)\n", nasdaq159941_qfq.size(), nasdaq159941_qfq.front().date.c_str(), nasdaq159941_qfq.back().date.c_str());

    // ========== 聚宽一致参数: 3200/2900（前复权） ==========
    printf("\n\n###############################################\n");
    printf("# 阈值: 3200/2900 (聚宽一致, 前复权数据)       #\n");
    printf("###############################################\n");

    auto r1a = run_rotation(index_bars, citic_bars, nasdaq513100_qfq, "纳指ETF(513100)", 2900.0, 3200.0);
    print_result(r1a, "中信证券 <-> 纳指ETF(513100) | 3200/2900 前复权");

    auto r2a = run_rotation(index_bars, citic_bars, nasdaq159941_qfq, "纳指100(159941)", 2900.0, 3200.0);
    print_result(r2a, "中信证券 <-> 纳指100ETF(159941) | 3200/2900 前复权 [聚宽同款]");

    // ========== 原始参数: 3400/2900（前复权） ==========
    printf("\n\n###############################################\n");
    printf("# 阈值: 3400/2900 (用户原始, 前复权数据)       #\n");
    printf("###############################################\n");

    auto r1b = run_rotation(index_bars, citic_bars, nasdaq513100_qfq, "纳指ETF(513100)", 2900.0, 3400.0);
    print_result(r1b, "中信证券 <-> 纳指ETF(513100) | 3400/2900 前复权");

    auto r2b = run_rotation(index_bars, citic_bars, nasdaq159941_qfq, "纳指100(159941)", 2900.0, 3400.0);
    print_result(r2b, "中信证券 <-> 纳指100ETF(159941) | 3400/2900 前复权");

    // ========== Buy & Hold 基准 ==========
    printf("\n\n========================================\n");
    printf("  基准对比 (Buy & Hold, 前复权)\n");
    printf("========================================\n");

    printf("\n159941策略区间 (%s ~ %s):\n", r2a.start_date.c_str(), r2a.end_date.c_str());
    print_buy_hold(index_bars, "上证指数", r2a.start_date, r2a.end_date);
    print_buy_hold(citic_bars, "中信证券(不复权)", r2a.start_date, r2a.end_date);
    print_buy_hold(nasdaq159941_qfq, "纳指100(159941前复权)", r2a.start_date, r2a.end_date);
    print_buy_hold(nasdaq513100_qfq, "纳指ETF(513100前复权)", r2a.start_date, r2a.end_date);

    // ========== 总结 ==========
    printf("\n\n========================================\n");
    printf("  总结对比（全部前复权）\n");
    printf("========================================\n");
    printf("%-50s %10s %10s %8s %6s %6s\n",
           "策略", "年化", "最大回撤", "Calmar", "交易", "胜率");
    printf("-----------------------------------------------------------------------------------------------\n");

    auto print_row = [](const char* name, const BacktestResult& r) {
        double calmar = r.max_drawdown_pct > 0 ? r.annualized_return_pct / r.max_drawdown_pct : 0;
        printf("%-50s %9.2f%% %9.2f%% %8.2f %6d %5.1f%%\n",
               name, r.annualized_return_pct, r.max_drawdown_pct, calmar,
               r.total_trades, r.win_rate);
    };

    print_row("中信+513100 | 3200/2900 前复权", r1a);
    print_row("中信+159941 | 3200/2900 前复权 [聚宽同款]", r2a);
    print_row("中信+513100 | 3400/2900 前复权", r1b);
    print_row("中信+159941 | 3400/2900 前复权", r2b);

    printf("\n");
    return 0;
}
