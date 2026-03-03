// 最大回撤分析工具
// 分析两个策略配置的回撤：
//   1. 基线: 2900/3200 无指标
//   2. 最优: 2900/3100 + RSI(14,30,70) + MA(10) + Confirm(1)
//
// 输出：
//   - 所有超过5%的回撤事件（峰值日期、谷底日期、恢复日期、持仓状态、上证点位）
//   - 最大回撤期间的逐日资金曲线
//
// 编译: g++ -O3 -std=c++17 -o drawdown_analysis drawdown_analysis.cpp

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct DailyBar {
    std::string date;
    double open, high, low, close, volume;
};

bool load_market_csv(const std::string& path, std::vector<DailyBar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    std::getline(file, line);
    bars.clear();
    bars.reserve(4096);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, s1, s2, s3, s4, s5;
        std::getline(ss, date, ',');
        std::getline(ss, s1, ',');
        std::getline(ss, s2, ',');
        std::getline(ss, s3, ',');
        std::getline(ss, s4, ',');
        std::getline(ss, s5, ',');
        try {
            bars.push_back({date, std::stod(s1), std::stod(s2), std::stod(s3), std::stod(s4), std::stod(s5)});
        } catch (...) {}
    }
    std::sort(bars.begin(), bars.end(), [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });
    return !bars.empty();
}

bool load_stock_csv(const std::string& path, std::vector<DailyBar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    std::getline(file, line);
    bars.clear();
    bars.reserve(4096);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, code, s_o, s_c, s_h, s_l, s_v;
        std::getline(ss, date, ',');
        std::getline(ss, code, ',');
        std::getline(ss, s_o, ',');
        std::getline(ss, s_c, ',');
        std::getline(ss, s_h, ',');
        std::getline(ss, s_l, ',');
        std::getline(ss, s_v, ',');
        try {
            bars.push_back({date, std::stod(s_o), std::stod(s_h), std::stod(s_l), std::stod(s_c), std::stod(s_v)});
        } catch (...) {}
    }
    std::sort(bars.begin(), bars.end(), [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });
    return !bars.empty();
}

std::vector<double> compute_rsi(const std::vector<double>& closes, int period) {
    std::vector<double> rsi(closes.size(), 50.0);
    if (period <= 0 || (int)closes.size() <= period) return rsi;
    double avg_gain = 0, avg_loss = 0;
    for (int i = 1; i <= period; i++) {
        double diff = closes[i] - closes[i-1];
        if (diff > 0) avg_gain += diff;
        else avg_loss -= diff;
    }
    avg_gain /= period;
    avg_loss /= period;
    rsi[period] = avg_loss == 0 ? 100.0 : 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
    for (int i = period + 1; i < (int)closes.size(); i++) {
        double diff = closes[i] - closes[i-1];
        double gain = diff > 0 ? diff : 0;
        double loss = diff < 0 ? -diff : 0;
        avg_gain = (avg_gain * (period - 1) + gain) / period;
        avg_loss = (avg_loss * (period - 1) + loss) / period;
        rsi[i] = avg_loss == 0 ? 100.0 : 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
    }
    return rsi;
}

std::vector<double> compute_sma(const std::vector<double>& data, int period) {
    std::vector<double> ma(data.size(), 0);
    if (period <= 0) return ma;
    double sum = 0;
    for (int i = 0; i < (int)data.size(); i++) {
        sum += data[i];
        if (i >= period) sum -= data[i - period];
        if (i >= period - 1) ma[i] = sum / period;
        else ma[i] = sum / (i + 1);
    }
    return ma;
}

struct DayData {
    std::string date;
    double idx_close;
    double citic_open, citic_close;
    double nasdaq_open, nasdaq_close;
    double idx_rsi;
    double nasdaq_ma;
};

struct Config {
    double low_thresh, high_thresh;
    int rsi_period, rsi_buy_thresh, rsi_sell_thresh;
    int ma_period, confirm_days;
};

// Drawdown event
struct DrawdownEvent {
    std::string peak_date;
    std::string trough_date;
    std::string recovery_date;  // "" if not recovered
    double peak_value;
    double trough_value;
    double drawdown_pct;
    int peak_idx, trough_idx, recovery_idx;  // indices into days[]
    std::string holding_at_peak;   // what asset at peak
    std::string holding_at_trough; // what asset at trough
    double idx_at_peak;    // 上证指数 at peak
    double idx_at_trough;  // 上证指数 at trough
    int duration_to_trough;  // trading days peak→trough
    int duration_to_recovery;  // trading days peak→recovery (-1 if not recovered)
};

// Portfolio snapshot per day
struct DaySnapshot {
    std::string date;
    double portfolio_value;
    double drawdown_pct;  // from running peak
    double idx_close;
    const char* holding;  // "现金", "中信证券", "纳指ETF"
    const char* action;   // "买入中信", "买入纳指", "卖出中信", "卖出纳指", ""
};

int main() {
    printf("加载数据...\n");
    
    std::vector<DailyBar> index_bars, citic_bars, nasdaq_bars;
    if (!load_market_csv("../market_data/sh000001.csv", index_bars)) { fprintf(stderr, "Failed: index\n"); return 1; }
    if (!load_stock_csv("600030.csv", citic_bars)) { fprintf(stderr, "Failed: citic\n"); return 1; }
    if (!load_market_csv("../stock_data_qfq/513100.csv", nasdaq_bars)) { fprintf(stderr, "Failed: nasdaq\n"); return 1; }
    
    // Build maps
    std::map<std::string, double> cc, co, nc, no_;
    for (auto& b : citic_bars) { cc[b.date] = b.close; co[b.date] = b.open; }
    for (auto& b : nasdaq_bars) { nc[b.date] = b.close; no_[b.date] = b.open; }
    
    // Unified day list
    struct RawDay { std::string date; double idx_close, citic_open, citic_close, nasdaq_open, nasdaq_close; };
    std::vector<RawDay> raw_days;
    for (auto& bar : index_bars) {
        if (cc.count(bar.date) && nc.count(bar.date)) {
            raw_days.push_back({bar.date, bar.close, co[bar.date], cc[bar.date], no_[bar.date], nc[bar.date]});
        }
    }
    printf("  重叠交易日: %zu (%s ~ %s)\n\n", raw_days.size(), raw_days.front().date.c_str(), raw_days.back().date.c_str());
    
    // Pre-compute indicators
    std::vector<double> idx_closes, nasdaq_closes;
    for (auto& d : raw_days) { idx_closes.push_back(d.idx_close); nasdaq_closes.push_back(d.nasdaq_close); }
    
    // Run analysis for both configs
    struct StrategyConfig {
        const char* name;
        Config cfg;
    };
    
    std::vector<StrategyConfig> strategies = {
        {"基线 (2900/3200 无指标)", {2900, 3200, 0, 0, 0, 0, 1}},
        {"最优 (2900/3100 RSI(14,30,70) MA(10) Confirm(1))", {2900, 3100, 14, 30, 70, 10, 1}},
    };
    
    for (auto& strat : strategies) {
        auto& cfg = strat.cfg;
        
        // Build day data with indicators
        auto rsi_vec = cfg.rsi_period > 0 ? compute_rsi(idx_closes, cfg.rsi_period) : std::vector<double>(raw_days.size(), 50.0);
        auto ma_vec = cfg.ma_period > 0 ? compute_sma(nasdaq_closes, cfg.ma_period) : std::vector<double>(raw_days.size(), 0);
        
        std::vector<DayData> days(raw_days.size());
        for (size_t i = 0; i < raw_days.size(); i++) {
            days[i] = {raw_days[i].date, raw_days[i].idx_close,
                       raw_days[i].citic_open, raw_days[i].citic_close,
                       raw_days[i].nasdaq_open, raw_days[i].nasdaq_close,
                       rsi_vec[i], cfg.ma_period > 0 ? ma_vec[i] : 0};
        }
        
        // ---- Run backtest with full tracking ----
        enum State { CASH, HOLD_CITIC, HOLD_NASDAQ };
        State state = CASH;
        double capital = 1000000.0;
        double shares = 0;
        double buy_price = 0;
        
        double peak_capital = capital;
        int confirm_buy_citic = 0, confirm_buy_nasdaq = 0;
        int needed = std::max(1, cfg.confirm_days);
        
        // Track snapshots & drawdown events
        std::vector<DaySnapshot> snapshots;
        std::vector<DrawdownEvent> dd_events;
        
        // Drawdown tracking
        std::string peak_date = days[0].date;
        int peak_idx = 0;
        bool in_drawdown = false;
        DrawdownEvent current_dd{};
        
        for (size_t i = 0; i < days.size(); i++) {
            const auto& today = days[i];
            const char* action = "";
            
            // Portfolio value
            double current_value = capital;
            if (state == HOLD_CITIC) current_value = shares * today.citic_close;
            else if (state == HOLD_NASDAQ) current_value = shares * today.nasdaq_close;
            
            // Check for new peak
            if (current_value > peak_capital) {
                // If we were in a drawdown, it's now recovered
                if (in_drawdown && current_dd.drawdown_pct >= 5.0) {
                    current_dd.recovery_date = today.date;
                    current_dd.recovery_idx = (int)i;
                    current_dd.duration_to_recovery = (int)i - current_dd.peak_idx;
                    dd_events.push_back(current_dd);
                }
                in_drawdown = false;
                peak_capital = current_value;
                peak_date = today.date;
                peak_idx = (int)i;
            }
            
            double dd_pct = (peak_capital - current_value) / peak_capital * 100.0;
            
            // Track drawdown event
            if (dd_pct > 0.1 && !in_drawdown) {
                // Start new drawdown
                in_drawdown = true;
                current_dd = {};
                current_dd.peak_date = peak_date;
                current_dd.peak_value = peak_capital;
                current_dd.peak_idx = peak_idx;
                current_dd.trough_value = current_value;
                current_dd.trough_date = today.date;
                current_dd.trough_idx = (int)i;
                current_dd.drawdown_pct = dd_pct;
                current_dd.idx_at_peak = days[peak_idx].idx_close;
                current_dd.idx_at_trough = today.idx_close;
                const char* h = state == HOLD_CITIC ? "中信证券" : (state == HOLD_NASDAQ ? "纳指ETF" : "现金");
                current_dd.holding_at_peak = h;
                current_dd.holding_at_trough = h;
            } else if (in_drawdown && dd_pct > current_dd.drawdown_pct) {
                // Deeper trough
                current_dd.trough_value = current_value;
                current_dd.trough_date = today.date;
                current_dd.trough_idx = (int)i;
                current_dd.drawdown_pct = dd_pct;
                current_dd.idx_at_trough = today.idx_close;
                current_dd.duration_to_trough = (int)i - current_dd.peak_idx;
                const char* h = state == HOLD_CITIC ? "中信证券" : (state == HOLD_NASDAQ ? "纳指ETF" : "现金");
                current_dd.holding_at_trough = h;
            }
            
            // ---- Trading logic (same as optimizer) ----
            if (i > 0) {
                double yesterday_idx = days[i-1].idx_close;
                double yesterday_rsi = days[i-1].idx_rsi;
                
                bool want_citic = yesterday_idx < cfg.low_thresh;
                bool want_nasdaq = yesterday_idx > cfg.high_thresh;
                
                if (cfg.rsi_period > 0 && want_citic) {
                    if (yesterday_rsi > cfg.rsi_buy_thresh) want_citic = false;
                }
                if (cfg.rsi_period > 0 && want_nasdaq) {
                    if (yesterday_rsi < cfg.rsi_sell_thresh) want_nasdaq = false;
                }
                if (cfg.ma_period > 0 && want_nasdaq) {
                    if (days[i-1].nasdaq_close < days[i-1].nasdaq_ma) want_nasdaq = false;
                }
                
                if (want_citic) { confirm_buy_citic++; confirm_buy_nasdaq = 0; }
                else if (want_nasdaq) { confirm_buy_nasdaq++; confirm_buy_citic = 0; }
                else { confirm_buy_citic = 0; confirm_buy_nasdaq = 0; }
                
                bool do_buy_citic = (confirm_buy_citic >= needed) && (state != HOLD_CITIC);
                bool do_buy_nasdaq = (confirm_buy_nasdaq >= needed) && (state != HOLD_NASDAQ);
                
                if (do_buy_citic) {
                    if (state == HOLD_NASDAQ) {
                        capital = shares * today.nasdaq_open;
                        action = "卖纳指→买中信";
                    } else {
                        action = "买入中信";
                    }
                    buy_price = today.citic_open;
                    shares = capital / buy_price;
                    state = HOLD_CITIC;
                } else if (do_buy_nasdaq) {
                    if (state == HOLD_CITIC) {
                        capital = shares * today.citic_open;
                        action = "卖中信→买纳指";
                    } else {
                        action = "买入纳指";
                    }
                    buy_price = today.nasdaq_open;
                    shares = capital / buy_price;
                    state = HOLD_NASDAQ;
                }
            }
            
            // Recalc value after trade (at close)
            current_value = capital;
            if (state == HOLD_CITIC) current_value = shares * today.citic_close;
            else if (state == HOLD_NASDAQ) current_value = shares * today.nasdaq_close;
            
            const char* holding = state == HOLD_CITIC ? "中信证券" : (state == HOLD_NASDAQ ? "纳指ETF" : "现金");
            snapshots.push_back({today.date, current_value, dd_pct, today.idx_close, holding, action});
        }
        
        // Handle unrecovered drawdown at end
        if (in_drawdown && current_dd.drawdown_pct >= 5.0) {
            current_dd.recovery_date = "";
            current_dd.recovery_idx = -1;
            current_dd.duration_to_recovery = -1;
            dd_events.push_back(current_dd);
        }
        
        // ---- Output ----
        printf("╔══════════════════════════════════════════════════════════════════╗\n");
        printf("║  回撤分析: %s\n", strat.name);
        printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
        
        // Sort by drawdown severity
        std::sort(dd_events.begin(), dd_events.end(),
                  [](const DrawdownEvent& a, const DrawdownEvent& b) { return a.drawdown_pct > b.drawdown_pct; });
        
        printf("所有超过5%%的回撤事件 (按严重程度排序):\n");
        printf("┌────┬──────────────┬──────────────┬──────────────┬──────────┬────────────┬────────────┬──────────────────┬──────────────────┐\n");
        printf("│ #  │   峰值日期   │   谷底日期   │   恢复日期   │ 回撤幅度 │ 峰值→谷(天)│ 峰值→恢复  │  谷底持仓         │ 上证(峰→谷)      │\n");
        printf("├────┼──────────────┼──────────────┼──────────────┼──────────┼────────────┼────────────┼──────────────────┼──────────────────┤\n");
        
        for (int i = 0; i < (int)dd_events.size(); i++) {
            auto& e = dd_events[i];
            char recovery[20];
            char dur_recovery[20];
            if (e.recovery_date.empty()) {
                snprintf(recovery, sizeof(recovery), "  未恢复    ");
                snprintf(dur_recovery, sizeof(dur_recovery), "    未恢复  ");
            } else {
                snprintf(recovery, sizeof(recovery), " %s ", e.recovery_date.c_str());
                snprintf(dur_recovery, sizeof(dur_recovery), "  %4d天    ", e.duration_to_recovery);
            }
            printf("│ %-2d │ %s │ %s │%s│  %5.2f%%  │  %4d天    │%s│ %-16s │ %.0f→%.0f       │\n",
                   i + 1, e.peak_date.c_str(), e.trough_date.c_str(), recovery,
                   e.drawdown_pct, e.duration_to_trough, dur_recovery,
                   e.holding_at_trough.c_str(), e.idx_at_peak, e.idx_at_trough);
        }
        printf("└────┴──────────────┴──────────────┴──────────────┴──────────┴────────────┴────────────┴──────────────────┴──────────────────┘\n\n");
        
        // Detailed daily view around max drawdown
        if (!dd_events.empty()) {
            auto& max_dd = dd_events[0];
            printf("━━━ 最大回撤详细分析 (%.2f%%) ━━━\n", max_dd.drawdown_pct);
            printf("峰值: %s  资金: ¥%.0f  上证: %.2f\n", max_dd.peak_date.c_str(), max_dd.peak_value, max_dd.idx_at_peak);
            printf("谷底: %s  资金: ¥%.0f  上证: %.2f\n", max_dd.trough_date.c_str(), max_dd.trough_value, max_dd.idx_at_trough);
            if (!max_dd.recovery_date.empty()) {
                printf("恢复: %s  (%d个交易日后恢复)\n", max_dd.recovery_date.c_str(), max_dd.duration_to_recovery);
            } else {
                printf("恢复: 截至回测结束尚未恢复\n");
            }
            printf("\n");
            
            // Print daily snapshots around max drawdown: from 10 days before peak to 10 days after trough (or recovery)
            int start = std::max(0, max_dd.peak_idx - 10);
            int end_idx = max_dd.recovery_idx > 0 ? std::min((int)snapshots.size() - 1, max_dd.recovery_idx + 5)
                                                   : std::min((int)snapshots.size() - 1, max_dd.trough_idx + 20);
            
            printf("逐日明细 (峰值前10天 ~ 谷底/恢复后):\n");
            printf("┌──────────────┬────────────────┬──────────┬──────────┬──────────────┬──────────────────┐\n");
            printf("│     日期     │    资金(万元)  │  回撤%%   │  上证指数│   持仓       │   操作           │\n");
            printf("├──────────────┼────────────────┼──────────┼──────────┼──────────────┼──────────────────┤\n");
            
            for (int i = start; i <= end_idx; i++) {
                auto& s = snapshots[i];
                const char* marker = "";
                if (s.date == max_dd.peak_date) marker = " ◆峰值";
                else if (s.date == max_dd.trough_date) marker = " ◆谷底";
                else if (!max_dd.recovery_date.empty() && s.date == max_dd.recovery_date) marker = " ◆恢复";
                
                char action_str[64] = "";
                if (s.action[0] != '\0') {
                    snprintf(action_str, sizeof(action_str), " %s", s.action);
                }
                if (marker[0] != '\0') {
                    if (action_str[0] != '\0') {
                        char tmp[128];
                        snprintf(tmp, sizeof(tmp), "%s%s", action_str, marker);
                        snprintf(action_str, sizeof(action_str), "%s", tmp);
                    } else {
                        snprintf(action_str, sizeof(action_str), "%s", marker);
                    }
                }
                
                printf("│ %s │ %12.2f   │  %5.2f%%  │ %7.2f  │ %-12s│%-18s│\n",
                       s.date.c_str(), s.portfolio_value / 10000.0, s.drawdown_pct,
                       s.idx_close, s.holding, action_str);
            }
            printf("└──────────────┴────────────────┴──────────┴──────────┴──────────────┴──────────────────┘\n\n");
            
            // Analysis summary
            printf("━━━ 回撤原因分析 ━━━\n");
            printf("回撤期间上证指数: %.2f → %.2f (变动 %+.2f%%)\n",
                   max_dd.idx_at_peak, max_dd.idx_at_trough,
                   (max_dd.idx_at_trough / max_dd.idx_at_peak - 1.0) * 100.0);
            printf("回撤期间持仓: %s\n", max_dd.holding_at_trough.c_str());
            
            // Check what happened during the drawdown: did the held asset drop?
            if (max_dd.holding_at_trough == "中信证券") {
                double citic_peak = snapshots[max_dd.peak_idx].portfolio_value;
                double citic_trough = snapshots[max_dd.trough_idx].portfolio_value;
                printf("中信证券持仓期间资金变动: ¥%.0f → ¥%.0f (%+.2f%%)\n",
                       citic_peak, citic_trough, (citic_trough / citic_peak - 1.0) * 100.0);
                printf("分析: 上证跌破%.0f后买入中信证券，中信跟随大盘继续下跌导致回撤\n", cfg.low_thresh);
            } else if (max_dd.holding_at_trough == "纳指ETF") {
                double nq_peak = snapshots[max_dd.peak_idx].portfolio_value;
                double nq_trough = snapshots[max_dd.trough_idx].portfolio_value;
                printf("纳指ETF持仓期间资金变动: ¥%.0f → ¥%.0f (%+.2f%%)\n",
                       nq_peak, nq_trough, (nq_trough / nq_peak - 1.0) * 100.0);
                printf("分析: 上证涨破%.0f后切换至纳指ETF，纳指随后下跌导致回撤\n", cfg.high_thresh);
            }
            printf("\n");
            
            // Also show all trade actions during the drawdown period
            printf("━━━ 回撤期间的交易操作 ━━━\n");
            bool any_trade = false;
            for (int i = max_dd.peak_idx; i <= (max_dd.recovery_idx > 0 ? max_dd.recovery_idx : max_dd.trough_idx + 20) && i < (int)snapshots.size(); i++) {
                if (snapshots[i].action[0] != '\0') {
                    printf("  %s: %s (上证 %.2f, 资金 ¥%.0f)\n",
                           snapshots[i].date.c_str(), snapshots[i].action,
                           snapshots[i].idx_close, snapshots[i].portfolio_value);
                    any_trade = true;
                }
            }
            if (!any_trade) printf("  回撤期间无交易操作（持仓不变）\n");
            printf("\n\n");
        }
    }
    
    return 0;
}
