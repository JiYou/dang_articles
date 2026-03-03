// ETF轮动策略优化器
// 
// 优化维度：
// 1. 阈值网格搜索: low=[2500,3100 step50], high=[low+100, 3800 step50]
// 2. 技术指标辅助调仓:
//    - RSI超卖/超买过滤 (买中信时要求RSI<阈值, 买纳指时要求RSI>阈值)
//    - 均线趋势过滤 (纳指在MA上方才买入)
//    - ATR波动率过滤 (高波动时不切换)
//    - 延迟确认 (连续N天满足条件才切换)
// 3. 评分: 年化收益, 最大回撤, Calmar, Sharpe
//
// 数据：前复权 (拆股修正后)
// 编译: g++ -O3 -std=c++17 -o etf_rotation_opt etf_rotation_opt.cpp -lpthread

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct DailyBar {
    std::string date;
    double open, high, low, close, volume;
};

struct Result {
    double annualized;
    double max_dd;
    double calmar;
    double sharpe;
    double cumulative;
    int trades;
    int wins;
    double win_rate;
};

struct Config {
    double low_thresh;
    double high_thresh;
    // 技术指标参数
    int rsi_period;        // 0=不用RSI
    int rsi_buy_thresh;    // 中信买入时上证RSI要 < 此值 (超卖)
    int rsi_sell_thresh;   // 纳指买入时上证RSI要 > 此值 (超买)
    int ma_period;         // 0=不用均线, >0: 纳指价格在MA之上才买入
    int confirm_days;      // 连续N天满足条件才切换, 0或1=立即
    
    std::string to_string() const {
        char buf[256];
        snprintf(buf, sizeof(buf), "L=%.0f H=%.0f RSI(%d,%d,%d) MA(%d) Confirm(%d)",
                 low_thresh, high_thresh, rsi_period, rsi_buy_thresh, rsi_sell_thresh,
                 ma_period, confirm_days);
        return buf;
    }
};

struct ScoredConfig {
    Config cfg;
    Result res;
};

// ---- Data Loading ----

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

// ---- Technical Indicators ----

// Compute RSI series for index closes
std::vector<double> compute_rsi(const std::vector<double>& closes, int period) {
    std::vector<double> rsi(closes.size(), 50.0); // default neutral
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

// Compute simple moving average
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

// ---- Unified Day Data ----

struct DayData {
    std::string date;
    double idx_close;
    double citic_open, citic_close;
    double nasdaq_open, nasdaq_close;
    double idx_rsi;       // pre-computed
    double nasdaq_ma;     // pre-computed (various periods cached externally)
};

// ---- Backtest Engine ----

Result run_backtest(const std::vector<DayData>& days, const Config& cfg) {
    Result res{};
    
    enum State { CASH, HOLD_CITIC, HOLD_NASDAQ };
    State state = CASH;
    double capital = 1000000.0;
    double buy_price = 0;
    
    double peak_capital = capital;
    double max_dd = 0;
    int trades = 0, wins = 0;
    
    // For Sharpe: track daily returns
    std::vector<double> daily_returns;
    double prev_value = capital;
    
    // Confirm counter
    int confirm_buy_citic = 0;
    int confirm_buy_nasdaq = 0;
    int needed = std::max(1, cfg.confirm_days);
    
    double shares = 0;
    
    for (size_t i = 0; i < days.size(); i++) {
        const auto& today = days[i];
        
        // Track portfolio value at close
        double current_value = capital;
        if (state == HOLD_CITIC) current_value = shares * today.citic_close;
        else if (state == HOLD_NASDAQ) current_value = shares * today.nasdaq_close;
        
        // Drawdown
        if (current_value > peak_capital) peak_capital = current_value;
        double dd = (peak_capital - current_value) / peak_capital;
        if (dd > max_dd) max_dd = dd;
        
        // Daily return
        if (prev_value > 0) daily_returns.push_back(current_value / prev_value - 1.0);
        prev_value = current_value;
        
        if (i == 0) continue;
        
        double yesterday_idx = days[i-1].idx_close;
        double yesterday_rsi = days[i-1].idx_rsi;
        
        // Check conditions
        bool want_citic = yesterday_idx < cfg.low_thresh;
        bool want_nasdaq = yesterday_idx > cfg.high_thresh;
        
        // RSI filter
        if (cfg.rsi_period > 0 && want_citic) {
            if (yesterday_rsi > cfg.rsi_buy_thresh) want_citic = false; // RSI not oversold enough
        }
        if (cfg.rsi_period > 0 && want_nasdaq) {
            if (yesterday_rsi < cfg.rsi_sell_thresh) want_nasdaq = false; // RSI not overbought enough
        }
        
        // MA filter: only buy nasdaq if nasdaq price > MA
        if (cfg.ma_period > 0 && want_nasdaq) {
            if (days[i-1].nasdaq_close < days[i-1].nasdaq_ma) want_nasdaq = false;
        }
        
        // Confirm days
        if (want_citic) { confirm_buy_citic++; confirm_buy_nasdaq = 0; }
        else if (want_nasdaq) { confirm_buy_nasdaq++; confirm_buy_citic = 0; }
        else { confirm_buy_citic = 0; confirm_buy_nasdaq = 0; }
        
        bool do_buy_citic = (confirm_buy_citic >= needed) && (state != HOLD_CITIC);
        bool do_buy_nasdaq = (confirm_buy_nasdaq >= needed) && (state != HOLD_NASDAQ);
        
        if (do_buy_citic) {
            if (state == HOLD_NASDAQ) {
                double sell_price = today.nasdaq_open;
                capital = shares * sell_price;
                double ret = (sell_price - buy_price) / buy_price;
                trades++;
                if (ret > 0) wins++;
            }
            buy_price = today.citic_open;
            shares = capital / buy_price;
            state = HOLD_CITIC;
        } else if (do_buy_nasdaq) {
            if (state == HOLD_CITIC) {
                double sell_price = today.citic_open;
                capital = shares * sell_price;
                double ret = (sell_price - buy_price) / buy_price;
                trades++;
                if (ret > 0) wins++;
            }
            buy_price = today.nasdaq_open;
            shares = capital / buy_price;
            state = HOLD_NASDAQ;
        }
    }
    
    // Close final position
    if (state == HOLD_CITIC) {
        capital = shares * days.back().citic_close;
        double ret = (days.back().citic_close - buy_price) / buy_price;
        trades++;
        if (ret > 0) wins++;
    } else if (state == HOLD_NASDAQ) {
        capital = shares * days.back().nasdaq_close;
        double ret = (days.back().nasdaq_close - buy_price) / buy_price;
        trades++;
        if (ret > 0) wins++;
    }
    
    int total_days_cal = date_to_days(days.back().date) - date_to_days(days.front().date);
    double years = total_days_cal / 365.25;
    
    res.cumulative = (capital / 1000000.0 - 1.0) * 100.0;
    res.annualized = compute_annualized(capital / 1000000.0, years);
    res.max_dd = max_dd * 100.0;
    res.calmar = res.max_dd > 0 ? res.annualized / res.max_dd : 0;
    res.trades = trades;
    res.wins = wins;
    res.win_rate = trades > 0 ? (double)wins / trades * 100.0 : 0;
    
    // Sharpe
    if (daily_returns.size() > 1) {
        double mean = 0;
        for (auto r : daily_returns) mean += r;
        mean /= daily_returns.size();
        double var = 0;
        for (auto r : daily_returns) var += (r - mean) * (r - mean);
        var /= (daily_returns.size() - 1);
        double std_dev = std::sqrt(var);
        double annual_ret = mean * 252;
        double annual_vol = std_dev * std::sqrt(252.0);
        res.sharpe = annual_vol > 0 ? (annual_ret - 0.02) / annual_vol : 0; // rf=2%
    }
    
    return res;
}

int main() {
    printf("加载数据...\n");
    
    std::vector<DailyBar> index_bars, citic_bars, nasdaq_bars;
    
    if (!load_market_csv("../market_data/sh000001.csv", index_bars)) { std::cerr << "Failed: index\n"; return 1; }
    if (!load_stock_csv("600030.csv", citic_bars)) { std::cerr << "Failed: citic\n"; return 1; }
    if (!load_market_csv("../stock_data_qfq/513100.csv", nasdaq_bars)) { std::cerr << "Failed: nasdaq\n"; return 1; }
    
    printf("  上证: %zu, 中信: %zu, 纳指513100(前复权): %zu\n",
           index_bars.size(), citic_bars.size(), nasdaq_bars.size());
    
    // Build maps
    std::map<std::string, double> citic_close_map, citic_open_map, nasdaq_close_map, nasdaq_open_map;
    for (auto& b : citic_bars) { citic_close_map[b.date] = b.close; citic_open_map[b.date] = b.open; }
    for (auto& b : nasdaq_bars) { nasdaq_close_map[b.date] = b.close; nasdaq_open_map[b.date] = b.open; }
    
    // Build unified day list (only dates where all 3 have data)
    struct RawDay {
        std::string date;
        double idx_close;
        double citic_open, citic_close;
        double nasdaq_open, nasdaq_close;
    };
    std::vector<RawDay> raw_days;
    for (auto& bar : index_bars) {
        if (citic_close_map.count(bar.date) && nasdaq_close_map.count(bar.date)) {
            raw_days.push_back({bar.date, bar.close,
                               citic_open_map[bar.date], citic_close_map[bar.date],
                               nasdaq_open_map[bar.date], nasdaq_close_map[bar.date]});
        }
    }
    printf("  重叠交易日: %zu (%s ~ %s)\n", raw_days.size(),
           raw_days.front().date.c_str(), raw_days.back().date.c_str());
    
    // Pre-compute index closes and nasdaq closes for indicator calculation
    std::vector<double> idx_closes, nasdaq_closes;
    for (auto& d : raw_days) { idx_closes.push_back(d.idx_close); nasdaq_closes.push_back(d.nasdaq_close); }
    
    // Pre-compute RSI for various periods
    std::map<int, std::vector<double>> rsi_cache;
    for (int p : {0, 6, 10, 14, 20}) {
        if (p > 0) rsi_cache[p] = compute_rsi(idx_closes, p);
    }
    
    // Pre-compute MA for various periods on nasdaq
    std::map<int, std::vector<double>> ma_cache;
    for (int p : {0, 5, 10, 20, 30, 60}) {
        if (p > 0) ma_cache[p] = compute_sma(nasdaq_closes, p);
    }
    
    // ====== Phase 1: Pure threshold grid search (no indicators) ======
    printf("\n========== Phase 1: 阈值网格搜索 (无指标) ==========\n");
    
    std::vector<ScoredConfig> all_results;
    std::mutex mtx;
    
    // Build base DayData (no RSI/MA for phase 1)
    std::vector<DayData> base_days(raw_days.size());
    for (size_t i = 0; i < raw_days.size(); i++) {
        base_days[i] = {raw_days[i].date, raw_days[i].idx_close,
                        raw_days[i].citic_open, raw_days[i].citic_close,
                        raw_days[i].nasdaq_open, raw_days[i].nasdaq_close,
                        50.0, 0};
    }
    
    // Grid: low in [2500, 3100] step 50, high in [low+200, 3800] step 50
    std::vector<Config> configs;
    for (double lo = 2500; lo <= 3100; lo += 50) {
        for (double hi = lo + 200; hi <= 3800; hi += 50) {
            configs.push_back({lo, hi, 0, 0, 0, 0, 0});
        }
    }
    printf("  搜索组合数: %zu\n", configs.size());
    
    // Multi-threaded execution
    int n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    printf("  线程数: %d\n", n_threads);
    
    auto run_chunk = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            auto res = run_backtest(base_days, configs[i]);
            std::lock_guard<std::mutex> lock(mtx);
            all_results.push_back({configs[i], res});
        }
    };
    
    std::vector<std::thread> threads;
    int chunk = (configs.size() + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        int s = t * chunk, e = std::min((int)configs.size(), s + chunk);
        if (s < e) threads.emplace_back(run_chunk, s, e);
    }
    for (auto& t : threads) t.join();
    
    // Sort by Calmar
    std::sort(all_results.begin(), all_results.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.calmar > b.res.calmar; });
    
    printf("\n  Top 10 by Calmar (纯阈值):\n");
    printf("  %-6s %-6s %10s %10s %8s %8s %6s %6s\n",
           "Low", "High", "年化", "最大回撤", "Calmar", "Sharpe", "交易", "胜率");
    printf("  -------------------------------------------------------------------------\n");
    for (int i = 0; i < std::min(10, (int)all_results.size()); i++) {
        auto& sc = all_results[i];
        printf("  %-6.0f %-6.0f %9.2f%% %9.2f%% %8.2f %8.2f %6d %5.1f%%\n",
               sc.cfg.low_thresh, sc.cfg.high_thresh,
               sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
               sc.res.trades, sc.res.win_rate);
    }
    
    // Also top 10 by annualized
    auto by_annual = all_results;
    std::sort(by_annual.begin(), by_annual.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.annualized > b.res.annualized; });
    
    printf("\n  Top 10 by 年化收益 (纯阈值):\n");
    printf("  %-6s %-6s %10s %10s %8s %8s %6s %6s\n",
           "Low", "High", "年化", "最大回撤", "Calmar", "Sharpe", "交易", "胜率");
    printf("  -------------------------------------------------------------------------\n");
    for (int i = 0; i < std::min(10, (int)by_annual.size()); i++) {
        auto& sc = by_annual[i];
        printf("  %-6.0f %-6.0f %9.2f%% %9.2f%% %8.2f %8.2f %6d %5.1f%%\n",
               sc.cfg.low_thresh, sc.cfg.high_thresh,
               sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
               sc.res.trades, sc.res.win_rate);
    }
    
    // Find baseline (3200/2900)
    for (auto& sc : all_results) {
        if (sc.cfg.low_thresh == 2900 && sc.cfg.high_thresh == 3200) {
            printf("\n  基线 (2900/3200): 年化=%.2f%% 回撤=%.2f%% Calmar=%.2f Sharpe=%.2f 交易=%d 胜率=%.1f%%\n",
                   sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
                   sc.res.trades, sc.res.win_rate);
            break;
        }
    }
    
    // ====== Phase 2: Add technical indicators to top thresholds ======
    printf("\n\n========== Phase 2: 技术指标优化 ==========\n");
    
    // Take top 5 threshold pairs by Calmar, and test indicators on them
    std::vector<std::pair<double,double>> top_thresholds;
    for (int i = 0; i < std::min(5, (int)all_results.size()); i++) {
        top_thresholds.push_back({all_results[i].cfg.low_thresh, all_results[i].cfg.high_thresh});
    }
    // Also include baseline
    top_thresholds.push_back({2900, 3200});
    
    std::vector<ScoredConfig> indicator_results;
    
    for (auto& [lo, hi] : top_thresholds) {
        // Test RSI variants
        for (int rsi_p : {0, 6, 10, 14, 20}) {
            std::vector<int> buy_thresholds = {30};
            std::vector<int> sell_thresholds = {50};
            if (rsi_p == 0) { buy_thresholds = {0}; sell_thresholds = {0}; }
            else {
                buy_thresholds = {25, 30, 35, 40, 45};
                sell_thresholds = {40, 50, 55, 60, 65, 70};
            }
            
            for (int rsi_buy : buy_thresholds) {
                for (int rsi_sell : sell_thresholds) {
                    // Test MA variants
                    for (int ma_p : {0, 5, 10, 20, 30, 60}) {
                        // Test confirm days
                        for (int confirm : {1, 2, 3, 5}) {
                            Config cfg{lo, hi, rsi_p, rsi_buy, rsi_sell, ma_p, confirm};
                            
                            // Build day data with indicators
                            std::vector<DayData> days(raw_days.size());
                            for (size_t i = 0; i < raw_days.size(); i++) {
                                days[i] = {raw_days[i].date, raw_days[i].idx_close,
                                           raw_days[i].citic_open, raw_days[i].citic_close,
                                           raw_days[i].nasdaq_open, raw_days[i].nasdaq_close,
                                           rsi_p > 0 ? rsi_cache[rsi_p][i] : 50.0,
                                           ma_p > 0 ? ma_cache[ma_p][i] : 0};
                            }
                            
                            auto res = run_backtest(days, cfg);
                            indicator_results.push_back({cfg, res});
                        }
                    }
                }
            }
        }
    }
    
    printf("  指标组合搜索数: %zu\n", indicator_results.size());
    
    // Sort by Calmar
    std::sort(indicator_results.begin(), indicator_results.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.calmar > b.res.calmar; });
    
    printf("\n  Top 15 by Calmar (含指标):\n");
    printf("  %-6s %-6s %-5s %-4s %-4s %-4s %-4s %10s %10s %8s %8s %6s %6s\n",
           "Low", "High", "RSI_P", "R_B", "R_S", "MA", "Cfm", "年化", "最大回撤", "Calmar", "Sharpe", "交易", "胜率");
    printf("  ---------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < std::min(15, (int)indicator_results.size()); i++) {
        auto& sc = indicator_results[i];
        printf("  %-6.0f %-6.0f %-5d %-4d %-4d %-4d %-4d %9.2f%% %9.2f%% %8.2f %8.2f %6d %5.1f%%\n",
               sc.cfg.low_thresh, sc.cfg.high_thresh,
               sc.cfg.rsi_period, sc.cfg.rsi_buy_thresh, sc.cfg.rsi_sell_thresh,
               sc.cfg.ma_period, sc.cfg.confirm_days,
               sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
               sc.res.trades, sc.res.win_rate);
    }
    
    // Top by annualized
    auto ind_by_annual = indicator_results;
    std::sort(ind_by_annual.begin(), ind_by_annual.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.annualized > b.res.annualized; });
    
    printf("\n  Top 15 by 年化收益 (含指标):\n");
    printf("  %-6s %-6s %-5s %-4s %-4s %-4s %-4s %10s %10s %8s %8s %6s %6s\n",
           "Low", "High", "RSI_P", "R_B", "R_S", "MA", "Cfm", "年化", "最大回撤", "Calmar", "Sharpe", "交易", "胜率");
    printf("  ---------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < std::min(15, (int)ind_by_annual.size()); i++) {
        auto& sc = ind_by_annual[i];
        printf("  %-6.0f %-6.0f %-5d %-4d %-4d %-4d %-4d %9.2f%% %9.2f%% %8.2f %8.2f %6d %5.1f%%\n",
               sc.cfg.low_thresh, sc.cfg.high_thresh,
               sc.cfg.rsi_period, sc.cfg.rsi_buy_thresh, sc.cfg.rsi_sell_thresh,
               sc.cfg.ma_period, sc.cfg.confirm_days,
               sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
               sc.res.trades, sc.res.win_rate);
    }
    
    // Top by Sharpe
    auto ind_by_sharpe = indicator_results;
    std::sort(ind_by_sharpe.begin(), ind_by_sharpe.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.sharpe > b.res.sharpe; });
    
    printf("\n  Top 15 by Sharpe (含指标):\n");
    printf("  %-6s %-6s %-5s %-4s %-4s %-4s %-4s %10s %10s %8s %8s %6s %6s\n",
           "Low", "High", "RSI_P", "R_B", "R_S", "MA", "Cfm", "年化", "最大回撤", "Calmar", "Sharpe", "交易", "胜率");
    printf("  ---------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < std::min(15, (int)ind_by_sharpe.size()); i++) {
        auto& sc = ind_by_sharpe[i];
        printf("  %-6.0f %-6.0f %-5d %-4d %-4d %-4d %-4d %9.2f%% %9.2f%% %8.2f %8.2f %6d %5.1f%%\n",
               sc.cfg.low_thresh, sc.cfg.high_thresh,
               sc.cfg.rsi_period, sc.cfg.rsi_buy_thresh, sc.cfg.rsi_sell_thresh,
               sc.cfg.ma_period, sc.cfg.confirm_days,
               sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
               sc.res.trades, sc.res.win_rate);
    }
    
    // ====== Final comparison ======
    printf("\n\n========================================\n");
    printf("  最终对比: 基线 vs 最优\n");
    printf("========================================\n");
    
    // Find baseline in indicator_results
    for (auto& sc : indicator_results) {
        if (sc.cfg.low_thresh == 2900 && sc.cfg.high_thresh == 3200 &&
            sc.cfg.rsi_period == 0 && sc.cfg.ma_period == 0 && sc.cfg.confirm_days == 1) {
            printf("基线 (2900/3200 无指标):  年化=%.2f%% 回撤=%.2f%% Calmar=%.2f Sharpe=%.2f\n",
                   sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe);
            break;
        }
    }
    
    auto& best_calmar = indicator_results.front();
    printf("最优Calmar:  %s\n", best_calmar.cfg.to_string().c_str());
    printf("  年化=%.2f%% 回撤=%.2f%% Calmar=%.2f Sharpe=%.2f 交易=%d 胜率=%.1f%%\n",
           best_calmar.res.annualized, best_calmar.res.max_dd, best_calmar.res.calmar,
           best_calmar.res.sharpe, best_calmar.res.trades, best_calmar.res.win_rate);
    
    auto& best_annual = ind_by_annual.front();
    printf("最优年化:    %s\n", best_annual.cfg.to_string().c_str());
    printf("  年化=%.2f%% 回撤=%.2f%% Calmar=%.2f Sharpe=%.2f 交易=%d 胜率=%.1f%%\n",
           best_annual.res.annualized, best_annual.res.max_dd, best_annual.res.calmar,
           best_annual.res.sharpe, best_annual.res.trades, best_annual.res.win_rate);
    
    auto& best_sharpe_cfg = ind_by_sharpe.front();
    printf("最优Sharpe:  %s\n", best_sharpe_cfg.cfg.to_string().c_str());
    printf("  年化=%.2f%% 回撤=%.2f%% Calmar=%.2f Sharpe=%.2f 交易=%d 胜率=%.1f%%\n",
           best_sharpe_cfg.res.annualized, best_sharpe_cfg.res.max_dd, best_sharpe_cfg.res.calmar,
           best_sharpe_cfg.res.sharpe, best_sharpe_cfg.res.trades, best_sharpe_cfg.res.win_rate);
    
    printf("\n");
    return 0;
}
