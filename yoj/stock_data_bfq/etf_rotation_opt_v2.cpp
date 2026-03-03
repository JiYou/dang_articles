// ETF轮动策略优化器 V2 — 加入资产自身技术指标
//
// V1只看上证指数的指标来决策，忽略了持有资产本身的走势。
// V2新增：
//   1. 资产自身MA止损: 持有纳指/中信时，资产价格跌破自身N日均线 → 卖出(转现金或切换)
//   2. 资产自身RSI: 买入时要求资产RSI不在危险区
//   3. 跟踪止损(trailing stop): 持仓期间从峰值回撤超过X% → 止损
//   4. 上证指数指标（V1已有）
//
// 策略状态: CASH / HOLD_CITIC / HOLD_NASDAQ
//   当止损触发时，资金转入现金，等待下一次上证阈值信号
//
// 分三个Phase:
//   Phase 1: 固定最优阈值(2900/3100)，搜索资产自身指标参数
//   Phase 2: 联合搜索（阈值 + 上证指标 + 资产指标）—— 用Phase1缩小的范围
//   Phase 3: 对比V1最优 vs V2最优 vs 基线
//
// 编译: g++ -O3 -std=c++17 -o etf_rotation_opt_v2 etf_rotation_opt_v2.cpp -lpthread

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
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
    // Drawdown detail
    std::string dd_peak_date, dd_trough_date;
};

struct Config {
    // 上证阈值
    double low_thresh;
    double high_thresh;
    // 上证指标 (V1)
    int idx_rsi_period;
    int idx_rsi_buy;      // 买中信时上证RSI要 < 此值
    int idx_rsi_sell;     // 买纳指时上证RSI要 > 此值
    int idx_ma_period;    // 纳指价格在上证MA趋势确认
    int confirm_days;
    // 资产自身指标 (V2新增)
    int asset_ma_period;    // 资产自身MA周期 (0=不用), 跌破则止损
    int asset_rsi_period;   // 资产自身RSI周期 (0=不用)
    int asset_rsi_ob;       // 超买阈值 (买入时资产RSI < 此值才买)
    int asset_rsi_os;       // 超卖阈值 (资产RSI < 此值时不买，可能还要跌)
    double trailing_stop;   // 跟踪止损百分比 (0=不用), 如15表示从持仓峰值回撤15%止损
    
    std::string to_string() const {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "L=%.0f H=%.0f | IdxRSI(%d,%d,%d) IdxMA(%d) Cfm(%d) | AssetMA(%d) AssetRSI(%d,%d,%d) Trail(%.0f%%)",
                 low_thresh, high_thresh,
                 idx_rsi_period, idx_rsi_buy, idx_rsi_sell,
                 idx_ma_period, confirm_days,
                 asset_ma_period, asset_rsi_period, asset_rsi_ob, asset_rsi_os,
                 trailing_stop);
        return buf;
    }
    
    std::string short_str() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "%.0f/%.0f iRSI(%d,%d,%d) iMA(%d) c%d aMA(%d) aRSI(%d,%d,%d) ts%.0f",
                 low_thresh, high_thresh,
                 idx_rsi_period, idx_rsi_buy, idx_rsi_sell,
                 idx_ma_period, confirm_days,
                 asset_ma_period, asset_rsi_period, asset_rsi_ob, asset_rsi_os,
                 trailing_stop);
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
    bars.clear(); bars.reserve(4096);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string d, s1, s2, s3, s4, s5;
        std::getline(ss, d, ','); std::getline(ss, s1, ','); std::getline(ss, s2, ',');
        std::getline(ss, s3, ','); std::getline(ss, s4, ','); std::getline(ss, s5, ',');
        try { bars.push_back({d, std::stod(s1), std::stod(s2), std::stod(s3), std::stod(s4), std::stod(s5)}); } catch (...) {}
    }
    std::sort(bars.begin(), bars.end(), [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });
    return !bars.empty();
}

bool load_stock_csv(const std::string& path, std::vector<DailyBar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    std::getline(file, line);
    bars.clear(); bars.reserve(4096);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string d, code, so, sc, sh, sl, sv;
        std::getline(ss, d, ','); std::getline(ss, code, ',');
        std::getline(ss, so, ','); std::getline(ss, sc, ','); std::getline(ss, sh, ',');
        std::getline(ss, sl, ','); std::getline(ss, sv, ',');
        try { bars.push_back({d, std::stod(so), std::stod(sh), std::stod(sl), std::stod(sc), std::stod(sv)}); } catch (...) {}
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
    int y = std::stoi(d.substr(0, 4)), m = std::stoi(d.substr(5, 2)), day = std::stoi(d.substr(8, 2));
    if (m <= 2) { y--; m += 12; }
    return 365*y + y/4 - y/100 + y/400 + (153*(m-3)+2)/5 + day;
}

// ---- Technical Indicators ----

std::vector<double> compute_rsi(const std::vector<double>& closes, int period) {
    std::vector<double> rsi(closes.size(), 50.0);
    if (period <= 0 || (int)closes.size() <= period) return rsi;
    double ag = 0, al = 0;
    for (int i = 1; i <= period; i++) {
        double d = closes[i] - closes[i-1];
        if (d > 0) ag += d; else al -= d;
    }
    ag /= period; al /= period;
    rsi[period] = al == 0 ? 100.0 : 100.0 - 100.0/(1.0 + ag/al);
    for (int i = period+1; i < (int)closes.size(); i++) {
        double d = closes[i] - closes[i-1];
        double g = d > 0 ? d : 0, l = d < 0 ? -d : 0;
        ag = (ag*(period-1)+g)/period;
        al = (al*(period-1)+l)/period;
        rsi[i] = al == 0 ? 100.0 : 100.0 - 100.0/(1.0 + ag/al);
    }
    return rsi;
}

std::vector<double> compute_sma(const std::vector<double>& data, int period) {
    std::vector<double> ma(data.size(), 0);
    if (period <= 0) return ma;
    double sum = 0;
    for (int i = 0; i < (int)data.size(); i++) {
        sum += data[i];
        if (i >= period) sum -= data[i-period];
        ma[i] = i >= period-1 ? sum/period : sum/(i+1);
    }
    return ma;
}

// ---- Day Data ----

struct DayData {
    std::string date;
    double idx_close;
    double citic_open, citic_close;
    double nasdaq_open, nasdaq_close;
    // Pre-computed indicators
    double idx_rsi;
    double nasdaq_sma_idx;   // 纳指在上证MA视角 (V1, 沿用)
    double citic_ma;         // 中信自身MA
    double nasdaq_ma;        // 纳指自身MA
    double citic_rsi;        // 中信自身RSI
    double nasdaq_rsi;       // 纳指自身RSI
};

// ---- Backtest Engine V2 ----

Result run_backtest(const std::vector<DayData>& days, const Config& cfg) {
    Result res{};
    
    enum State { CASH, HOLD_CITIC, HOLD_NASDAQ };
    State state = CASH;
    double capital = 1000000.0;
    double shares = 0;
    double buy_price = 0;
    
    double peak_capital = capital;
    double max_dd = 0;
    std::string dd_peak_date, dd_trough_date, peak_date;
    int trades = 0, wins = 0;
    
    std::vector<double> daily_returns;
    double prev_value = capital;
    
    int confirm_buy_citic = 0, confirm_buy_nasdaq = 0;
    int needed = std::max(1, cfg.confirm_days);
    
    // Trailing stop tracking
    double position_peak = 0;  // 持仓期间的最高资产价格
    
    for (size_t i = 0; i < days.size(); i++) {
        const auto& today = days[i];
        
        // Portfolio value at close
        double current_value = capital;
        if (state == HOLD_CITIC) current_value = shares * today.citic_close;
        else if (state == HOLD_NASDAQ) current_value = shares * today.nasdaq_close;
        
        // Update position peak for trailing stop
        if (state == HOLD_CITIC && today.citic_close > position_peak) position_peak = today.citic_close;
        if (state == HOLD_NASDAQ && today.nasdaq_close > position_peak) position_peak = today.nasdaq_close;
        
        // Global drawdown
        if (current_value > peak_capital) { peak_capital = current_value; peak_date = today.date; }
        double dd = (peak_capital - current_value) / peak_capital;
        if (dd > max_dd) { max_dd = dd; dd_peak_date = peak_date; dd_trough_date = today.date; }
        
        // Daily return
        if (prev_value > 0) daily_returns.push_back(current_value / prev_value - 1.0);
        prev_value = current_value;
        
        if (i == 0) continue;
        
        double yesterday_idx = days[i-1].idx_close;
        double yesterday_idx_rsi = days[i-1].idx_rsi;
        
        // ======== V2: Asset-level stop-loss checks (BEFORE entry signals) ========
        bool force_sell = false;
        
        // 1. Asset MA stop-loss: 资产跌破自身MA → 止损
        if (cfg.asset_ma_period > 0) {
            if (state == HOLD_CITIC && days[i-1].citic_close < days[i-1].citic_ma) {
                force_sell = true;
            }
            if (state == HOLD_NASDAQ && days[i-1].nasdaq_close < days[i-1].nasdaq_ma) {
                force_sell = true;
            }
        }
        
        // 2. Trailing stop: 持仓价格从峰值回撤超过X%
        if (cfg.trailing_stop > 0 && state != CASH) {
            double asset_price = (state == HOLD_CITIC) ? days[i-1].citic_close : days[i-1].nasdaq_close;
            if (position_peak > 0) {
                double asset_dd = (position_peak - asset_price) / position_peak * 100.0;
                if (asset_dd >= cfg.trailing_stop) {
                    force_sell = true;
                }
            }
        }
        
        // Execute forced sell at today's open
        if (force_sell) {
            if (state == HOLD_CITIC) {
                double sell_price = today.citic_open;
                capital = shares * sell_price;
                double ret = (sell_price - buy_price) / buy_price;
                trades++; if (ret > 0) wins++;
            } else if (state == HOLD_NASDAQ) {
                double sell_price = today.nasdaq_open;
                capital = shares * sell_price;
                double ret = (sell_price - buy_price) / buy_price;
                trades++; if (ret > 0) wins++;
            }
            state = CASH;
            shares = 0;
            buy_price = 0;
            position_peak = 0;
            confirm_buy_citic = 0;
            confirm_buy_nasdaq = 0;
        }
        
        // ======== Entry/switch signals (same as V1 + asset RSI filter) ========
        bool want_citic = yesterday_idx < cfg.low_thresh;
        bool want_nasdaq = yesterday_idx > cfg.high_thresh;
        
        // Index RSI filter (V1)
        if (cfg.idx_rsi_period > 0 && want_citic) {
            if (yesterday_idx_rsi > cfg.idx_rsi_buy) want_citic = false;
        }
        if (cfg.idx_rsi_period > 0 && want_nasdaq) {
            if (yesterday_idx_rsi < cfg.idx_rsi_sell) want_nasdaq = false;
        }
        
        // Index MA filter on nasdaq (V1)
        if (cfg.idx_ma_period > 0 && want_nasdaq) {
            if (days[i-1].nasdaq_close < days[i-1].nasdaq_sma_idx) want_nasdaq = false;
        }
        
        // V2: Asset RSI filter — don't buy if asset's own RSI is extreme
        if (cfg.asset_rsi_period > 0) {
            // 买中信时：中信RSI不能太高(超买)，也不能太低(超卖，可能继续跌)
            if (want_citic && days[i-1].citic_rsi > cfg.asset_rsi_ob) want_citic = false;
            if (want_citic && days[i-1].citic_rsi < cfg.asset_rsi_os) want_citic = false;
            // 买纳指时同理
            if (want_nasdaq && days[i-1].nasdaq_rsi > cfg.asset_rsi_ob) want_nasdaq = false;
            if (want_nasdaq && days[i-1].nasdaq_rsi < cfg.asset_rsi_os) want_nasdaq = false;
        }
        
        // V2: Asset MA filter — only buy if asset price is above its own MA (趋势向上)
        if (cfg.asset_ma_period > 0) {
            if (want_citic && days[i-1].citic_close < days[i-1].citic_ma) want_citic = false;
            if (want_nasdaq && days[i-1].nasdaq_close < days[i-1].nasdaq_ma) want_nasdaq = false;
        }
        
        // Confirm days
        if (want_citic) { confirm_buy_citic++; confirm_buy_nasdaq = 0; }
        else if (want_nasdaq) { confirm_buy_nasdaq++; confirm_buy_citic = 0; }
        else { confirm_buy_citic = 0; confirm_buy_nasdaq = 0; }
        
        bool do_buy_citic = (confirm_buy_citic >= needed) && (state != HOLD_CITIC);
        bool do_buy_nasdaq = (confirm_buy_nasdaq >= needed) && (state != HOLD_NASDAQ);
        
        if (do_buy_citic) {
            if (state == HOLD_NASDAQ) {
                capital = shares * today.nasdaq_open;
                double ret = (today.nasdaq_open - buy_price) / buy_price;
                trades++; if (ret > 0) wins++;
            }
            buy_price = today.citic_open;
            shares = capital / buy_price;
            state = HOLD_CITIC;
            position_peak = today.citic_close;  // reset position peak
        } else if (do_buy_nasdaq) {
            if (state == HOLD_CITIC) {
                capital = shares * today.citic_open;
                double ret = (today.citic_open - buy_price) / buy_price;
                trades++; if (ret > 0) wins++;
            }
            buy_price = today.nasdaq_open;
            shares = capital / buy_price;
            state = HOLD_NASDAQ;
            position_peak = today.nasdaq_close;  // reset position peak
        }
    }
    
    // Close final position
    if (state == HOLD_CITIC) {
        capital = shares * days.back().citic_close;
        double ret = (days.back().citic_close - buy_price) / buy_price;
        trades++; if (ret > 0) wins++;
    } else if (state == HOLD_NASDAQ) {
        capital = shares * days.back().nasdaq_close;
        double ret = (days.back().nasdaq_close - buy_price) / buy_price;
        trades++; if (ret > 0) wins++;
    }
    
    int total_cal = date_to_days(days.back().date) - date_to_days(days.front().date);
    double years = total_cal / 365.25;
    
    res.cumulative = (capital / 1000000.0 - 1.0) * 100.0;
    res.annualized = compute_annualized(capital / 1000000.0, years);
    res.max_dd = max_dd * 100.0;
    res.calmar = res.max_dd > 0 ? res.annualized / res.max_dd : 0;
    res.trades = trades;
    res.wins = wins;
    res.win_rate = trades > 0 ? (double)wins / trades * 100.0 : 0;
    res.dd_peak_date = dd_peak_date;
    res.dd_trough_date = dd_trough_date;
    
    if (daily_returns.size() > 1) {
        double mean = 0;
        for (auto r : daily_returns) mean += r;
        mean /= daily_returns.size();
        double var = 0;
        for (auto r : daily_returns) var += (r - mean) * (r - mean);
        var /= (daily_returns.size() - 1);
        double sd = std::sqrt(var);
        res.sharpe = sd > 0 ? (mean*252 - 0.02) / (sd*std::sqrt(252.0)) : 0;
    }
    
    return res;
}

void print_table_header() {
    printf("  %-6s %-6s %-4s %-3s %-3s %-3s %-2s %-4s %-4s %-3s %-3s %-5s %9s %9s %7s %7s %5s %6s  %-12s %-12s\n",
           "Lo", "Hi", "iRp", "iRb", "iRs", "iMa", "cf", "aMa", "aRp", "aOb", "aOs", "trail",
           "年化", "回撤", "Calmar", "Sharpe", "交易", "胜率", "回撤峰", "回撤谷");
    printf("  ");
    for (int i = 0; i < 145; i++) printf("-");
    printf("\n");
}

void print_table_row(int rank, const ScoredConfig& sc) {
    printf("  %-6.0f %-6.0f %-4d %-3d %-3d %-3d %-2d %-4d %-4d %-3d %-3d %-5.0f %8.2f%% %8.2f%% %7.2f %7.2f %5d %5.1f%%  %-12s %-12s\n",
           sc.cfg.low_thresh, sc.cfg.high_thresh,
           sc.cfg.idx_rsi_period, sc.cfg.idx_rsi_buy, sc.cfg.idx_rsi_sell,
           sc.cfg.idx_ma_period, sc.cfg.confirm_days,
           sc.cfg.asset_ma_period, sc.cfg.asset_rsi_period,
           sc.cfg.asset_rsi_ob, sc.cfg.asset_rsi_os,
           sc.cfg.trailing_stop,
           sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
           sc.res.trades, sc.res.win_rate,
           sc.res.dd_peak_date.c_str(), sc.res.dd_trough_date.c_str());
}

int main() {
    printf("加载数据...\n");
    
    std::vector<DailyBar> index_bars, citic_bars, nasdaq_bars;
    if (!load_market_csv("../market_data/sh000001.csv", index_bars)) { fprintf(stderr, "Failed: index\n"); return 1; }
    if (!load_stock_csv("600030.csv", citic_bars)) { fprintf(stderr, "Failed: citic\n"); return 1; }
    if (!load_market_csv("../stock_data_qfq/513100.csv", nasdaq_bars)) { fprintf(stderr, "Failed: nasdaq\n"); return 1; }
    
    // Build maps
    std::map<std::string, double> cc_map, co_map, nc_map, no_map;
    for (auto& b : citic_bars) { cc_map[b.date] = b.close; co_map[b.date] = b.open; }
    for (auto& b : nasdaq_bars) { nc_map[b.date] = b.close; no_map[b.date] = b.open; }
    
    struct RawDay { std::string date; double idx_close, co, cc, no_, nc; };
    std::vector<RawDay> raw_days;
    for (auto& bar : index_bars) {
        if (cc_map.count(bar.date) && nc_map.count(bar.date)) {
            raw_days.push_back({bar.date, bar.close, co_map[bar.date], cc_map[bar.date],
                               no_map[bar.date], nc_map[bar.date]});
        }
    }
    printf("  重叠交易日: %zu (%s ~ %s)\n", raw_days.size(), raw_days.front().date.c_str(), raw_days.back().date.c_str());
    
    // Extract close arrays for indicator computation
    std::vector<double> idx_closes, citic_closes, nasdaq_closes;
    for (auto& d : raw_days) {
        idx_closes.push_back(d.idx_close);
        citic_closes.push_back(d.cc);
        nasdaq_closes.push_back(d.nc);
    }
    
    // Pre-compute all indicator variants
    // RSI caches
    std::vector<int> rsi_periods = {0, 6, 10, 14, 20};
    std::map<int, std::vector<double>> idx_rsi_cache, citic_rsi_cache, nasdaq_rsi_cache;
    for (int p : rsi_periods) {
        if (p > 0) {
            idx_rsi_cache[p] = compute_rsi(idx_closes, p);
            citic_rsi_cache[p] = compute_rsi(citic_closes, p);
            nasdaq_rsi_cache[p] = compute_rsi(nasdaq_closes, p);
        }
    }
    
    // MA caches
    std::vector<int> ma_periods = {0, 5, 10, 20, 30, 60};
    std::map<int, std::vector<double>> idx_ma_cache, citic_ma_cache, nasdaq_ma_cache;
    for (int p : ma_periods) {
        if (p > 0) {
            idx_ma_cache[p] = compute_sma(idx_closes, p);
            citic_ma_cache[p] = compute_sma(citic_closes, p);
            nasdaq_ma_cache[p] = compute_sma(nasdaq_closes, p);
        }
    }
    
    int n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    std::mutex mtx;
    
    auto run_configs_parallel = [&](std::vector<Config>& configs, 
                                     std::vector<ScoredConfig>& results) {
        auto worker = [&](int start, int end) {
            for (int ci = start; ci < end; ci++) {
                auto& cfg = configs[ci];
                
                // Build day data with indicators for this config
                std::vector<DayData> days(raw_days.size());
                for (size_t i = 0; i < raw_days.size(); i++) {
                    days[i].date = raw_days[i].date;
                    days[i].idx_close = raw_days[i].idx_close;
                    days[i].citic_open = raw_days[i].co;
                    days[i].citic_close = raw_days[i].cc;
                    days[i].nasdaq_open = raw_days[i].no_;
                    days[i].nasdaq_close = raw_days[i].nc;
                    
                    // Index indicators
                    days[i].idx_rsi = cfg.idx_rsi_period > 0 ? idx_rsi_cache[cfg.idx_rsi_period][i] : 50.0;
                    days[i].nasdaq_sma_idx = cfg.idx_ma_period > 0 ? nasdaq_ma_cache[cfg.idx_ma_period][i] : 0;
                    
                    // Asset indicators
                    days[i].citic_ma = cfg.asset_ma_period > 0 ? citic_ma_cache[cfg.asset_ma_period][i] : 0;
                    days[i].nasdaq_ma = cfg.asset_ma_period > 0 ? nasdaq_ma_cache[cfg.asset_ma_period][i] : 0;
                    days[i].citic_rsi = cfg.asset_rsi_period > 0 ? citic_rsi_cache[cfg.asset_rsi_period][i] : 50.0;
                    days[i].nasdaq_rsi = cfg.asset_rsi_period > 0 ? nasdaq_rsi_cache[cfg.asset_rsi_period][i] : 50.0;
                }
                
                auto res = run_backtest(days, cfg);
                std::lock_guard<std::mutex> lock(mtx);
                results.push_back({cfg, res});
            }
        };
        
        std::vector<std::thread> threads;
        int chunk = ((int)configs.size() + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; t++) {
            int s = t*chunk, e = std::min((int)configs.size(), s+chunk);
            if (s < e) threads.emplace_back(worker, s, e);
        }
        for (auto& t : threads) t.join();
    };
    
    // ====================================================================
    // Phase 1: 固定最优V1阈值和上证指标，只搜索资产自身指标
    // ====================================================================
    printf("\n========== Phase 1: 资产自身指标搜索 ==========\n");
    printf("  固定: 2900/3100, IdxRSI(14,30,70), IdxMA(10), Confirm(1)\n");
    printf("  搜索: AssetMA, AssetRSI, TrailingStop\n\n");
    
    std::vector<Config> p1_configs;
    for (int ama : {0, 5, 10, 20, 30, 60}) {
        for (int arp : {0, 14}) {
            std::vector<int> obs = {80}, oss = {0};
            if (arp > 0) {
                obs = {65, 70, 75, 80, 85};
                oss = {0, 15, 20, 25, 30};
            }
            for (int aob : obs) {
                for (int aos : oss) {
                    for (double ts : {0.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0}) {
                        // Skip no-asset-indicator config (that's V1 baseline)
                        if (ama == 0 && arp == 0 && ts == 0.0) continue;
                        p1_configs.push_back({2900, 3100, 14, 30, 70, 10, 1,
                                             ama, arp, aob, aos, ts});
                    }
                }
            }
        }
    }
    
    // Also add V1 baseline for comparison
    p1_configs.push_back({2900, 3100, 14, 30, 70, 10, 1, 0, 0, 80, 0, 0.0});
    // And pure baseline (no indicators at all)
    p1_configs.push_back({2900, 3200, 0, 0, 0, 0, 1, 0, 0, 80, 0, 0.0});
    
    printf("  搜索组合数: %zu, 线程数: %d\n", p1_configs.size(), n_threads);
    
    std::vector<ScoredConfig> p1_results;
    run_configs_parallel(p1_configs, p1_results);
    
    // Sort by Calmar
    std::sort(p1_results.begin(), p1_results.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.calmar > b.res.calmar; });
    
    printf("\n  Top 20 by Calmar:\n");
    print_table_header();
    for (int i = 0; i < std::min(20, (int)p1_results.size()); i++) print_table_row(i+1, p1_results[i]);
    
    // Find V1 baseline in results
    printf("\n  参考 - V1最优 (无资产指标):\n");
    for (auto& sc : p1_results) {
        if (sc.cfg.low_thresh == 2900 && sc.cfg.high_thresh == 3100 &&
            sc.cfg.asset_ma_period == 0 && sc.cfg.asset_rsi_period == 0 && sc.cfg.trailing_stop == 0) {
            printf("  ");
            print_table_row(0, sc);
            break;
        }
    }
    printf("\n  参考 - 基线 (2900/3200 无指标):\n");
    for (auto& sc : p1_results) {
        if (sc.cfg.low_thresh == 2900 && sc.cfg.high_thresh == 3200 &&
            sc.cfg.idx_rsi_period == 0 && sc.cfg.asset_ma_period == 0) {
            printf("  ");
            print_table_row(0, sc);
            break;
        }
    }
    
    // Also sort by annualized
    auto p1_by_ann = p1_results;
    std::sort(p1_by_ann.begin(), p1_by_ann.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.annualized > b.res.annualized; });
    printf("\n  Top 20 by 年化收益:\n");
    print_table_header();
    for (int i = 0; i < std::min(20, (int)p1_by_ann.size()); i++) print_table_row(i+1, p1_by_ann[i]);
    
    // Sort by max_dd (ascending = lower drawdown better)
    auto p1_by_dd = p1_results;
    std::sort(p1_by_dd.begin(), p1_by_dd.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) {
                  // Among configs with positive returns, prefer lower drawdown
                  if (a.res.annualized > 10 && b.res.annualized > 10) return a.res.max_dd < b.res.max_dd;
                  return a.res.annualized > b.res.annualized;
              });
    printf("\n  Top 20 by 最低回撤 (年化>10%%):\n");
    print_table_header();
    for (int i = 0; i < std::min(20, (int)p1_by_dd.size()); i++) print_table_row(i+1, p1_by_dd[i]);
    
    // ====================================================================
    // Phase 2: 联合搜索 — 用Phase 1发现的最佳资产指标范围 + 多组阈值
    // ====================================================================
    printf("\n\n========== Phase 2: 联合优化 (阈值+上证指标+资产指标) ==========\n");
    
    // Pick best asset indicator patterns from Phase 1
    // Extract unique (asset_ma, asset_rsi, trailing_stop) combos from top results
    struct AssetParams { int ama, arp, aob, aos; double ts; };
    std::vector<AssetParams> top_asset_params;
    
    // Collect top 10 unique asset param combos by Calmar
    for (auto& sc : p1_results) {
        AssetParams ap{sc.cfg.asset_ma_period, sc.cfg.asset_rsi_period,
                      sc.cfg.asset_rsi_ob, sc.cfg.asset_rsi_os, sc.cfg.trailing_stop};
        bool dup = false;
        for (auto& existing : top_asset_params) {
            if (existing.ama == ap.ama && existing.arp == ap.arp && existing.aob == ap.aob &&
                existing.aos == ap.aos && existing.ts == ap.ts) { dup = true; break; }
        }
        if (!dup) {
            top_asset_params.push_back(ap);
            if ((int)top_asset_params.size() >= 15) break;
        }
    }
    // Also include no-asset-indicator baseline
    top_asset_params.push_back({0, 0, 80, 0, 0.0});
    
    std::vector<Config> p2_configs;
    // Threshold ranges
    for (double lo : {2800, 2850, 2900, 2950, 3000}) {
        for (double hi = lo + 100; hi <= 3400; hi += 50) {
            // Index indicator combos (narrowed from V1 findings)
            struct IdxParams { int rp, rb, rs, ma, cf; };
            std::vector<IdxParams> idx_combos = {
                {0, 0, 0, 0, 1},       // no index indicators
                {14, 30, 70, 0, 1},    // RSI only
                {14, 30, 70, 10, 1},   // RSI + MA (V1 best)
                {14, 35, 70, 10, 1},
                {14, 30, 65, 10, 1},
                {14, 30, 70, 20, 1},
                {14, 30, 70, 5, 1},
            };
            
            for (auto& ip : idx_combos) {
                for (auto& ap : top_asset_params) {
                    p2_configs.push_back({lo, hi, ip.rp, ip.rb, ip.rs, ip.ma, ip.cf,
                                         ap.ama, ap.arp, ap.aob, ap.aos, ap.ts});
                }
            }
        }
    }
    
    printf("  搜索组合数: %zu, 线程数: %d\n", p2_configs.size(), n_threads);
    
    std::vector<ScoredConfig> p2_results;
    run_configs_parallel(p2_configs, p2_results);
    
    // Sort by Calmar
    std::sort(p2_results.begin(), p2_results.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.calmar > b.res.calmar; });
    
    printf("\n  Top 25 by Calmar:\n");
    print_table_header();
    for (int i = 0; i < std::min(25, (int)p2_results.size()); i++) print_table_row(i+1, p2_results[i]);
    
    auto p2_by_ann = p2_results;
    std::sort(p2_by_ann.begin(), p2_by_ann.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.annualized > b.res.annualized; });
    printf("\n  Top 25 by 年化收益:\n");
    print_table_header();
    for (int i = 0; i < std::min(25, (int)p2_by_ann.size()); i++) print_table_row(i+1, p2_by_ann[i]);
    
    auto p2_by_dd = p2_results;
    std::sort(p2_by_dd.begin(), p2_by_dd.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) {
                  if (a.res.annualized > 15 && b.res.annualized > 15) return a.res.max_dd < b.res.max_dd;
                  return a.res.annualized > b.res.annualized;
              });
    printf("\n  Top 25 by 最低回撤 (年化>15%%):\n");
    print_table_header();
    for (int i = 0; i < std::min(25, (int)p2_by_dd.size()); i++) print_table_row(i+1, p2_by_dd[i]);
    
    auto p2_by_sharpe = p2_results;
    std::sort(p2_by_sharpe.begin(), p2_by_sharpe.end(),
              [](const ScoredConfig& a, const ScoredConfig& b) { return a.res.sharpe > b.res.sharpe; });
    printf("\n  Top 25 by Sharpe:\n");
    print_table_header();
    for (int i = 0; i < std::min(25, (int)p2_by_sharpe.size()); i++) print_table_row(i+1, p2_by_sharpe[i]);
    
    // ====================================================================
    // Final comparison
    // ====================================================================
    printf("\n\n╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  最终对比: 基线 vs V1最优 vs V2最优                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");
    
    // Find specific configs in results
    auto find_config = [&](const std::vector<ScoredConfig>& results,
                           double lo, double hi, int irp, int irb, int irs, int ima, int cf,
                           int ama, int arp, double ts) -> const ScoredConfig* {
        for (auto& sc : results) {
            if (sc.cfg.low_thresh == lo && sc.cfg.high_thresh == hi &&
                sc.cfg.idx_rsi_period == irp && sc.cfg.idx_rsi_buy == irb &&
                sc.cfg.idx_rsi_sell == irs && sc.cfg.idx_ma_period == ima &&
                sc.cfg.asset_ma_period == ama && sc.cfg.asset_rsi_period == arp &&
                sc.cfg.trailing_stop == ts) return &sc;
        }
        return nullptr;
    };
    
    // Print comparison
    auto print_comparison = [](const char* label, const ScoredConfig& sc) {
        printf("%-45s  年化=%6.2f%%  回撤=%6.2f%%  Calmar=%5.2f  Sharpe=%5.2f  交易=%2d  胜率=%5.1f%%\n",
               label, sc.res.annualized, sc.res.max_dd, sc.res.calmar, sc.res.sharpe,
               sc.res.trades, sc.res.win_rate);
        printf("%-45s  回撤区间: %s ~ %s\n", "", sc.res.dd_peak_date.c_str(), sc.res.dd_trough_date.c_str());
        printf("%-45s  %s\n\n", "", sc.cfg.to_string().c_str());
    };
    
    // Run baselines directly if not in p2
    // Baseline: 2900/3200 no indicators
    {
        Config baseline_cfg{2900, 3200, 0, 0, 0, 0, 1, 0, 0, 80, 0, 0.0};
        std::vector<DayData> days(raw_days.size());
        for (size_t i = 0; i < raw_days.size(); i++) {
            days[i] = {raw_days[i].date, raw_days[i].idx_close,
                       raw_days[i].co, raw_days[i].cc, raw_days[i].no_, raw_days[i].nc,
                       50.0, 0, 0, 0, 50.0, 50.0};
        }
        auto res = run_backtest(days, baseline_cfg);
        ScoredConfig baseline{baseline_cfg, res};
        print_comparison("基线 (2900/3200 无指标)", baseline);
    }
    
    // V1 best: 2900/3100 IdxRSI(14,30,70) IdxMA(10) Confirm(1)
    {
        Config v1_cfg{2900, 3100, 14, 30, 70, 10, 1, 0, 0, 80, 0, 0.0};
        std::vector<DayData> days(raw_days.size());
        for (size_t i = 0; i < raw_days.size(); i++) {
            days[i] = {raw_days[i].date, raw_days[i].idx_close,
                       raw_days[i].co, raw_days[i].cc, raw_days[i].no_, raw_days[i].nc,
                       idx_rsi_cache[14][i], nasdaq_ma_cache[10][i],
                       0, 0, 50.0, 50.0};
        }
        auto res = run_backtest(days, v1_cfg);
        ScoredConfig v1{v1_cfg, res};
        print_comparison("V1最优 (2900/3100 +上证指标)", v1);
    }
    
    // V2 best by Calmar
    printf("V2最优 Calmar:\n");
    print_comparison("", p2_results.front());
    
    printf("V2最优 年化:\n");
    print_comparison("", p2_by_ann.front());
    
    printf("V2最优 最低回撤 (年化>15%%):\n");
    print_comparison("", p2_by_dd.front());
    
    printf("V2最优 Sharpe:\n");
    print_comparison("", p2_by_sharpe.front());
    
    return 0;
}
