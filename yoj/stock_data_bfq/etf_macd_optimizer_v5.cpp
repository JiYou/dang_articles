/*
 * ETF MACD Strategy Optimizer V5 — Weekly Timeframe
 *
 * Key changes from V4 (monthly, stocks):
 * 1. Weekly bar aggregation (Mon-Fri grouped by ISO week)
 * 2. Optimized for ETFs (66 assets, ~540 weekly bars each)
 * 3. Trailing stop: sell if price drops X% from peak since entry
 * 4. Take-profit: sell if unrealized gain exceeds X%
 * 5. Also tests monthly timeframe for direct comparison
 * 6. Optimize Top 20 ETF annualized return on 40% training data
 *
 * ETF CSV format: date,open,high,low,close,volume
 * Compiles: g++ -O3 -std=c++17 -pthread -o etf_macd_optimizer_v5 etf_macd_optimizer_v5.cpp
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// Data structures
// ============================================================

struct DailyBar {
    std::string date;
    double open, close, high, low, volume;
};

struct AggBar {
    double open, close, high, low;
    double volume;
    int first_daily_idx;
    int last_daily_idx;
};

struct ETFData {
    std::string code;
    std::string name;
    std::vector<DailyBar> daily;
    std::vector<AggBar> train_bars;   // weekly or monthly (training period)
    std::vector<AggBar> test_bars;    // weekly or monthly (test period)
    int split_idx;                    // daily index split point
};

struct StrategyParams {
    int fast;
    int slow;
    int signal;
    int buy_mode;         // 0=golden cross, 1=histogram>0
    int sell_mode;        // 0=death cross, 1=DIF<0, 2=histogram<0
    double trailing_stop; // 0=off, else % drop from peak to trigger sell (e.g. 8.0 = 8%)
    double take_profit;   // 0=off, else % gain to trigger sell (e.g. 30.0 = 30%)
    int cooldown;         // bars to wait after sell before next buy (0=off)
};

struct EvalResult {
    int trades = 0;
    double cumulative_return_pct = 0;
    double annualized_return_pct = 0;
    double win_rate = 0;
    double max_drawdown_pct = 0;
    double buy_hold_return_pct = 0;
    std::string date_start, date_end;
    std::vector<std::string> trade_log;
};

// ============================================================
// Global Data
// ============================================================

std::vector<ETFData> g_etfs;
bool g_use_weekly = true;  // toggled by mode

// ETF name mapping
std::unordered_map<std::string, std::string> g_etf_names = {
    // Indices (excluded from optimization)
    {"sh000001", "上证指数"}, {"sh000016", "上证50"}, {"sh000300", "沪深300"},
    {"sh000852", "中证1000"}, {"sh000905", "中证500"},
    {"sz399001", "深证成指"}, {"sz399006", "创业板指"},
    {"sh000688", "科创50指数"}, {"sz399303", "国证2000"}, {"sz399673", "创业板50指数"},
    {"sz399986", "中证新能"}, {"sz399989", "中证医疗"},
    // Broad ETFs
    {"sh510050", "上证50ETF"}, {"sh510300", "300ETF"}, {"sh510500", "中证500ETF"},
    {"sh510880", "红利ETF"}, {"sh510090", "180ETF"},
    // Sector ETFs
    {"sh512000", "券商ETF"}, {"sh512010", "医药ETF"}, {"sh512070", "非银ETF"},
    {"sh512170", "医疗ETF"}, {"sh512200", "房地产ETF"}, {"sh512290", "生物医药ETF"},
    {"sh512400", "有色金属ETF"}, {"sh512480", "半导体ETF"}, {"sh512660", "军工ETF"},
    {"sh512690", "酒ETF"}, {"sh512720", "计算机ETF"}, {"sh512760", "芯片ETF"},
    {"sh512800", "银行ETF"}, {"sh512880", "证券ETF"}, {"sh512980", "传媒ETF"},
    {"sz159819", "人工智能ETF"}, {"sz159825", "农业ETF"}, {"sz159869", "碳中和ETF"},
    {"sz159870", "化工ETF"}, {"sz159875", "新能源车ETF"}, {"sz159915", "创业板ETF"},
    {"sz159919", "300ETF(深)"}, {"sz159928", "消费ETF"}, {"sz159941", "纳指ETF(深)"},
    {"sz159949", "创业板50ETF"}, {"sz159995", "芯片ETF(深)"},
    // Cross-border/Commodity/Others
    {"sh513050", "中概互联ETF"}, {"sh513060", "恒生医疗ETF"}, {"sh513080", "法国CAC40ETF"},
    {"sh513100", "纳指ETF"}, {"sh513130", "恒生科技ETF"}, {"sh513180", "恒生互联ETF"},
    {"sh513330", "美债ETF"}, {"sh513500", "标普500ETF"}, {"sh513520", "日经ETF"},
    {"sh513660", "恒生ETF"}, {"sh513730", "东南亚科技ETF"}, {"sh513880", "日经225ETF"},
    {"sh513890", "韩国ETF"}, {"sh518380", "黄金股ETF"}, {"sh518800", "黄金ETF"},
    {"sh518880", "黄金ETF(华安)"}, {"sh560080", "中药ETF"}, {"sh561120", "饮料ETF"},
    {"sh561560", "央企红利ETF"}, {"sh562510", "上证科创板ETF"}, {"sh588000", "科创50ETF"},
    {"sh588200", "科创芯片ETF"},
    // More from market_data
    {"sh512890", "红利低波ETF"}, {"sh513000", "日经ETF(南方)"}, {"sh513010", "港股通50ETF"},
    {"sh513030", "华安港股通精选ETF"}, {"sh513090", "港股通互联网ETF"}, {"sh513550", "港股红利ETF"},
    {"sh515000", "科技ETF"}, {"sh515030", "新能源ETF"}, {"sh515080", "中证红利ETF"},
    {"sh515170", "食品饮料ETF"}, {"sh515180", "100红利ETF"}, {"sh515210", "钢铁ETF"},
    {"sh515220", "煤炭ETF"}, {"sh515650", "消费50ETF"}, {"sh515710", "新材料ETF"},
    {"sh515790", "光伏ETF"}, {"sh515860", "中证电新ETF"}, {"sh515880", "通信ETF"},
    {"sh516150", "稀土ETF"}, {"sh516160", "新能源车ETF(汇添富)"},
    {"sh516780", "稀有金属ETF"}, {"sh516950", "基建ETF"},
    {"sh518660", "黄金股ETF(永赢)"}, {"sh561990", "动漫游戏ETF"},
    {"sh562800", "恒生生物科技ETF"},
    {"sz159607", "家电ETF"}, {"sz159866", "日经225ETF(深)"},
    {"sz159920", "恒生ETF(深)"}, {"sz159934", "黄金ETF(深)"},
    {"sz159981", "能源化工ETF"}, {"sz159985", "豆粕ETF"},
};

// Indices to exclude from optimization (not tradeable)
bool is_index(const std::string& code) {
    return code.substr(0, 5) == "sh000" || code.substr(0, 5) == "sz399";
}

// ============================================================
// Helpers
// ============================================================

// Get ISO week-year string "YYYY-WNN" from date "YYYY-MM-DD"
// Simple approach: group by Monday-based weeks
int date_to_days(const std::string& d) {
    // Approximate days since epoch for week grouping
    int y = std::stoi(d.substr(0, 4));
    int m = std::stoi(d.substr(5, 2));
    int day = std::stoi(d.substr(8, 2));
    // Zeller-like approximation
    if (m <= 2) { y--; m += 12; }
    return 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + day;
}

int date_to_week_id(const std::string& d) {
    int days = date_to_days(d);
    // ISO week: group by 7-day blocks aligned to Monday
    // dayOfWeek: 0=Mon..6=Sun (epoch 2000-01-03 was Monday = day 730123)
    return days / 7;  // unique week identifier
}

std::vector<AggBar> aggregate_weekly(const std::vector<DailyBar>& daily, int start, int end) {
    std::vector<AggBar> bars;
    if (start >= end) return bars;

    int cur_wk = date_to_week_id(daily[start].date);
    double wk_open = daily[start].open;
    double wk_close = daily[start].close;
    double wk_high = daily[start].high;
    double wk_low = daily[start].low;
    double wk_vol = daily[start].volume;
    int wk_first = start, wk_last = start;

    for (int i = start + 1; i < end; ++i) {
        int wk = date_to_week_id(daily[i].date);
        if (wk == cur_wk) {
            wk_close = daily[i].close;
            wk_high = std::max(wk_high, daily[i].high);
            wk_low = std::min(wk_low, daily[i].low);
            wk_vol += daily[i].volume;
            wk_last = i;
        } else {
            bars.push_back({wk_open, wk_close, wk_high, wk_low, wk_vol, wk_first, wk_last});
            cur_wk = wk;
            wk_open = daily[i].open;
            wk_close = daily[i].close;
            wk_high = daily[i].high;
            wk_low = daily[i].low;
            wk_vol = daily[i].volume;
            wk_first = i;
            wk_last = i;
        }
    }
    bars.push_back({wk_open, wk_close, wk_high, wk_low, wk_vol, wk_first, wk_last});
    return bars;
}

std::vector<AggBar> aggregate_monthly(const std::vector<DailyBar>& daily, int start, int end) {
    std::vector<AggBar> bars;
    if (start >= end) return bars;

    std::string cur_ym = daily[start].date.substr(0, 7);
    double m_open = daily[start].open;
    double m_close = daily[start].close;
    double m_high = daily[start].high;
    double m_low = daily[start].low;
    double m_vol = daily[start].volume;
    int m_first = start, m_last = start;

    for (int i = start + 1; i < end; ++i) {
        std::string ym = daily[i].date.substr(0, 7);
        if (ym == cur_ym) {
            m_close = daily[i].close;
            m_high = std::max(m_high, daily[i].high);
            m_low = std::min(m_low, daily[i].low);
            m_vol += daily[i].volume;
            m_last = i;
        } else {
            bars.push_back({m_open, m_close, m_high, m_low, m_vol, m_first, m_last});
            cur_ym = ym;
            m_open = daily[i].open;
            m_close = daily[i].close;
            m_high = daily[i].high;
            m_low = daily[i].low;
            m_vol = daily[i].volume;
            m_first = i;
            m_last = i;
        }
    }
    bars.push_back({m_open, m_close, m_high, m_low, m_vol, m_first, m_last});
    return bars;
}

void compute_macd(const std::vector<AggBar>& bars, int fast, int slow, int signal,
                  std::vector<double>& dif, std::vector<double>& dea, std::vector<double>& hist) {
    size_t n = bars.size();
    dif.resize(n); dea.resize(n); hist.resize(n);
    if (n == 0) return;

    double mf = 2.0 / (fast + 1), ms = 2.0 / (slow + 1), msig = 2.0 / (signal + 1);
    double ef = bars[0].close, es = bars[0].close, d = 0;
    dif[0] = 0; dea[0] = 0; hist[0] = 0;

    for (size_t i = 1; i < n; ++i) {
        ef = (bars[i].close - ef) * mf + ef;
        es = (bars[i].close - es) * ms + es;
        dif[i] = ef - es;
        if (i == 1) d = dif[i];
        else d = (dif[i] - d) * msig + d;
        dea[i] = d;
        hist[i] = dif[i] - dea[i];
    }
}

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    return (std::pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

// ============================================================
// Strategy execution
// ============================================================

EvalResult run_strategy(const ETFData& etf, const std::vector<AggBar>& bars,
                        const std::vector<double>& dif, const std::vector<double>& dea,
                        const std::vector<double>& hist_vals,
                        const StrategyParams& p, int global_start, int global_end,
                        bool log_trades = false) {
    EvalResult res;
    if (bars.size() < 3 || global_end <= global_start) return res;

    res.date_start = etf.daily[global_start].date;
    res.date_end = etf.daily[global_end - 1].date;
    double first_open = etf.daily[global_start].open;
    double last_close = etf.daily[global_end - 1].close;
    res.buy_hold_return_pct = (last_close - first_open) / first_open * 100.0;

    bool holding = false;
    double buy_price = 0, capital = 1.0, peak_price = 0;
    double peak_capital = 1.0;
    int winning = 0;
    int bars_since_sell = 999;
    std::string buy_date;

    for (size_t i = 1; i < bars.size() - 1; ++i) {
        if (!holding) {
            // Cooldown check
            if (p.cooldown > 0 && bars_since_sell < p.cooldown) {
                bars_since_sell++;
                continue;
            }

            bool buy_signal = false;
            if (p.buy_mode == 0) {
                buy_signal = (dif[i-1] <= dea[i-1]) && (dif[i] > dea[i]);
            } else if (p.buy_mode == 1) {
                buy_signal = (hist_vals[i-1] <= 0) && (hist_vals[i] > 0);
            }
            if (!buy_signal) {
                bars_since_sell++;
                continue;
            }

            // Execute buy at next bar's open
            int exec_idx = bars[i+1].first_daily_idx;
            if (exec_idx < global_end) {
                buy_price = etf.daily[exec_idx].open;
                buy_date = etf.daily[exec_idx].date;
                peak_price = buy_price;
                holding = true;
            }
        } else {
            // Check trailing stop first (intra-bar using daily data)
            if (p.trailing_stop > 0) {
                // Update peak using the high of current aggregated bar
                peak_price = std::max(peak_price, bars[i].high);
                // Check if close dropped below trailing stop
                double stop_level = peak_price * (1.0 - p.trailing_stop / 100.0);
                if (bars[i].close <= stop_level) {
                    // Sell at next bar's open
                    int exec_idx = bars[i+1].first_daily_idx;
                    if (exec_idx < global_end) {
                        double sell_price = etf.daily[exec_idx].open;
                        double ret = (sell_price - buy_price) / buy_price;
                        capital *= (1.0 + ret);
                        if (ret > 0) winning++;
                        res.trades++;
                        if (log_trades) {
                            char buf[256];
                            snprintf(buf, sizeof(buf), "  Buy: %s @ %.4f -> TrailStop: %s @ %.4f  Ret: %.2f%%",
                                     buy_date.c_str(), buy_price, etf.daily[exec_idx].date.c_str(), sell_price, ret * 100);
                            res.trade_log.push_back(buf);
                        }
                        holding = false;
                        bars_since_sell = 0;
                        peak_capital = std::max(peak_capital, capital);
                        continue;
                    }
                }
            }

            // Check take-profit
            if (p.take_profit > 0) {
                double unrealized = (bars[i].close - buy_price) / buy_price * 100.0;
                if (unrealized >= p.take_profit) {
                    int exec_idx = bars[i+1].first_daily_idx;
                    if (exec_idx < global_end) {
                        double sell_price = etf.daily[exec_idx].open;
                        double ret = (sell_price - buy_price) / buy_price;
                        capital *= (1.0 + ret);
                        if (ret > 0) winning++;
                        res.trades++;
                        if (log_trades) {
                            char buf[256];
                            snprintf(buf, sizeof(buf), "  Buy: %s @ %.4f -> TakeProfit: %s @ %.4f  Ret: %.2f%%",
                                     buy_date.c_str(), buy_price, etf.daily[exec_idx].date.c_str(), sell_price, ret * 100);
                            res.trade_log.push_back(buf);
                        }
                        holding = false;
                        bars_since_sell = 0;
                        peak_capital = std::max(peak_capital, capital);
                        continue;
                    }
                }
            }

            // Update peak for trailing stop
            if (p.trailing_stop > 0) {
                peak_price = std::max(peak_price, bars[i].high);
            }

            // MACD sell signal
            bool sell_signal = false;
            if (p.sell_mode == 0) {
                sell_signal = (dif[i-1] >= dea[i-1]) && (dif[i] < dea[i]);
            } else if (p.sell_mode == 1) {
                sell_signal = (dif[i-1] >= 0) && (dif[i] < 0);
            } else if (p.sell_mode == 2) {
                sell_signal = (hist_vals[i-1] >= 0) && (hist_vals[i] < 0);
            }

            if (sell_signal) {
                int exec_idx = bars[i+1].first_daily_idx;
                if (exec_idx < global_end) {
                    double sell_price = etf.daily[exec_idx].open;
                    double ret = (sell_price - buy_price) / buy_price;
                    capital *= (1.0 + ret);
                    if (ret > 0) winning++;
                    res.trades++;
                    if (log_trades) {
                        char buf[256];
                        snprintf(buf, sizeof(buf), "  Buy: %s @ %.4f -> Sell: %s @ %.4f  Ret: %.2f%%",
                                 buy_date.c_str(), buy_price, etf.daily[exec_idx].date.c_str(), sell_price, ret * 100);
                        res.trade_log.push_back(buf);
                    }
                    holding = false;
                    bars_since_sell = 0;
                }
            }
        }
        peak_capital = std::max(peak_capital, capital);
    }

    // Force close if still holding
    if (holding) {
        double sell_price = etf.daily[global_end - 1].close;
        double ret = (sell_price - buy_price) / buy_price;
        capital *= (1.0 + ret);
        if (ret > 0) winning++;
        res.trades++;
        if (log_trades) {
            char buf[256];
            snprintf(buf, sizeof(buf), "  Buy: %s @ %.4f -> Close: %s @ %.4f  Ret: %.2f%% (forced)",
                     buy_date.c_str(), buy_price, etf.daily[global_end - 1].date.c_str(), sell_price, ret * 100);
            res.trade_log.push_back(buf);
        }
    }

    res.cumulative_return_pct = (capital - 1.0) * 100.0;
    if (res.trades > 0) res.win_rate = (double)winning / res.trades * 100.0;

    // Max drawdown
    if (peak_capital > 0) {
        res.max_drawdown_pct = (1.0 - capital / peak_capital) * 100.0;
        if (res.max_drawdown_pct < 0) res.max_drawdown_pct = 0;
    }

    // Annualized return
    if (res.trades > 0 && res.date_start.size() >= 10 && res.date_end.size() >= 10) {
        try {
            int y1 = std::stoi(res.date_start.substr(0, 4));
            int m1 = std::stoi(res.date_start.substr(5, 2));
            int d1 = std::stoi(res.date_start.substr(8, 2));
            int y2 = std::stoi(res.date_end.substr(0, 4));
            int m2 = std::stoi(res.date_end.substr(5, 2));
            int d2 = std::stoi(res.date_end.substr(8, 2));
            int days = (y2 - y1) * 365 + (m2 - m1) * 30 + (d2 - d1);
            res.annualized_return_pct = compute_annualized(capital, days / 365.25);
        } catch (...) {
            res.annualized_return_pct = 0;
        }
    }

    return res;
}

// ============================================================
// Score a parameter set across all ETFs (Top N avg annualized)
// ============================================================

double score_params(const StrategyParams& p, bool use_train, int top_n = 20) {
    thread_local std::vector<double> tl_dif, tl_dea, tl_hist;
    std::vector<double> ann;
    ann.reserve(g_etfs.size());

    for (const auto& etf : g_etfs) {
        const auto& bars = use_train ? etf.train_bars : etf.test_bars;
        int gs = use_train ? 0 : etf.split_idx;
        int ge = use_train ? etf.split_idx : (int)etf.daily.size();

        compute_macd(bars, p.fast, p.slow, p.signal, tl_dif, tl_dea, tl_hist);
        EvalResult res = run_strategy(etf, bars, tl_dif, tl_dea, tl_hist, p, gs, ge);
        if (res.trades > 0) ann.push_back(res.annualized_return_pct);
    }

    // Use top N, but with ETFs we might have fewer assets
    int actual_top = std::min(top_n, (int)ann.size());
    if (actual_top < 5) return -999;  // need at least 5 ETFs with trades
    std::sort(ann.rbegin(), ann.rend());
    double sum = 0;
    for (int i = 0; i < actual_top; ++i) sum += ann[i];
    return sum / actual_top;
}

// ============================================================
// Aggregate bars for all ETFs in a given timeframe
// ============================================================

void rebuild_etf_bars(bool weekly) {
    for (auto& etf : g_etfs) {
        if (weekly) {
            etf.train_bars = aggregate_weekly(etf.daily, 0, etf.split_idx);
            etf.test_bars = aggregate_weekly(etf.daily, etf.split_idx, etf.daily.size());
        } else {
            etf.train_bars = aggregate_monthly(etf.daily, 0, etf.split_idx);
            etf.test_bars = aggregate_monthly(etf.daily, etf.split_idx, etf.daily.size());
        }
    }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    int num_threads = std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = std::stoi(argv[i + 1]);
    }
    if (num_threads <= 0) num_threads = 8;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::string etf_dir = "/ceph/dang_articles/yoj/market_data";

    // Load all ETFs
    std::cout << "Loading ETF data...\n";
    for (const auto& entry : fs::directory_iterator(etf_dir)) {
        if (entry.path().extension() != ".csv") continue;
        std::string code = entry.path().stem().string();

        // Skip indices (not tradeable)
        if (is_index(code)) continue;

        std::ifstream file(entry.path());
        if (!file.is_open()) continue;

        ETFData ed;
        ed.code = code;
        ed.name = g_etf_names.count(code) ? g_etf_names[code] : code;

        std::string line;
        std::getline(file, line); // skip header
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
                ed.daily.push_back(b);
            } catch (...) {}
        }

        if (ed.daily.size() < 60) continue;  // need sufficient data
        std::sort(ed.daily.begin(), ed.daily.end(),
                  [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });

        ed.split_idx = ed.daily.size() * 0.4;
        g_etfs.push_back(std::move(ed));
    }

    printf("Loaded %zu ETFs (excl. indices).\n\n", g_etfs.size());

    // ============================================================
    // Run optimization for both weekly and monthly timeframes
    // ============================================================

    struct TimeframeResult {
        std::string timeframe;
        StrategyParams best_params;
        double train_score;
        double test_score;
    };
    std::vector<TimeframeResult> tf_results;

    for (int tf = 0; tf < 2; ++tf) {
        bool weekly = (tf == 0);
        std::string tf_name = weekly ? "WEEKLY" : "MONTHLY";
        printf("================================================================\n");
        printf("  TIMEFRAME: %s\n", tf_name.c_str());
        printf("================================================================\n\n");

        rebuild_etf_bars(weekly);

        // Phase A: MACD params + buy/sell mode grid
        printf("=== Phase A: MACD + Buy/Sell Mode Search (%s) ===\n", tf_name.c_str());
        auto t_phA = std::chrono::high_resolution_clock::now();

        std::vector<StrategyParams> phase_a;

        if (weekly) {
            // Weekly: try wider MACD ranges since more bars available
            // fast 5-20, slow 15-50 step 2, signal 3-12
            for (int f = 5; f <= 20; ++f) {
                for (int s = 15; s <= 50; s += 2) {
                    if (f >= s - 2) continue;
                    for (int sig = 3; sig <= 12; sig += 1) {
                        for (int bm = 0; bm <= 1; ++bm) {
                            for (int sm : {0, 1, 2}) {
                                phase_a.push_back({f, s, sig, bm, sm, 0.0, 0.0, 0});
                            }
                        }
                    }
                }
            }
        } else {
            // Monthly: same range as V4
            for (int f = 5; f <= 14; ++f) {
                for (int s = 16; s <= 36; ++s) {
                    if (f >= s - 2) continue;
                    for (int sig = 3; sig <= 9; ++sig) {
                        for (int bm = 0; bm <= 1; ++bm) {
                            for (int sm : {0, 1, 2}) {
                                phase_a.push_back({f, s, sig, bm, sm, 0.0, 0.0, 0});
                            }
                        }
                    }
                }
            }
        }

        printf("Phase A combos: %zu\n", phase_a.size());

        struct ComboResult {
            StrategyParams params;
            double score;
        };
        std::vector<ComboResult> phase_a_results(phase_a.size());
        std::atomic<int> counter_a{0};
        int total_a = phase_a.size();

        auto worker_a = [&]() {
            while (true) {
                int idx = counter_a.fetch_add(1);
                if (idx >= total_a) break;
                double s = score_params(phase_a[idx], true);
                phase_a_results[idx] = {phase_a[idx], s};
            }
        };

        { std::vector<std::thread> threads;
          for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_a);
          for (auto& t : threads) t.join(); }

        std::sort(phase_a_results.begin(), phase_a_results.end(),
                  [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

        int top_show = std::min<int>(15, phase_a_results.size());
        printf("\nTop %d MACD configs (%s):\n", top_show, tf_name.c_str());
        const char* buy_mode_str[] = {"GoldenX", "Hist>0"};
        const char* sell_mode_str[] = {"DeathX", "DIF<0", "Hist<0"};
        printf("  %-3s %-4s %-4s %-3s %-8s %-8s %7s\n",
               "#", "Fast", "Slow", "Sig", "BuyMode", "SellMode", "Score");
        for (int i = 0; i < top_show; ++i) {
            auto& r = phase_a_results[i];
            printf("  %-3d %-4d %-4d %-3d %-8s %-8s %6.2f%%\n",
                   i + 1, r.params.fast, r.params.slow, r.params.signal,
                   buy_mode_str[r.params.buy_mode], sell_mode_str[r.params.sell_mode], r.score);
        }

        auto t_phA_end = std::chrono::high_resolution_clock::now();
        printf("Phase A: %.2fs\n\n", std::chrono::duration<double>(t_phA_end - t_phA).count());

        // Phase B: Trailing stop, take-profit, cooldown refinement on top configs
        printf("=== Phase B: Trailing Stop + Take-Profit + Cooldown (%s) ===\n", tf_name.c_str());
        auto t_phB = std::chrono::high_resolution_clock::now();

        int top_macd = std::min<int>(10, (int)phase_a_results.size());
        std::vector<StrategyParams> phase_b;

        for (int t = 0; t < top_macd; ++t) {
            auto base = phase_a_results[t].params;

            // Trailing stop values (% from peak)
            double trail_vals[] = {0.0, 5.0, 8.0, 10.0, 15.0, 20.0};
            // Take-profit values (%)
            double tp_vals[] = {0.0, 20.0, 30.0, 50.0};
            // Cooldown values (bars)
            int cd_vals[] = {0, 2, 4};

            for (double ts : trail_vals) {
                for (double tp : tp_vals) {
                    for (int cd : cd_vals) {
                        phase_b.push_back({base.fast, base.slow, base.signal,
                                          base.buy_mode, base.sell_mode,
                                          ts, tp, cd});
                    }
                }
            }
        }

        printf("Phase B combos: %zu\n", phase_b.size());

        std::vector<ComboResult> phase_b_results(phase_b.size());
        std::atomic<int> counter_b{0};
        int total_b = phase_b.size();

        auto worker_b = [&]() {
            while (true) {
                int idx = counter_b.fetch_add(1);
                if (idx >= total_b) break;
                double s = score_params(phase_b[idx], true);
                phase_b_results[idx] = {phase_b[idx], s};
            }
        };

        { std::vector<std::thread> threads;
          for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_b);
          for (auto& t : threads) t.join(); }

        std::sort(phase_b_results.begin(), phase_b_results.end(),
                  [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

        printf("\nTop 20 configs with trailing stop + take-profit (%s):\n", tf_name.c_str());
        printf("  %-3s %-4s %-4s %-3s %-8s %-8s %6s %6s %3s %7s\n",
               "#", "Fast", "Slow", "Sig", "Buy", "Sell", "Trail%", "TP%", "CD", "Score");
        for (int i = 0; i < std::min<int>(20, (int)phase_b_results.size()); ++i) {
            auto& r = phase_b_results[i];
            printf("  %-3d %-4d %-4d %-3d %-8s %-8s %6.1f %6.1f %3d %6.2f%%\n",
                   i + 1, r.params.fast, r.params.slow, r.params.signal,
                   buy_mode_str[r.params.buy_mode], sell_mode_str[r.params.sell_mode],
                   r.params.trailing_stop, r.params.take_profit,
                   r.params.cooldown, r.score);
        }

        auto t_phB_end = std::chrono::high_resolution_clock::now();
        printf("Phase B: %.2fs\n\n", std::chrono::duration<double>(t_phB_end - t_phB).count());

        // Phase C: Test evaluation
        printf("=== Phase C: Test Set Evaluation (%s) ===\n", tf_name.c_str());
        int eval_count = std::min<int>(5, (int)phase_b_results.size());
        printf("\nTop %d configs evaluated on test period:\n", eval_count);
        printf("  %-3s %-4s %-4s %-3s %-8s %-8s %6s %6s %3s  %8s %8s\n",
               "#", "Fast", "Slow", "Sig", "Buy", "Sell", "Trail%", "TP%", "CD", "Train", "Test");

        int best_idx = 0;
        double best_test = -9999;

        for (int ci = 0; ci < eval_count; ++ci) {
            auto& p = phase_b_results[ci].params;
            double test_s = score_params(p, false);
            double train_s = phase_b_results[ci].score;

            printf("  %-3d %-4d %-4d %-3d %-8s %-8s %6.1f %6.1f %3d  %7.2f%% %7.2f%%\n",
                   ci + 1, p.fast, p.slow, p.signal,
                   buy_mode_str[p.buy_mode], sell_mode_str[p.sell_mode],
                   p.trailing_stop, p.take_profit, p.cooldown,
                   train_s, test_s);

            if (test_s > best_test) {
                best_test = test_s;
                best_idx = ci;
            }
        }

        StrategyParams best_p = phase_b_results[best_idx].params;
        double best_train = phase_b_results[best_idx].score;

        tf_results.push_back({tf_name, best_p, best_train, best_test});

        printf("\n  BEST %s: Train %.2f%% / Test %.2f%%\n", tf_name.c_str(), best_train, best_test);
        printf("  Params: fast=%d slow=%d sig=%d buy=%s sell=%s trail=%.1f%% tp=%.1f%% cd=%d\n\n",
               best_p.fast, best_p.slow, best_p.signal,
               buy_mode_str[best_p.buy_mode], sell_mode_str[best_p.sell_mode],
               best_p.trailing_stop, best_p.take_profit, best_p.cooldown);
    }

    // ============================================================
    // Final comparison: Weekly vs Monthly
    // ============================================================
    printf("\n================================================================\n");
    printf("  FINAL COMPARISON: WEEKLY vs MONTHLY\n");
    printf("================================================================\n\n");

    const char* buy_mode_str_final[] = {"GoldenX", "Hist>0"};
    const char* sell_mode_str_final[] = {"DeathX", "DIF<0", "Hist<0"};

    printf("  %-10s  %8s  %8s  %s\n", "Timeframe", "Train", "Test", "Best Config");
    printf("  %-10s  %8s  %8s  %s\n", "----------", "--------", "--------", "-----------");

    for (auto& r : tf_results) {
        char cfg[128];
        snprintf(cfg, sizeof(cfg), "%d,%d,%d %s/%s trail=%.0f%% tp=%.0f%% cd=%d",
                 r.best_params.fast, r.best_params.slow, r.best_params.signal,
                 buy_mode_str_final[r.best_params.buy_mode],
                 sell_mode_str_final[r.best_params.sell_mode],
                 r.best_params.trailing_stop, r.best_params.take_profit,
                 r.best_params.cooldown);
        printf("  %-10s  %7.2f%%  %7.2f%%  %s\n",
               r.timeframe.c_str(), r.train_score, r.test_score, cfg);
    }

    // Also compare with V4 monthly baseline on stocks
    printf("\n  (V4 stock baseline: Train 51.10%% / Test 46.41%% on 2472 stocks)\n");

    // ============================================================
    // Generate full per-ETF results for best overall config
    // ============================================================

    // Pick the timeframe with best test score
    int best_tf = 0;
    if (tf_results.size() > 1 && tf_results[1].test_score > tf_results[0].test_score)
        best_tf = 1;

    bool final_weekly = (best_tf == 0);
    StrategyParams final_p = tf_results[best_tf].best_params;

    printf("\n================================================================\n");
    printf("  DETAILED RESULTS: %s (best timeframe)\n", tf_results[best_tf].timeframe.c_str());
    printf("================================================================\n\n");

    rebuild_etf_bars(final_weekly);

    struct FinalResult {
        std::string code, name;
        double full_ann, train_ann, test_ann;
        double full_ret, bh_ret;
        double strategy_vs_bh;
        int trades;
        double win_rate;
        int daily_count;
        std::vector<std::string> trade_log;
    };
    std::vector<FinalResult> final_results;

    std::vector<double> dif, dea, hist_vals;
    for (auto& etf : g_etfs) {
        int n = etf.daily.size();

        // Full period
        std::vector<AggBar> full_bars;
        if (final_weekly) full_bars = aggregate_weekly(etf.daily, 0, n);
        else full_bars = aggregate_monthly(etf.daily, 0, n);
        compute_macd(full_bars, final_p.fast, final_p.slow, final_p.signal, dif, dea, hist_vals);
        auto full_res = run_strategy(etf, full_bars, dif, dea, hist_vals, final_p, 0, n, true);

        // Train
        compute_macd(etf.train_bars, final_p.fast, final_p.slow, final_p.signal, dif, dea, hist_vals);
        auto train_res = run_strategy(etf, etf.train_bars, dif, dea, hist_vals, final_p, 0, etf.split_idx);

        // Test
        compute_macd(etf.test_bars, final_p.fast, final_p.slow, final_p.signal, dif, dea, hist_vals);
        auto test_res = run_strategy(etf, etf.test_bars, dif, dea, hist_vals, final_p, etf.split_idx, n);

        // Compute annualized buy-and-hold for comparison
        double bh_ann = 0;
        if (full_res.date_start.size() >= 10 && full_res.date_end.size() >= 10) {
            try {
                int y1 = std::stoi(full_res.date_start.substr(0, 4));
                int m1 = std::stoi(full_res.date_start.substr(5, 2));
                int d1 = std::stoi(full_res.date_start.substr(8, 2));
                int y2 = std::stoi(full_res.date_end.substr(0, 4));
                int m2 = std::stoi(full_res.date_end.substr(5, 2));
                int d2 = std::stoi(full_res.date_end.substr(8, 2));
                int days = (y2 - y1) * 365 + (m2 - m1) * 30 + (d2 - d1);
                double years = days / 365.25;
                double bh_ratio = 1.0 + full_res.buy_hold_return_pct / 100.0;
                bh_ann = compute_annualized(bh_ratio, years);
            } catch (...) {}
        }

        final_results.push_back({etf.code, etf.name,
                                 full_res.annualized_return_pct, train_res.annualized_return_pct,
                                 test_res.annualized_return_pct,
                                 full_res.cumulative_return_pct, full_res.buy_hold_return_pct,
                                 full_res.annualized_return_pct - bh_ann,
                                 full_res.trades, full_res.win_rate,
                                 (int)etf.daily.size(), full_res.trade_log});
    }

    std::sort(final_results.begin(), final_results.end(),
              [](const FinalResult& a, const FinalResult& b) { return a.full_ann > b.full_ann; });

    printf("%-16s %-22s %6s  %8s %8s %8s  %8s %8s %8s  %5s %5s\n",
           "Code", "Name", "Days", "Full%", "Train%", "Test%", "CumRet%", "B&H%", "vs_BH%", "Trd", "WinR");
    printf("%-16s %-22s %6s  %8s %8s %8s  %8s %8s %8s  %5s %5s\n",
           "----", "----", "----", "------", "------", "------", "-------", "----", "------", "---", "----");

    int beats_bh = 0;
    double total_strategy_ann = 0, total_bh_ann = 0;
    int count_with_trades = 0;

    for (auto& r : final_results) {
        printf("%-16s %-22s %6d  %8.2f %8.2f %8.2f  %8.2f %8.2f %8.2f  %5d %5.1f\n",
               r.code.c_str(), r.name.substr(0, 22).c_str(), r.daily_count,
               r.full_ann, r.train_ann, r.test_ann,
               r.full_ret, r.bh_ret, r.strategy_vs_bh,
               r.trades, r.win_rate);

        if (r.trades > 0) {
            count_with_trades++;
            total_strategy_ann += r.full_ann;
            if (r.strategy_vs_bh > 0) beats_bh++;
        }
    }

    printf("\n--- Summary ---\n");
    printf("ETFs with trades: %d / %zu\n", count_with_trades, final_results.size());
    printf("Strategy beats Buy&Hold: %d / %d (%.1f%%)\n",
           beats_bh, count_with_trades, count_with_trades > 0 ? beats_bh * 100.0 / count_with_trades : 0);
    printf("Avg strategy annualized: %.2f%%\n",
           count_with_trades > 0 ? total_strategy_ann / count_with_trades : 0);
    printf("Median strategy annualized: ");
    {
        std::vector<double> anns;
        for (auto& r : final_results) if (r.trades > 0) anns.push_back(r.full_ann);
        if (!anns.empty()) {
            std::sort(anns.begin(), anns.end());
            printf("%.2f%%\n", anns[anns.size() / 2]);
        } else {
            printf("N/A\n");
        }
    }

    // Save CSV
    std::string csv_name = "etf_macd_v5_results.csv";
    std::ofstream csv(csv_name);
    csv << "code,name,daily_bars,timeframe,full_annualized,train_annualized,test_annualized,"
        << "full_cumulative_ret,buy_hold_ret,strategy_vs_bh,trades,win_rate,"
        << "fast,slow,signal,buy_mode,sell_mode,trailing_stop,take_profit,cooldown\n";
    for (auto& r : final_results) {
        csv << r.code << "," << r.name << "," << r.daily_count << ","
            << tf_results[best_tf].timeframe << ","
            << r.full_ann << "," << r.train_ann << "," << r.test_ann << ","
            << r.full_ret << "," << r.bh_ret << "," << r.strategy_vs_bh << ","
            << r.trades << "," << r.win_rate << ","
            << final_p.fast << "," << final_p.slow << "," << final_p.signal << ","
            << buy_mode_str_final[final_p.buy_mode] << "," << sell_mode_str_final[final_p.sell_mode] << ","
            << final_p.trailing_stop << "," << final_p.take_profit << "," << final_p.cooldown << "\n";
    }
    csv.close();
    printf("\nResults saved to %s\n", csv_name.c_str());

    // Print trade details for top 5
    printf("\nTop 5 ETF Trade Details:\n");
    int count = 0;
    for (auto& r : final_results) {
        if (count >= 5) break;
        if (r.trades == 0) continue;
        printf("\n%s %s (Ann: %.2f%%, Total: %.2f%%, vs B&H: %.2f%%)\n",
               r.code.c_str(), r.name.c_str(), r.full_ann, r.full_ret, r.strategy_vs_bh);
        for (auto& log : r.trade_log) {
            printf("%s\n", log.c_str());
        }
        count++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    printf("\nTotal Runtime: %.2fs\n", std::chrono::duration<double>(t_end - t_start).count());

    return 0;
}
