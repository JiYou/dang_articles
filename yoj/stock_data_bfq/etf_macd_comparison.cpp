/*
 * ETF MACD Strategy Comparison — Unified Portfolio Backtest
 * 
 * Compares V4 (monthly), V5 (weekly), V7 (dual-TF), V8 (sector rotation),
 * V10-Sharpe, V10-Calmar strategies on a shared portfolio basis.
 * 
 * Architecture based on V10's proven d_to_w/w_to_m mapping.
 * Each strategy uses its own MACD parameters, buy/sell logic, and risk management.
 * All strategies start with $1M capital at test period start (60% of data).
 * 
 * ETF CSV format: date,open,high,low,close,volume
 * Compiles: g++ -O3 -std=c++17 -pthread -o etf_macd_comparison etf_macd_comparison.cpp
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <filesystem>
#include <iomanip>
#include <set>
#include <numeric>

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Data structures
// ============================================================

struct DailyBar {
    string date;
    double open, high, low, close, volume;
};

struct AggBar {
    double open, close, high, low, volume;
    int first_daily_idx;
    int last_daily_idx;
};

struct ETFData {
    string code, name;
    int sector_id = -1;
    vector<DailyBar> daily;
    int split_idx;
    
    // Aggregated from FULL data (for MACD warmup) 
    vector<AggBar> weekly_bars;
    vector<AggBar> monthly_bars;
    
    // Global-indexed price arrays
    vector<double> global_close;
    vector<double> global_open;
    vector<double> global_high;
};

struct Position {
    int etf_idx;
    double entry_price;
    double highest_price;
    string entry_date;
    double shares;
};

struct BacktestResult {
    string version;
    double ann_return;
    double mdd;
    double sharpe;
    int total_trades;
    double win_rate;
};

// ============================================================
// Globals
// ============================================================

unordered_map<string, string> g_etf_names = {
    {"sh510050", "上证50ETF"}, {"sh510300", "300ETF"}, {"sh510500", "中证500ETF"},
    {"sh510880", "红利ETF"}, {"sh510090", "180ETF"},
    {"sh512000", "券商ETF"}, {"sh512010", "医药ETF"}, {"sh512070", "非银ETF"},
    {"sh512170", "医疗ETF"}, {"sh512200", "房地产ETF"}, {"sh512290", "生物医药ETF"},
    {"sh512400", "有色金属ETF"}, {"sh512480", "半导体ETF"}, {"sh512660", "军工ETF"},
    {"sh512690", "酒ETF"}, {"sh512720", "计算机ETF"}, {"sh512760", "芯片ETF"},
    {"sh512800", "银行ETF"}, {"sh512880", "证券ETF"}, {"sh512980", "传媒ETF"},
    {"sz159819", "人工智能ETF"}, {"sz159825", "农业ETF"}, {"sz159869", "碳中和ETF"},
    {"sz159870", "化工ETF"}, {"sz159875", "新能源车ETF"}, {"sz159915", "创业板ETF"},
    {"sz159919", "300ETF(深)"}, {"sz159928", "消费ETF"}, {"sz159941", "纳指ETF(深)"},
    {"sz159949", "创业板50ETF"}, {"sz159995", "芯片ETF(深)"},
    {"sh513050", "中概互联ETF"}, {"sh513060", "恒生医疗ETF"}, {"sh513080", "法国CAC40ETF"},
    {"sh513100", "纳指ETF"}, {"sh513130", "恒生科技ETF"}, {"sh513180", "恒生互联ETF"},
    {"sh513330", "美债ETF"}, {"sh513500", "标普500ETF"}, {"sh513520", "日经ETF"},
    {"sh513660", "恒生ETF"}, {"sh513730", "东南亚科技ETF"}, {"sh513880", "日经225ETF"},
    {"sh513890", "韩国ETF"}, {"sh518380", "黄金股ETF"}, {"sh518800", "黄金ETF"},
    {"sh518880", "黄金ETF(华安)"}, {"sh560080", "中药ETF"}, {"sh561120", "饮料ETF"},
    {"sh561560", "央企红利ETF"}, {"sh562510", "上证科创板ETF"}, {"sh588000", "科创50ETF"},
    {"sh588200", "科创芯片ETF"},
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
    {"sz159981", "能源化工ETF"}, {"sz159985", "豆粕ETF"}
};

unordered_map<string, int> g_date_to_idx;
vector<string> global_dates;

// Sector definitions for V8
unordered_map<string, int> sector_map;
void init_sectors() {
    vector<vector<string>> sectors = {
        {"sh510050", "sh510300", "sh510500", "sh510880", "sh510090", "sz159915", "sz159919", "sz159949"},
        {"sh512480", "sh512760", "sh512720", "sz159995", "sz159819", "sh515000", "sh588000", "sh588200", "sh562510"},
        {"sh512000", "sh512070", "sh512800", "sh512880", "sh512890"},
        {"sh512010", "sh512170", "sh512290", "sh513060", "sh560080", "sh562800"},
        {"sh512690", "sh515170", "sh515650", "sz159928", "sz159825", "sh561120", "sh561990"},
        {"sh515030", "sh516160", "sz159875", "sz159869", "sh515860", "sh515790"},
        {"sh512400", "sh516150", "sh516780", "sh515710", "sh515210", "sh515220", "sz159870", "sz159981"},
        {"sh512660"},
        {"sh516950", "sh512200", "sh515880", "sh512980"},
        {"sh510880", "sh515080", "sh515180", "sh561560", "sh513550"},
        {"sh518800", "sh518880", "sz159934", "sh518380", "sh518660", "sz159985"},
        {"sh513100", "sh513050", "sh513080", "sh513130", "sh513180", "sh513330", "sh513500", "sh513520", "sh513660", "sh513730", "sh513880", "sh513890", "sz159941", "sh513000", "sh513010", "sh513030", "sh513090"},
    };
    for (size_t i = 0; i < sectors.size(); ++i)
        for (const auto& c : sectors[i]) sector_map[c] = i;
}

// ============================================================
// Aggregation & MACD
// ============================================================

int date_to_days(const string& d) {
    if (d.size() < 10) return 0;
    int y = stoi(d.substr(0, 4));
    int m = stoi(d.substr(5, 2));
    int day = stoi(d.substr(8, 2));
    if (m <= 2) { y--; m += 12; }
    return 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + day;
}
int date_to_week_id(const string& d) { return date_to_days(d) / 7; }
int date_to_month_id(const string& d) {
    if (d.size() < 10) return 0;
    return stoi(d.substr(0, 4)) * 12 + stoi(d.substr(5, 2));
}

vector<AggBar> aggregate_weekly(const vector<DailyBar>& daily, int start, int end) {
    vector<AggBar> bars;
    if (start >= end || start >= (int)daily.size()) return bars;
    int cid = date_to_week_id(daily[start].date);
    AggBar b = {daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    for (int i = start + 1; i < end; ++i) {
        int id = date_to_week_id(daily[i].date);
        if (id != cid) { bars.push_back(b); cid = id; b = {daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i}; }
        else { b.high = max(b.high, daily[i].high); b.low = min(b.low, daily[i].low); b.close = daily[i].close; b.volume += daily[i].volume; b.last_daily_idx = i; }
    }
    bars.push_back(b);
    return bars;
}

vector<AggBar> aggregate_monthly(const vector<DailyBar>& daily, int start, int end) {
    vector<AggBar> bars;
    if (start >= end || start >= (int)daily.size()) return bars;
    int cid = date_to_month_id(daily[start].date);
    AggBar b = {daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    for (int i = start + 1; i < end; ++i) {
        int id = date_to_month_id(daily[i].date);
        if (id != cid) { bars.push_back(b); cid = id; b = {daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i}; }
        else { b.high = max(b.high, daily[i].high); b.low = min(b.low, daily[i].low); b.close = daily[i].close; b.volume += daily[i].volume; b.last_daily_idx = i; }
    }
    bars.push_back(b);
    return bars;
}

void compute_macd(const vector<AggBar>& bars, int fast, int slow, int signal,
                  vector<double>& dif, vector<double>& dea, vector<double>& hist) {
    size_t n = bars.size();
    dif.assign(n, 0.0); dea.assign(n, 0.0); hist.assign(n, 0.0);
    if (n == 0) return;
    double mf = 2.0 / (fast + 1), ms = 2.0 / (slow + 1), msig = 2.0 / (signal + 1);
    double ef = bars[0].close, es = bars[0].close, d = 0;
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

// ============================================================
// Strategy parameter sets
// ============================================================

struct StrategyConfig {
    string name;
    // Monthly MACD params (for V4 monthly-only, or monthly filter for dual-TF)
    int m_fast, m_slow, m_sig;
    // Weekly MACD params
    int w_fast, w_slow, w_sig;
    // Strategy modes
    bool monthly_only;    // V4: use monthly bars only
    bool weekly_only;     // V5: use weekly bars only
    bool use_sector;      // V8: sector rotation filter
    // Buy signal type for monthly-only
    int m_buy_mode;       // 0=golden cross, 1=histogram>0
    // Monthly trend mode for dual-TF (0=DIF>0, 1=DIF>DEA, 2=Hist>0, 3=DIF rising)
    int m_trend_mode;
    // Weekly buy mode for weekly/dual-TF (0=golden cross, 1=histogram>0)
    int w_buy_mode;
    // Sell modes
    int m_sell_mode;      // For monthly-only: 0=death cross, 1=DIF<0
    int w_sell_mode;      // For weekly: 0=death cross, 1=DIF<0, 2=histogram<0 crossover, 3=histogram<0 level
    // Risk management
    double trailing_stop; // % from peak to sell (0=off)
    double take_profit;   // % gain to sell (0=off)
    int max_positions;    // 0=unlimited
};

// ============================================================
// Metrics calculation
// ============================================================

BacktestResult calc_metrics(const string& version, const vector<double>& equity, int trades, int wins, int test_days) {
    BacktestResult res;
    res.version = version;
    res.total_trades = trades;
    res.win_rate = trades > 0 ? (double)wins / trades * 100.0 : 0;
    
    if (equity.size() < 2) {
        res.ann_return = 0; res.mdd = 0; res.sharpe = 0;
        return res;
    }
    
    double years = (double)test_days / 252.0;
    if (years <= 0) years = 1.0;
    res.ann_return = (pow(equity.back() / equity.front(), 1.0 / years) - 1.0) * 100.0;
    
    double peak = equity[0];
    double mdd = 0;
    double sum_ret = 0;
    vector<double> daily_ret;
    daily_ret.reserve(equity.size() - 1);
    
    for (size_t i = 1; i < equity.size(); ++i) {
        if (equity[i] > peak) peak = equity[i];
        double dd = (peak - equity[i]) / peak;
        if (dd > mdd) mdd = dd;
        double r = (equity[i] - equity[i-1]) / equity[i-1];
        daily_ret.push_back(r);
        sum_ret += r;
    }
    res.mdd = mdd * 100.0;
    
    if (daily_ret.empty()) { res.sharpe = 0; return res; }
    
    double mean_ret = sum_ret / daily_ret.size();
    double var_ret = 0;
    for (double r : daily_ret) var_ret += (r - mean_ret) * (r - mean_ret);
    double vol = sqrt(var_ret / daily_ret.size()) * sqrt(252);
    
    res.sharpe = vol > 1e-6 ? (res.ann_return / 100.0 - 0.02) / vol : 0;
    return res;
}

// ============================================================
// Main Strategy Runner (using V10's proven architecture)
// ============================================================

BacktestResult run_strategy(const StrategyConfig& cfg, vector<ETFData>& etfs, 
                            int start_g_idx, int end_g_idx) {
    int num_etfs = etfs.size();
    
    // Precompute MACD for all ETFs
    vector<vector<double>> m_dif(num_etfs), m_dea(num_etfs), m_hist(num_etfs);
    vector<vector<double>> w_dif(num_etfs), w_dea(num_etfs), w_hist(num_etfs);
    
    // Build d_to_w, d_to_m, w_to_m mappings (global-date-indexed)
    vector<vector<int>> d_to_w(num_etfs, vector<int>(global_dates.size(), -1));
    vector<vector<int>> d_to_m(num_etfs, vector<int>(global_dates.size(), -1));
    vector<vector<int>> w_to_m(num_etfs);
    
    for (int i = 0; i < num_etfs; ++i) {
        compute_macd(etfs[i].monthly_bars, cfg.m_fast, cfg.m_slow, cfg.m_sig, m_dif[i], m_dea[i], m_hist[i]);
        compute_macd(etfs[i].weekly_bars, cfg.w_fast, cfg.w_slow, cfg.w_sig, w_dif[i], w_dea[i], w_hist[i]);
        
        for (size_t m = 0; m < etfs[i].monthly_bars.size(); ++m) {
            for (int d = etfs[i].monthly_bars[m].first_daily_idx; d <= etfs[i].monthly_bars[m].last_daily_idx; ++d) {
                if (g_date_to_idx.count(etfs[i].daily[d].date))
                    d_to_m[i][g_date_to_idx[etfs[i].daily[d].date]] = m;
            }
        }
        for (size_t w = 0; w < etfs[i].weekly_bars.size(); ++w) {
            for (int d = etfs[i].weekly_bars[w].first_daily_idx; d <= etfs[i].weekly_bars[w].last_daily_idx; ++d) {
                if (g_date_to_idx.count(etfs[i].daily[d].date))
                    d_to_w[i][g_date_to_idx[etfs[i].daily[d].date]] = w;
            }
        }
        
        w_to_m[i].assign(etfs[i].weekly_bars.size(), -1);
        int mi = 0;
        for (size_t w = 0; w < etfs[i].weekly_bars.size(); ++w) {
            while (mi + 1 < (int)etfs[i].monthly_bars.size() &&
                   etfs[i].monthly_bars[mi].last_daily_idx < etfs[i].weekly_bars[w].first_daily_idx)
                mi++;
            if (mi < (int)etfs[i].monthly_bars.size() &&
                etfs[i].monthly_bars[mi].first_daily_idx <= etfs[i].weekly_bars[w].last_daily_idx)
                w_to_m[i][w] = mi;
        }
    }
    
    // Portfolio state
    double cash = 1000000;
    vector<Position> positions;
    vector<double> equity_curve;
    int trades = 0, wins = 0;
    
    // Sector momentum state for V8
    int sector_lookback = 42;  // ~2 months of daily bars
    int sector_topK = 7;
    
    for (int d = start_g_idx; d <= end_g_idx; ++d) {
        // --- 1. SELLS (evaluate at end of today, execute next open) ---
        vector<Position> remaining;
        for (auto& pos : positions) {
            int ei = pos.etf_idx;
            double cur_close = etfs[ei].global_close[d];
            double cur_high = etfs[ei].global_high[d];
            if (cur_high > pos.highest_price) pos.highest_price = cur_high;
            
            bool sell = false;
            
            if (cfg.monthly_only) {
                // V4: sell at end of month bar when DIF < 0
                int mi = d_to_m[ei][d];
                if (mi > 0 && d == g_date_to_idx[etfs[ei].daily[etfs[ei].monthly_bars[mi].last_daily_idx].date]) {
                    if (cfg.m_sell_mode == 0) {
                        // Death cross
                        sell = (m_dif[ei][mi] < m_dea[ei][mi] && m_dif[ei][mi-1] >= m_dea[ei][mi-1]);
                    } else if (cfg.m_sell_mode == 1) {
                        // DIF < 0
                        sell = (m_dif[ei][mi] < 0);
                    }
                }
            } else {
                // Weekly-based sell (V5, V7, V8, V10)
                int wi = d_to_w[ei][d];
                if (wi > 0 && d == g_date_to_idx[etfs[ei].daily[etfs[ei].weekly_bars[wi].last_daily_idx].date]) {
                    if (cfg.w_sell_mode == 0) {
                        // Death cross
                        sell = (w_dif[ei][wi] < w_dea[ei][wi] && w_dif[ei][wi-1] >= w_dea[ei][wi-1]);
                    } else if (cfg.w_sell_mode == 1) {
                        // DIF < 0
                        sell = (w_dif[ei][wi] < 0);
                    } else if (cfg.w_sell_mode == 2) {
                        // Histogram < 0 crossover (crosses from positive to negative)
                        sell = (w_hist[ei][wi] < 0 && w_hist[ei][wi-1] >= 0);
                    } else if (cfg.w_sell_mode == 3) {
                        // Histogram < 0 level (just negative)
                        sell = (w_hist[ei][wi] < 0);
                    }
                }
            }
            
            // Trailing stop (daily)
            if (!sell && cfg.trailing_stop > 0 && cur_close > 0) {
                if (cur_close <= pos.highest_price * (1.0 - cfg.trailing_stop / 100.0)) sell = true;
            }
            // Take profit (daily)
            if (!sell && cfg.take_profit > 0 && cur_close > 0) {
                if (cur_close >= pos.entry_price * (1.0 + cfg.take_profit / 100.0)) sell = true;
            }
            // Force close on last day
            if (d == end_g_idx) sell = true;
            
            if (sell) {
                // Execute at next day's open if available, else today's close
                double exit_price = cur_close;
                if (d < end_g_idx) {
                    double nxt_open = etfs[ei].global_open[d + 1];
                    if (nxt_open > 0) exit_price = nxt_open;
                }
                if (exit_price > 0) {
                    cash += pos.shares * exit_price;
                    trades++;
                    if (exit_price > pos.entry_price) wins++;
                }
            } else {
                remaining.push_back(pos);
            }
        }
        positions = remaining;
        
        // --- 2. Sector momentum for V8 ---
        set<int> top_sectors;
        if (cfg.use_sector && d >= start_g_idx + sector_lookback) {
            unordered_map<int, double> sector_mom;
            unordered_map<int, int> sector_cnt;
            for (int i = 0; i < num_etfs; ++i) {
                if (etfs[i].sector_id == -1) continue;
                double p_old = etfs[i].global_close[d - sector_lookback];
                double p_now = etfs[i].global_close[d];
                if (p_old > 0 && p_now > 0) {
                    sector_mom[etfs[i].sector_id] += (p_now - p_old) / p_old;
                    sector_cnt[etfs[i].sector_id]++;
                }
            }
            vector<pair<int, double>> avg_mom;
            for (auto& sm : sector_mom) {
                if (sector_cnt[sm.first] > 0) avg_mom.push_back({sm.first, sm.second / sector_cnt[sm.first]});
            }
            sort(avg_mom.begin(), avg_mom.end(), [](auto& a, auto& b){ return a.second > b.second; });
            for (int i = 0; i < min(sector_topK, (int)avg_mom.size()); ++i) top_sectors.insert(avg_mom[i].first);
        }
        
        // --- 3. Record equity BEFORE buys (so buys take effect next day) ---
        double daily_equity = cash;
        for (const auto& pos : positions) {
            double p = etfs[pos.etf_idx].global_close[d];
            if (p <= 0) p = pos.entry_price;
            daily_equity += pos.shares * p;
        }
        
        // --- 4. BUYS ---
        int max_pos = cfg.max_positions;
        if (max_pos == 0) max_pos = 9999; // effectively unlimited
        
        if ((int)positions.size() < max_pos) {
            for (int i = 0; i < num_etfs && (int)positions.size() < max_pos; ++i) {
                // Skip if already holding this ETF
                bool already = false;
                for (const auto& pos : positions) { if (pos.etf_idx == i) { already = true; break; } }
                if (already) continue;
                
                // Sector filter for V8
                if (cfg.use_sector && (etfs[i].sector_id == -1 || top_sectors.find(etfs[i].sector_id) == top_sectors.end())) continue;
                
                bool buy_signal = false;
                
                if (cfg.monthly_only) {
                    // V4: Buy at end of month bar  
                    int mi = d_to_m[i][d];
                    if (mi > 0 && d == g_date_to_idx[etfs[i].daily[etfs[i].monthly_bars[mi].last_daily_idx].date]) {
                        if (cfg.m_buy_mode == 0) {
                            // Golden cross
                            buy_signal = (m_dif[i][mi] > m_dea[i][mi] && m_dif[i][mi-1] <= m_dea[i][mi-1]);
                        } else if (cfg.m_buy_mode == 1) {
                            // Histogram > 0 crossover
                            buy_signal = (m_hist[i][mi] > 0 && m_hist[i][mi-1] <= 0);
                        }
                    }
                } else if (cfg.weekly_only) {
                    // V5: Buy at end of week bar
                    int wi = d_to_w[i][d];
                    if (wi > 0 && d == g_date_to_idx[etfs[i].daily[etfs[i].weekly_bars[wi].last_daily_idx].date]) {
                        if (cfg.w_buy_mode == 0) {
                            buy_signal = (w_dif[i][wi] > w_dea[i][wi] && w_dif[i][wi-1] <= w_dea[i][wi-1]);
                        } else if (cfg.w_buy_mode == 1) {
                            buy_signal = (w_hist[i][wi] > 0 && w_hist[i][wi-1] <= 0);
                        }
                    }
                } else {
                    // Dual-TF: Weekly buy signal with monthly filter (V7, V8, V10)
                    int wi = d_to_w[i][d];
                    if (wi > 0 && d == g_date_to_idx[etfs[i].daily[etfs[i].weekly_bars[wi].last_daily_idx].date]) {
                        int mi = w_to_m[i][wi];
                        if (mi > 0) {
                            // Monthly trend filter
                            bool m_bull = false;
                            if (cfg.m_trend_mode == 0) m_bull = m_dif[i][mi] > 0;
                            else if (cfg.m_trend_mode == 1) m_bull = m_dif[i][mi] > m_dea[i][mi];
                            else if (cfg.m_trend_mode == 2) m_bull = m_hist[i][mi] > 0;
                            else if (cfg.m_trend_mode == 3) m_bull = m_dif[i][mi] > m_dif[i][mi-1]; // DIF rising
                            
                            // Weekly buy signal
                            bool w_buy = false;
                            if (cfg.w_buy_mode == 0) {
                                w_buy = (w_dif[i][wi] > w_dea[i][wi] && w_dif[i][wi-1] <= w_dea[i][wi-1]);
                            } else if (cfg.w_buy_mode == 1) {
                                w_buy = (w_hist[i][wi] > 0 && w_hist[i][wi-1] <= 0);
                            }
                            
                            if (m_bull && w_buy) buy_signal = true;
                        }
                    }
                }
                
                if (buy_signal) {
                    // Execute at next day's open
                    double entry_price = 0;
                    if (d < end_g_idx) {
                        entry_price = etfs[i].global_open[d + 1];
                        if (entry_price <= 0) entry_price = etfs[i].global_close[d + 1];
                    }
                    if (entry_price <= 0) entry_price = etfs[i].global_close[d];
                    if (entry_price <= 0) continue;
                    
                    double alloc;
                    if (cfg.max_positions > 0) {
                        alloc = daily_equity / cfg.max_positions;
                    } else {
                        // Unlimited: allocate equal weight based on current positions + 1
                        alloc = daily_equity / ((int)positions.size() + 1);
                    }
                    if (alloc > cash) alloc = cash;
                    if (alloc < 1000) continue; // minimum allocation
                    
                    double shares = alloc / entry_price;
                    cash -= shares * entry_price;
                    
                    string entry_date = (d + 1 < (int)global_dates.size()) ? global_dates[d + 1] : global_dates[d];
                    positions.push_back({i, entry_price, entry_price, entry_date, shares});
                }
            }
        }
        
        // --- 5. Record equity ---
        double eq = cash;
        for (const auto& pos : positions) {
            double p = etfs[pos.etf_idx].global_close[d];
            if (p <= 0) p = pos.entry_price;
            eq += pos.shares * p;
        }
        equity_curve.push_back(eq);
    }
    
    return calc_metrics(cfg.name, equity_curve, trades, wins, end_g_idx - start_g_idx + 1);
}

// ============================================================
// Buy-and-Hold Baseline
// ============================================================

BacktestResult run_baseline(vector<ETFData>& etfs, int start_g_idx, int end_g_idx) {
    double cash = 1000000;
    double alloc_per = cash / etfs.size();
    
    struct BHPos { int idx; double shares; double entry; };
    vector<BHPos> bh_pos;
    
    for (int i = 0; i < (int)etfs.size(); ++i) {
        double price = etfs[i].global_open[start_g_idx];
        if (price <= 0) price = etfs[i].global_close[start_g_idx];
        if (price > 0) {
            double s = alloc_per / price;
            cash -= s * price;
            bh_pos.push_back({i, s, price});
        }
    }
    
    vector<double> equity_curve;
    for (int d = start_g_idx; d <= end_g_idx; ++d) {
        double eq = cash;
        for (const auto& p : bh_pos) {
            double c = etfs[p.idx].global_close[d];
            if (c <= 0) c = p.entry;
            eq += p.shares * c;
        }
        equity_curve.push_back(eq);
    }
    
    return calc_metrics("Buy-and-Hold", equity_curve, 0, 0, end_g_idx - start_g_idx + 1);
}

// ============================================================
// Main
// ============================================================

int main() {
    init_sectors();
    string data_dir = "/ceph/dang_articles/yoj/market_data/";
    vector<ETFData> etfs;
    
    // Load ETFs
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        string path = entry.path().string();
        if (path.find(".csv") == string::npos) continue;
        
        string code = entry.path().stem().string();
        if (code.find("sh000") == 0 || code.find("sz399") == 0) continue; // skip indices
        
        ifstream file(path);
        string line;
        getline(file, line); // header
        
        ETFData etf;
        etf.code = code;
        etf.name = g_etf_names.count(code) ? g_etf_names[code] : code;
        if (sector_map.count(code)) etf.sector_id = sector_map[code];
        
        while (getline(file, line)) {
            stringstream ss(line);
            string d, o, h, l, c, v;
            getline(ss, d, ',');
            getline(ss, o, ',');
            getline(ss, h, ',');
            getline(ss, l, ',');
            getline(ss, c, ',');
            getline(ss, v, ',');
            try {
                etf.daily.push_back({d, stod(o), stod(h), stod(l), stod(c), stod(v)});
            } catch (...) {}
        }
        
        if (etf.daily.size() < 60) continue;
        etf.split_idx = etf.daily.size() * 0.4;
        
        // Full-data aggregations (for MACD warmup)
        etf.weekly_bars = aggregate_weekly(etf.daily, 0, etf.daily.size());
        etf.monthly_bars = aggregate_monthly(etf.daily, 0, etf.daily.size());
        
        etfs.push_back(etf);
    }
    
    cout << "Loaded " << etfs.size() << " tradeable ETFs.\n";
    
    // Build global date index
    set<string> all_dates_set;
    for (const auto& etf : etfs) {
        for (const auto& b : etf.daily) all_dates_set.insert(b.date);
    }
    global_dates.assign(all_dates_set.begin(), all_dates_set.end());
    for (int i = 0; i < (int)global_dates.size(); ++i) {
        g_date_to_idx[global_dates[i]] = i;
    }
    
    // Find test period start (minimum split_idx date across all ETFs)
    int start_g_idx = global_dates.size();
    for (const auto& etf : etfs) {
        if (etf.split_idx < (int)etf.daily.size()) {
            start_g_idx = min(start_g_idx, g_date_to_idx[etf.daily[etf.split_idx].date]);
        }
    }
    int end_g_idx = global_dates.size() - 1;
    
    cout << "Test period: " << global_dates[start_g_idx] << " to " << global_dates[end_g_idx] << "\n";
    cout << "Test days: " << (end_g_idx - start_g_idx + 1) << "\n\n";
    
    // Build global price arrays with forward-fill for close
    for (auto& etf : etfs) {
        etf.global_close.assign(global_dates.size(), 0.0);
        etf.global_open.assign(global_dates.size(), 0.0);
        etf.global_high.assign(global_dates.size(), 0.0);
        for (const auto& b : etf.daily) {
            int gidx = g_date_to_idx[b.date];
            etf.global_close[gidx] = b.close;
            etf.global_open[gidx] = b.open;
            etf.global_high[gidx] = b.high;
        }
        // Forward-fill close only (for equity calculation)
        for (int i = 1; i < (int)global_dates.size(); ++i) {
            if (etf.global_close[i] == 0) etf.global_close[i] = etf.global_close[i-1];
        }
    }
    
    // ============================================================
    // Define strategy configs
    // ============================================================
    // Reference params from original optimizers:
    // V4:  Monthly (10,24,3), buy=Hist>0, sell=DIF<0
    // V5:  Weekly (11,15,3), buy=Hist>0, sell=Hist<0 crossover, trail=20%, TP=50%
    // V7:  Monthly(8,17,3) DIF↑ + Weekly(8,30,3) GC, sell=Hist<0 level, TP=30%
    // V8:  Monthly(10,17,5) DIF↑ + Weekly(8,30,3) GC, sell=Hist<0 level, trail=20%, TP=30%, sector
    // V10-Sharpe: Monthly(6,15,5) DIF↑ + Weekly(6,25,3) GC, sell=DC, trail=20%, TP=20%, maxPos=5
    // V10-Calmar: Monthly(8,17,5) DIF↑ + Weekly(10,40,3) GC, sell=DC, trail=20%, TP=30%, maxPos=5
    
    vector<StrategyConfig> configs = {
        // V4-Monthly: monthly-only, histogram>0 buy, DIF<0 sell
        {"V4-Monthly", 10, 24, 3,  12, 26, 9,  true, false, false,  1/*hist>0*/, 0, 0,  1/*DIF<0*/, 0,  0, 0, 0},
        
        // V5-Weekly: weekly-only, histogram>0 buy, histogram<0 crossover sell, trail=20%, TP=50%
        {"V5-Weekly", 12, 26, 9,  11, 15, 3,  false, true, false,  0, 0, 1/*hist>0*/,  0, 2/*hist<0 cross*/,  20.0, 50.0, 0},
        
        // V7-DualTF: monthly DIF↑ filter + weekly GC buy, weekly hist<0 level sell, TP=30%
        {"V7-DualTF", 8, 17, 3,  8, 30, 3,  false, false, false,  0, 3/*DIF↑*/, 0/*GC*/,  0, 3/*hist<0 level*/,  0, 30.0, 0},
        
        // V8-Sector: same as V7 but with sector filter + trail=20%
        {"V8-Sector", 10, 17, 5,  8, 30, 3,  false, false, true,  0, 3/*DIF↑*/, 0/*GC*/,  0, 3/*hist<0 level*/,  20.0, 30.0, 0},
        
        // V10-Sharpe: monthly DIF↑ + weekly GC, weekly DC sell, trail=20%, TP=20%, max=5
        {"V10-Sharpe", 6, 15, 5,  6, 25, 3,  false, false, false,  0, 3/*DIF↑*/, 0/*GC*/,  0, 0/*DC*/,  20.0, 20.0, 5},
        
        // V10-Calmar: monthly DIF↑ + weekly GC, weekly DC sell, trail=20%, TP=30%, max=5
        {"V10-Calmar", 8, 17, 5,  10, 40, 3,  false, false, false,  0, 3/*DIF↑*/, 0/*GC*/,  0, 0/*DC*/,  20.0, 30.0, 5},
    };
    
    vector<BacktestResult> results;
    
    // Buy-and-Hold baseline
    cout << "Running Buy-and-Hold Baseline...\n";
    results.push_back(run_baseline(etfs, start_g_idx, end_g_idx));
    
    // Run each strategy
    for (auto& cfg : configs) {
        cout << "Running " << cfg.name << "...\n";
        results.push_back(run_strategy(cfg, etfs, start_g_idx, end_g_idx));
    }
    
    // ============================================================
    // Output results
    // ============================================================
    
    cout << "\n========== ETF MACD Strategy Comparison (Portfolio Backtest) ==========\n";
    cout << "Test period: " << global_dates[start_g_idx] << " ~ " << global_dates[end_g_idx] << "\n";
    cout << "Starting capital: $1,000,000\n\n";
    
    cout << left << setw(15) << "Strategy" << " | "
         << setw(10) << "Ann.Return" << " | "
         << setw(11) << "MaxDrawdown" << " | "
         << setw(6) << "Sharpe" << " | "
         << setw(6) << "Trades" << " | "
         << "WinRate\n";
    cout << string(68, '-') << "\n";
    
    for (const auto& r : results) {
        cout << left << setw(15) << r.version << " | "
             << right << fixed << setprecision(2) << setw(9) << r.ann_return << "% | "
             << setw(10) << r.mdd << "% | "
             << setw(6) << setprecision(2) << r.sharpe << " | "
             << setw(6) << r.total_trades << " | "
             << setw(6) << setprecision(1) << r.win_rate << "%\n";
    }
    
    // Save CSV
    ofstream out("etf_macd_comparison_results.csv");
    out << "Version,AnnReturn(%),MaxDrawdown(%),Sharpe,Trades,WinRate(%)\n";
    for (const auto& r : results) {
        out << r.version << ","
            << fixed << setprecision(2) << r.ann_return << ","
            << r.mdd << ","
            << r.sharpe << ","
            << r.total_trades << ","
            << setprecision(1) << r.win_rate << "\n";
    }
    out.close();
    cout << "\nResults saved to etf_macd_comparison_results.csv\n";
    
    return 0;
}
