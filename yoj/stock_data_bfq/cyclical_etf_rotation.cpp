/**
 * 周期性行业ETF轮动策略 (Cyclical Sector ETF Rotation Strategy)
 * 
 * 核心思路：利用月线MACD检测行业周期底部，通过ETF轮动捕获行业轮动收益。
 * 当一个行业的周期见底（月线MACD金叉 / DIF上穿）时买入该行业ETF，
 * 当周期见顶（月线MACD死叉 / DIF下穿）时卖出，轮动到下一个周期见底的行业。
 * 
 * 基于 V10 MACD优化器架构，针对周期行业ETF池做了定制化设计。
 * 
 * ETF池 (14只周期性行业ETF):
 *   核心6个: 军工(sh512660), 煤炭(sh515220), 游戏(sh516780), 
 *            旅游(sh562510), 化工(sz159870), 电力(sz159611)
 *   扩展8个: 有色金属(sh512400), 钢铁(sh515210), 基建(sh516950),
 *            能源化工(sz159981), 稀有金属(sh515650), 稀土(sh516150),
 *            新能源(sh516160), 光伏(sh515790)
 * 
 * 编译: g++ -O3 -std=c++17 -o cyclical_etf_rotation cyclical_etf_rotation.cpp -lpthread
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <filesystem>
#include <iomanip>
#include <atomic>
#include <numeric>
#include <set>

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Data Structures
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
    vector<DailyBar> daily;
    vector<AggBar> monthly;
    vector<AggBar> weekly;
    vector<double> global_close;
    vector<double> global_open;
    vector<double> global_high;
};

struct StrategyParams {
    // Monthly MACD params (trend filter)
    int m_fast, m_slow, m_signal;
    int buy_signal;    // 0=golden cross (DIF>DEA & prev DIF<=DEA), 1=DIF turning up, 2=MACD hist > 0
    int sell_signal;   // 0=death cross (DIF<DEA & prev DIF>=DEA), 1=DIF turning down, 2=DIF < 0
    
    // Weekly MACD params (entry timing)  
    int w_fast, w_slow, w_signal;
    int w_confirm;     // 0=no weekly confirm needed, 1=weekly golden cross, 2=weekly DIF>0, 3=weekly DIF rising
    
    // Risk management
    double trailing_stop;
    double take_profit;
    
    // Portfolio
    int max_positions;
};

struct Trade {
    string etf_code, etf_name;
    string entry_date, exit_date;
    double entry_price, exit_price;
    double pnl_pct;
    int hold_days;
};

struct BacktestResult {
    double sharpe = -100;
    double calmar = -100;
    double combined_score = -100;
    double mdd = 0;
    double annualized = 0;
    int trades = 0;
    double win_rate = 0;
    double avg_hold_days = 0;
    vector<Trade> trade_list;
    vector<double> equity_curve;
};

struct Position {
    int etf_idx;
    double entry_price;
    double highest_price;
    string entry_date;
    double shares;
    int entry_daily_idx;
};

// ============================================================
// Globals
// ============================================================

vector<string> global_dates;
unordered_map<string, int> g_date_to_idx;
const double RISK_FREE_RATE = 0.02;

// Only include cyclical sector ETFs
const vector<pair<string, string>> CYCLICAL_ETFS = {
    // Core 6 (user specified)
    {"sh512660", "军工ETF"},
    {"sh515220", "煤炭ETF"},
    {"sh516780", "游戏ETF"},
    {"sh562510", "旅游ETF"},
    {"sz159870", "化工ETF"},
    {"sz159611", "电力ETF"},
    // Expansion (highly cyclical industries)
    {"sh512400", "有色金属ETF"},
    {"sh515210", "钢铁ETF"},
    {"sh516950", "基建ETF"},
    {"sz159981", "能源化工ETF"},
    {"sh515650", "稀有金属ETF"},
    {"sh516150", "稀土ETF"},
    {"sh516160", "新能源ETF"},
    {"sh515790", "光伏ETF"},
};

// ============================================================
// Bar Aggregation
// ============================================================

vector<AggBar> aggregate_monthly(const vector<DailyBar>& daily, int start, int end_idx) {
    vector<AggBar> result;
    if (start >= end_idx) return result;
    
    string cur_month = daily[start].date.substr(0, 7);
    AggBar bar = {daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    
    for (int i = start + 1; i < end_idx; ++i) {
        string month = daily[i].date.substr(0, 7);
        if (month != cur_month) {
            result.push_back(bar);
            cur_month = month;
            bar = {daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i};
        } else {
            bar.close = daily[i].close;
            bar.high = max(bar.high, daily[i].high);
            bar.low = min(bar.low, daily[i].low);
            bar.volume += daily[i].volume;
            bar.last_daily_idx = i;
        }
    }
    result.push_back(bar);
    return result;
}

vector<AggBar> aggregate_weekly(const vector<DailyBar>& daily, int start, int end_idx) {
    vector<AggBar> result;
    if (start >= end_idx) return result;
    
    auto get_weekday = [](const string& date) -> int {
        int y = stoi(date.substr(0, 4));
        int m = stoi(date.substr(5, 2));
        int d = stoi(date.substr(8, 2));
        if (m <= 2) { y--; m += 12; }
        return (d + 13*(m+1)/5 + y + y/4 - y/100 + y/400) % 7; // 0=Sat
    };
    
    AggBar bar = {daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    int prev_wd = get_weekday(daily[start].date);
    
    for (int i = start + 1; i < end_idx; ++i) {
        int cur_wd = get_weekday(daily[i].date);
        if (cur_wd <= prev_wd) {
            result.push_back(bar);
            bar = {daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i};
        } else {
            bar.close = daily[i].close;
            bar.high = max(bar.high, daily[i].high);
            bar.low = min(bar.low, daily[i].low);
            bar.volume += daily[i].volume;
            bar.last_daily_idx = i;
        }
        prev_wd = cur_wd;
    }
    result.push_back(bar);
    return result;
}

// ============================================================
// MACD Calculation
// ============================================================

void compute_macd(const vector<AggBar>& bars, int fast, int slow, int signal,
                  vector<double>& dif, vector<double>& dea, vector<double>& hist) {
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

// ============================================================
// Portfolio Backtest
// ============================================================

BacktestResult run_backtest(const vector<ETFData>& etfs, const StrategyParams& params) {
    int num_etfs = etfs.size();
    
    // Precompute MACD for all ETFs
    vector<vector<double>> m_dif(num_etfs), m_dea(num_etfs), m_hist(num_etfs);
    vector<vector<double>> w_dif(num_etfs), w_dea(num_etfs), w_hist(num_etfs);
    vector<vector<int>> w_to_m(num_etfs);  // weekly bar -> corresponding monthly bar
    vector<vector<int>> d_to_w(num_etfs);  // global date -> weekly bar idx
    vector<vector<int>> d_to_m(num_etfs);  // global date -> monthly bar idx
    
    int min_warmup = max(params.m_slow, params.w_slow) + 2;
    
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].monthly.size() < 5 || etfs[i].weekly.size() < 10) continue;
        
        compute_macd(etfs[i].monthly, params.m_fast, params.m_slow, params.m_signal, m_dif[i], m_dea[i], m_hist[i]);
        compute_macd(etfs[i].weekly, params.w_fast, params.w_slow, params.w_signal, w_dif[i], w_dea[i], w_hist[i]);
        
        // Map weekly -> monthly
        w_to_m[i].assign(etfs[i].weekly.size(), -1);
        int m_idx = 0;
        for (size_t w = 0; w < etfs[i].weekly.size(); ++w) {
            while (m_idx + 1 < (int)etfs[i].monthly.size() && 
                   etfs[i].monthly[m_idx].last_daily_idx < etfs[i].weekly[w].first_daily_idx)
                m_idx++;
            if (m_idx < (int)etfs[i].monthly.size() &&
                etfs[i].monthly[m_idx].first_daily_idx <= etfs[i].weekly[w].last_daily_idx)
                w_to_m[i][w] = m_idx;
        }
        
        // Map daily -> weekly
        d_to_w[i].assign(global_dates.size(), -1);
        for (size_t w = 0; w < etfs[i].weekly.size(); ++w) {
            for (int d = etfs[i].weekly[w].first_daily_idx; d <= etfs[i].weekly[w].last_daily_idx; ++d) {
                if (d < (int)etfs[i].daily.size()) {
                    auto it = g_date_to_idx.find(etfs[i].daily[d].date);
                    if (it != g_date_to_idx.end())
                        d_to_w[i][it->second] = w;
                }
            }
        }
        
        // Map daily -> monthly
        d_to_m[i].assign(global_dates.size(), -1);
        for (size_t m = 0; m < etfs[i].monthly.size(); ++m) {
            for (int d = etfs[i].monthly[m].first_daily_idx; d <= etfs[i].monthly[m].last_daily_idx; ++d) {
                if (d < (int)etfs[i].daily.size()) {
                    auto it = g_date_to_idx.find(etfs[i].daily[d].date);
                    if (it != g_date_to_idx.end())
                        d_to_m[i][it->second] = m;
                }
            }
        }
    }
    
    double capital = 1000000.0;
    double cash = capital;
    vector<Position> active_positions;
    BacktestResult res;
    
    // Find the common date range
    int start_idx = (int)global_dates.size(), end_idx = 0;
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].daily.empty()) continue;
        auto it_s = g_date_to_idx.find(etfs[i].daily.front().date);
        auto it_e = g_date_to_idx.find(etfs[i].daily.back().date);
        if (it_s != g_date_to_idx.end()) start_idx = min(start_idx, it_s->second);
        if (it_e != g_date_to_idx.end()) end_idx = max(end_idx, it_e->second);
    }
    
    if (start_idx >= end_idx) return res;
    
    res.equity_curve.reserve(end_idx - start_idx + 1);
    
    for (int d = start_idx; d <= end_idx; ++d) {
        string cur_date = global_dates[d];
        
        // ============ SELL LOGIC ============
        vector<Position> remaining_positions;
        for (auto& pos : active_positions) {
            int e_idx = pos.etf_idx;
            const auto& etf = etfs[e_idx];
            
            double cur_close = etf.global_close[d];
            double cur_high = etf.global_high[d];
            if (cur_close <= 0) {
                remaining_positions.push_back(pos);
                continue;
            }
            
            if (cur_high > pos.highest_price) pos.highest_price = cur_high;
            
            bool sell = false;
            
            // Monthly MACD sell signal — check at month end
            int mi = d_to_m[e_idx][d];
            if (mi > 0 && mi < (int)m_dif[e_idx].size()) {
                // Check on last trading day of the month
                if (d == g_date_to_idx[etf.daily[etf.monthly[mi].last_daily_idx].date]) {
                    if (params.sell_signal == 0) {
                        sell = (m_dif[e_idx][mi] < m_dea[e_idx][mi] && m_dif[e_idx][mi-1] >= m_dea[e_idx][mi-1]);
                    } else if (params.sell_signal == 1) {
                        sell = (m_dif[e_idx][mi] < m_dif[e_idx][mi-1]);
                    } else if (params.sell_signal == 2) {
                        sell = (m_dif[e_idx][mi] < 0);
                    }
                }
            }
            
            // Weekly MACD sell — additional weekly check for faster exit
            int wi = d_to_w[e_idx][d];
            if (!sell && wi > 0 && wi < (int)w_dif[e_idx].size()) {
                if (d == g_date_to_idx[etf.daily[etf.weekly[wi].last_daily_idx].date]) {
                    // Weekly death cross as a supplementary sell
                    if (w_dif[e_idx][wi] < w_dea[e_idx][wi] && w_dif[e_idx][wi-1] >= w_dea[e_idx][wi-1]) {
                        // Only sell on weekly death cross if monthly trend also turning
                        if (mi > 0 && mi < (int)m_dif[e_idx].size() && m_dif[e_idx][mi] < m_dif[e_idx][mi > 0 ? mi-1 : mi]) {
                            sell = true;
                        }
                    }
                }
            }
            
            // Trailing stop
            if (!sell && params.trailing_stop > 0) {
                if (cur_close <= pos.highest_price * (1.0 - params.trailing_stop / 100.0)) sell = true;
            }
            // Take profit
            if (!sell && params.take_profit > 0) {
                if (cur_close >= pos.entry_price * (1.0 + params.take_profit / 100.0)) sell = true;
            }
            
            if (sell || d == end_idx) {
                double exit_price = cur_close;
                if (d < end_idx && sell) {
                    double nxt_open = etf.global_open[d + 1];
                    if (nxt_open > 0) exit_price = nxt_open;
                }
                
                double value = pos.shares * exit_price;
                cash += value;
                
                Trade t;
                t.etf_code = etf.code;
                t.etf_name = etf.name;
                t.entry_date = pos.entry_date;
                t.exit_date = cur_date;
                t.entry_price = pos.entry_price;
                t.exit_price = exit_price;
                t.pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0;
                t.hold_days = d - pos.entry_daily_idx;
                res.trade_list.push_back(t);
                res.trades++;
            } else {
                remaining_positions.push_back(pos);
            }
        }
        active_positions = remaining_positions;
        
        // Record equity
        double daily_equity = cash;
        for (const auto& pos : active_positions) {
            double p = etfs[pos.etf_idx].global_close[d];
            if (p <= 0) p = pos.entry_price;
            daily_equity += pos.shares * p;
        }
        res.equity_curve.push_back(daily_equity);
        
        // ============ BUY LOGIC ============
        if ((int)active_positions.size() < params.max_positions) {
            // Collect buy candidates with their signal strength
            struct BuyCandidate {
                int etf_idx;
                double strength; // signal strength for ranking
            };
            vector<BuyCandidate> candidates;
            
            for (int i = 0; i < num_etfs; ++i) {
                if (m_dif[i].empty() || w_dif[i].empty()) continue;
                
                int mi = d_to_m[i][d];
                int wi = d_to_w[i][d];
                if (mi < 2 || wi < 2) continue;
                if (mi >= (int)m_dif[i].size() || wi >= (int)w_dif[i].size()) continue;
                
                // Only check at end of week for entry timing
                const auto& etf = etfs[i];
                if (d != g_date_to_idx[etf.daily[etf.weekly[wi].last_daily_idx].date]) continue;
                
                // Already holding this ETF?
                bool already_held = false;
                for (const auto& pos : active_positions) {
                    if (pos.etf_idx == i) { already_held = true; break; }
                }
                if (already_held) continue;
                
                // Monthly MACD buy signal
                bool m_buy = false;
                if (params.buy_signal == 0) {
                    m_buy = (m_dif[i][mi] > m_dea[i][mi] && m_dif[i][mi-1] <= m_dea[i][mi-1]);
                } else if (params.buy_signal == 1) {
                    m_buy = (m_dif[i][mi] > m_dif[i][mi-1]); // DIF turning up
                } else if (params.buy_signal == 2) {
                    m_buy = (m_hist[i][mi] > 0 && m_hist[i][mi-1] <= 0); // MACD hist turn positive
                } else if (params.buy_signal == 3) {
                    // Underwater golden cross: DIF<0 and DIF crosses above DEA
                    m_buy = (m_dif[i][mi] < 0 && m_dif[i][mi] > m_dea[i][mi] && m_dif[i][mi-1] <= m_dea[i][mi-1]);
                } else if (params.buy_signal == 4) {
                    // DIF > DEA (already above, trend confirmation)
                    m_buy = (m_dif[i][mi] > m_dea[i][mi]);
                }
                
                if (!m_buy) continue;
                
                // Weekly MACD confirmation
                bool w_ok = true;
                if (params.w_confirm == 1) {
                    w_ok = (w_dif[i][wi] > w_dea[i][wi] && w_dif[i][wi-1] <= w_dea[i][wi-1]);
                } else if (params.w_confirm == 2) {
                    w_ok = (w_dif[i][wi] > 0);
                } else if (params.w_confirm == 3) {
                    w_ok = (w_dif[i][wi] > w_dif[i][wi-1]); // DIF rising
                } else if (params.w_confirm == 4) {
                    w_ok = (w_dif[i][wi] > w_dea[i][wi]); // Weekly DIF > DEA
                }
                
                if (!w_ok) continue;
                
                // Signal strength: use monthly DIF momentum for ranking
                double strength = m_dif[i][mi] - m_dif[i][mi-1]; // DIF acceleration
                candidates.push_back({i, strength});
            }
            
            // Sort by signal strength (strongest cycle bottom signal first)
            sort(candidates.begin(), candidates.end(), [](const BuyCandidate& a, const BuyCandidate& b) {
                return a.strength > b.strength;
            });
            
            // Buy the top candidates
            for (const auto& cand : candidates) {
                if ((int)active_positions.size() >= params.max_positions) break;
                
                int i = cand.etf_idx;
                const auto& etf = etfs[i];
                double entry_price = etf.global_close[d];
                if (d < end_idx) {
                    double nxt_open = etf.global_open[d + 1];
                    if (nxt_open > 0) entry_price = nxt_open;
                }
                if (entry_price <= 0) continue;
                
                double alloc = daily_equity / params.max_positions;
                if (alloc > cash) alloc = cash;
                if (alloc < 100) continue; // too small
                
                double shares = alloc / entry_price;
                cash -= shares * entry_price;
                
                Position p;
                p.etf_idx = i;
                p.entry_price = entry_price;
                p.highest_price = entry_price;
                p.entry_date = (d < end_idx) ? global_dates[d + 1] : cur_date;
                p.shares = shares;
                p.entry_daily_idx = d;
                active_positions.push_back(p);
            }
        }
    }
    
    // ============ Metrics ============
    double final_equity = res.equity_curve.empty() ? capital : res.equity_curve.back();
    double yrs = (double)res.equity_curve.size() / 252.0;
    
    if (res.equity_curve.size() > 1 && res.trades > 0) {
        double max_eq = res.equity_curve[0];
        double mdd = 0;
        double sum_ret = 0;
        vector<double> daily_rets(res.equity_curve.size() - 1);
        
        for (size_t i = 1; i < res.equity_curve.size(); ++i) {
            double r = (res.equity_curve[i] - res.equity_curve[i - 1]) / res.equity_curve[i - 1];
            daily_rets[i - 1] = r;
            sum_ret += r;
            if (res.equity_curve[i] > max_eq) max_eq = res.equity_curve[i];
            double dd = (max_eq - res.equity_curve[i]) / max_eq;
            if (dd > mdd) mdd = dd;
        }
        
        double mean_ret = sum_ret / daily_rets.size();
        double var = 0;
        for (double r : daily_rets) var += (r - mean_ret) * (r - mean_ret);
        var /= daily_rets.size();
        double std_ret = sqrt(var);
        
        double ann_ret = pow(final_equity / capital, 1.0 / yrs) - 1.0;
        double ann_vol = std_ret * sqrt(252.0);
        
        res.sharpe = ann_vol > 0 ? (ann_ret - RISK_FREE_RATE) / ann_vol : 0;
        res.mdd = mdd;
        res.calmar = mdd > 0 ? ann_ret / mdd : 0;
        if (mdd == 0 && ann_ret > 0) res.calmar = 100.0;
        res.annualized = ann_ret * 100.0;
        res.combined_score = 0.5 * res.sharpe + 0.5 * min(res.calmar, 10.0);
        
        // Win rate
        int wins = 0;
        double total_hold = 0;
        for (const auto& t : res.trade_list) {
            if (t.pnl_pct > 0) wins++;
            total_hold += t.hold_days;
        }
        res.win_rate = res.trades > 0 ? (double)wins / res.trades * 100.0 : 0;
        res.avg_hold_days = res.trades > 0 ? total_hold / res.trades : 0;
    }
    
    return res;
}

// ============================================================
// Main
// ============================================================

int main() {
    string data_dir = "/ceph/dang_articles/yoj/market_data/";
    
    // Build allowed set
    unordered_map<string, string> etf_map;
    for (const auto& [code, name] : CYCLICAL_ETFS) {
        etf_map[code] = name;
    }
    
    // Load data
    vector<ETFData> etfs;
    for (const auto& [code, name] : CYCLICAL_ETFS) {
        string path = data_dir + code + ".csv";
        ifstream file(path);
        if (!file.is_open()) {
            cerr << "  [SKIP] " << name << " (" << code << "): file not found\n";
            continue;
        }
        
        string line;
        getline(file, line); // header
        
        ETFData etf;
        etf.code = code;
        etf.name = name;
        
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
        
        if (etf.daily.size() < 60) {
            cerr << "  [SKIP] " << name << " (" << code << "): only " << etf.daily.size() << " bars\n";
            continue;
        }
        
        // Aggregate all data (no train/test split — use full period)
        etf.monthly = aggregate_monthly(etf.daily, 0, etf.daily.size());
        etf.weekly = aggregate_weekly(etf.daily, 0, etf.daily.size());
        
        cout << "  [OK] " << name << " (" << code << "): " << etf.daily.size() << " daily, "
             << etf.monthly.size() << " monthly, " << etf.weekly.size() << " weekly  ("
             << etf.daily.front().date << " ~ " << etf.daily.back().date << ")\n";
        
        etfs.push_back(etf);
    }
    
    cout << "\n已加载 " << etfs.size() << " 只周期行业ETF\n\n";
    
    // Build global dates
    set<string> all_dates_set;
    for (const auto& etf : etfs) {
        for (const auto& b : etf.daily) all_dates_set.insert(b.date);
    }
    global_dates.assign(all_dates_set.begin(), all_dates_set.end());
    for (int i = 0; i < (int)global_dates.size(); ++i) {
        g_date_to_idx[global_dates[i]] = i;
    }
    
    // Build global price arrays
    for (auto& etf : etfs) {
        etf.global_close.assign(global_dates.size(), 0.0);
        etf.global_open.assign(global_dates.size(), 0.0);
        etf.global_high.assign(global_dates.size(), 0.0);
        
        double last_close = 0.0;
        for (const auto& b : etf.daily) {
            int d_idx = g_date_to_idx[b.date];
            etf.global_close[d_idx] = b.close;
            etf.global_open[d_idx] = b.open;
            etf.global_high[d_idx] = b.high;
        }
        // Forward fill close prices
        for (int i = 0; i < (int)global_dates.size(); ++i) {
            if (etf.global_close[i] > 0) {
                last_close = etf.global_close[i];
            } else {
                etf.global_close[i] = last_close;
            }
        }
    }
    
    // ============================================================
    // Parameter Grid Search
    // ============================================================
    
    vector<StrategyParams> grid;
    
    // Monthly MACD params
    vector<int> m_fasts = {6, 8, 10, 12, 14};
    vector<int> m_slows = {15, 17, 20, 24, 28};
    vector<int> m_sigs = {3, 5, 9};
    vector<int> buy_signals = {0, 1, 2, 3, 4};
    vector<int> sell_signals = {0, 1, 2};
    
    // Weekly confirm modes
    vector<int> w_confirms = {0, 3, 4};
    
    // Risk management
    vector<double> trailing_stops = {0.0, 15.0, 20.0};
    vector<double> take_profits = {0.0, 30.0, 50.0};
    
    // Portfolio
    vector<int> max_positions_opts = {1, 2, 3, 5};
    
    for (int mf : m_fasts) {
        for (int ms : m_slows) {
            if (mf >= ms) continue;
            for (int msg : m_sigs) {
                for (int bs : buy_signals) {
                    for (int ss : sell_signals) {
                        for (int wc : w_confirms) {
                            for (double tr : trailing_stops) {
                                for (double tp : take_profits) {
                                    for (int mp : max_positions_opts) {
                                        StrategyParams p;
                                        p.m_fast = mf; p.m_slow = ms; p.m_signal = msg;
                                        p.buy_signal = bs; p.sell_signal = ss;
                                        p.w_fast = 8; p.w_slow = 30; p.w_signal = 3;
                                        p.w_confirm = wc;
                                        p.trailing_stop = tr; p.take_profit = tp;
                                        p.max_positions = mp;
                                        grid.push_back(p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    cout << "参数网格大小: " << grid.size() << "\n";
    cout << "开始优化...\n\n";
    
    // Multi-threaded execution
    vector<pair<StrategyParams, BacktestResult>> all_results(grid.size());
    atomic<int> idx(0);
    atomic<int> progress(0);
    int total = grid.size();
    
    vector<thread> threads;
    int num_threads = thread::hardware_concurrency();
    cout << "使用 " << num_threads << " 个线程\n";
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            while (true) {
                int i = idx++;
                if (i >= total) break;
                BacktestResult r = run_backtest(etfs, grid[i]);
                all_results[i] = {grid[i], r};
                
                int p = ++progress;
                if (p % 5000 == 0) {
                    cout << "  进度: " << p << "/" << total << " (" << fixed << setprecision(1) << (100.0 * p / total) << "%)\n";
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    
    // Filter valid results
    vector<pair<StrategyParams, BacktestResult>> valid;
    for (auto& [params, res] : all_results) {
        if (res.trades >= 3 && res.annualized > -50) {
            valid.push_back({params, res});
        }
    }
    
    cout << "\n有效结果数: " << valid.size() << " / " << total << "\n\n";
    
    // ============================================================
    // Print Results
    // ============================================================
    
    auto print_result = [](const string& label, const StrategyParams& p, const BacktestResult& r) {
        cout << label << ":\n";
        cout << "  月线MACD: M(" << p.m_fast << "," << p.m_slow << "," << p.m_signal << ")";
        cout << " 买入信号=" << p.buy_signal << " 卖出信号=" << p.sell_signal;
        cout << " 周线确认=" << p.w_confirm << "\n";
        cout << "  风控: 移动止损=" << p.trailing_stop << "% 止盈=" << p.take_profit << "%";
        cout << " 最大持仓=" << p.max_positions << "\n";
        cout << "  年化收益: " << fixed << setprecision(2) << r.annualized << "%";
        cout << "  最大回撤: " << r.mdd * 100.0 << "%";
        cout << "  Sharpe: " << r.sharpe;
        cout << "  Calmar: " << r.calmar << "\n";
        cout << "  交易次数: " << r.trades;
        cout << "  胜率: " << r.win_rate << "%";
        cout << "  平均持有天数: " << (int)r.avg_hold_days << "\n";
    };
    
    auto print_trades = [](const BacktestResult& r) {
        cout << "\n  交易明细:\n";
        cout << "  " << setw(14) << "ETF" << setw(12) << "买入日期" << setw(12) << "卖出日期" 
             << setw(10) << "买入价" << setw(10) << "卖出价" << setw(10) << "收益%" << setw(8) << "天数" << "\n";
        cout << "  " << string(76, '-') << "\n";
        for (const auto& t : r.trade_list) {
            cout << "  " << setw(14) << t.etf_name 
                 << setw(12) << t.entry_date << setw(12) << t.exit_date
                 << setw(10) << fixed << setprecision(4) << t.entry_price 
                 << setw(10) << t.exit_price
                 << setw(10) << setprecision(2) << t.pnl_pct << "%"
                 << setw(7) << t.hold_days << "\n";
        }
    };
    
    // Sort by different criteria
    cout << "==========================================================\n";
    cout << "      周期性行业ETF轮动策略 — 优化结果\n";
    cout << "==========================================================\n\n";
    
    // Top 5 by Calmar
    sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        return a.second.calmar > b.second.calmar;
    });
    cout << ">>> Top 5 by Calmar Ratio <<<\n\n";
    for (int i = 0; i < min(5, (int)valid.size()); ++i) {
        print_result("  #" + to_string(i + 1), valid[i].first, valid[i].second);
        cout << "\n";
    }
    
    // Top 5 by Sharpe
    sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        return a.second.sharpe > b.second.sharpe;
    });
    cout << ">>> Top 5 by Sharpe Ratio <<<\n\n";
    for (int i = 0; i < min(5, (int)valid.size()); ++i) {
        print_result("  #" + to_string(i + 1), valid[i].first, valid[i].second);
        cout << "\n";
    }
    
    // Top 5 by Annual Return
    sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        return a.second.annualized > b.second.annualized;
    });
    cout << ">>> Top 5 by Annual Return <<<\n\n";
    for (int i = 0; i < min(5, (int)valid.size()); ++i) {
        print_result("  #" + to_string(i + 1), valid[i].first, valid[i].second);
        cout << "\n";
    }
    
    // Top 5 by Combined Score  
    sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        return a.second.combined_score > b.second.combined_score;
    });
    cout << ">>> Top 5 by Combined Score (0.5*Sharpe + 0.5*Calmar) <<<\n\n";
    for (int i = 0; i < min(5, (int)valid.size()); ++i) {
        print_result("  #" + to_string(i + 1), valid[i].first, valid[i].second);
        cout << "\n";
    }
    
    // Print full trade list of best combined strategy
    if (!valid.empty()) {
        cout << "\n==========================================================\n";
        cout << "  最优组合得分策略 — 完整交易记录\n";
        cout << "==========================================================\n";
        print_result("  最优策略", valid[0].first, valid[0].second);
        print_trades(valid[0].second);
        
        // Per-ETF breakdown
        cout << "\n  各ETF贡献:\n";
        unordered_map<string, pair<int, double>> etf_stats; // code -> (trades, total_pnl)
        for (const auto& t : valid[0].second.trade_list) {
            etf_stats[t.etf_name].first++;
            etf_stats[t.etf_name].second += t.pnl_pct;
        }
        for (const auto& [name, stats] : etf_stats) {
            cout << "    " << setw(14) << name << ": " << stats.first << " 笔, 累计收益 " 
                 << fixed << setprecision(2) << stats.second << "%\n";
        }
    }
    
    // Save results CSV
    ofstream csv("cyclical_etf_rotation_results.csv");
    csv << "rank,m_fast,m_slow,m_signal,buy_signal,sell_signal,w_confirm,trailing_stop,take_profit,max_positions,annualized,mdd,sharpe,calmar,combined,trades,win_rate,avg_hold_days\n";
    sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        return a.second.combined_score > b.second.combined_score;
    });
    for (int i = 0; i < min(100, (int)valid.size()); ++i) {
        const auto& p = valid[i].first;
        const auto& r = valid[i].second;
        csv << i + 1 << "," << p.m_fast << "," << p.m_slow << "," << p.m_signal
            << "," << p.buy_signal << "," << p.sell_signal << "," << p.w_confirm
            << "," << p.trailing_stop << "," << p.take_profit << "," << p.max_positions
            << "," << fixed << setprecision(4) << r.annualized << "," << r.mdd * 100.0
            << "," << r.sharpe << "," << r.calmar << "," << r.combined_score
            << "," << r.trades << "," << r.win_rate << "," << (int)r.avg_hold_days << "\n";
    }
    csv.close();
    cout << "\n结果已保存到 cyclical_etf_rotation_results.csv\n";
    
    return 0;
}
