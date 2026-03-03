/**
 * 全ETF MACD轮动策略 — 交易分析工具
 * 
 * 只跑最优参数 M(6,15,3) buy=0/2 sell=2 wc=0 ts=0 tp=30 mp=3
 * 输出详细交易记录CSV + 策略有效性分析
 * 
 * 编译: g++ -O3 -std=c++17 -o analyze_trades analyze_trades.cpp -lpthread
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
#include <numeric>
#include <set>
#include <map>

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Data Structures (same as all_etf_rotation.cpp)
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
    int m_fast, m_slow, m_signal;
    int buy_signal;
    int sell_signal;
    int w_fast, w_slow, w_signal;
    int w_confirm;
    double trailing_stop;
    double take_profit;
    int max_positions;
};

struct Trade {
    string etf_code, etf_name;
    string entry_date, exit_date;
    double entry_price, exit_price;
    double pnl_pct;
    int hold_days;
};

struct Position {
    int etf_idx;
    double entry_price;
    double highest_price;
    string entry_date;
    double shares;
    int entry_daily_idx;
};

// Globals
vector<string> global_dates;
unordered_map<string, int> g_date_to_idx;

// ============================================================
// Helper Functions (copied from all_etf_rotation.cpp)
// ============================================================

bool is_index_file(const string& code) {
    if (code.size() < 8) return false;
    string prefix = code.substr(0, 2);
    string num = code.substr(2);
    if (prefix == "sh" && num.size() == 6 && num[0] == '0' && num[1] == '0' && num[2] == '0') return true;
    if (prefix == "sz" && num.size() == 6 && num[0] == '3' && num[1] == '9' && num[2] == '9') return true;
    return false;
}

unordered_map<string, string> load_etf_names(const string& cache_path) {
    unordered_map<string, string> names;
    ifstream file(cache_path);
    if (!file.is_open()) return names;
    
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    size_t pos = 0;
    while (pos < content.size()) {
        size_t code_key = content.find("\"code\"", pos);
        if (code_key == string::npos) break;
        size_t code_colon = content.find(':', code_key + 6);
        if (code_colon == string::npos) break;
        size_t code_start = content.find('"', code_colon + 1);
        if (code_start == string::npos) break;
        size_t code_end = content.find('"', code_start + 1);
        if (code_end == string::npos) break;
        string code = content.substr(code_start + 1, code_end - code_start - 1);
        
        size_t name_key = content.find("\"name\"", code_end);
        if (name_key == string::npos) break;
        size_t name_colon = content.find(':', name_key + 6);
        if (name_colon == string::npos) break;
        size_t name_start = content.find('"', name_colon + 1);
        if (name_start == string::npos) break;
        size_t name_end = content.find('"', name_start + 1);
        if (name_end == string::npos) break;
        string name = content.substr(name_start + 1, name_end - name_start - 1);
        
        names[code] = name;
        pos = name_end + 1;
    }
    return names;
}

vector<pair<string, string>> discover_etfs(const string& data_dir, const string& cache_path) {
    auto name_map = load_etf_names(cache_path);
    vector<pair<string, string>> etfs;
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_regular_file()) continue;
        string filename = entry.path().filename().string();
        if (filename.size() < 5 || filename.substr(filename.size() - 4) != ".csv") continue;
        string code = filename.substr(0, filename.size() - 4);
        if (is_index_file(code)) continue;
        
        string numeric_code = code.substr(2);
        string name = numeric_code;
        auto it = name_map.find(numeric_code);
        if (it != name_map.end()) name = it->second;
        
        etfs.push_back({code, name});
    }
    sort(etfs.begin(), etfs.end());
    return etfs;
}

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
        return (d + 13*(m+1)/5 + y + y/4 - y/100 + y/400) % 7;
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
// Backtest with detailed trade recording
// ============================================================

struct BacktestDetail {
    vector<Trade> trades;
    vector<double> equity_curve;
    double annualized = 0, mdd = 0, sharpe = 0, calmar = 0;
    int total_trades = 0;
    double win_rate = 0;
};

BacktestDetail run_backtest_detailed(const vector<ETFData>& etfs, const StrategyParams& params) {
    int num_etfs = etfs.size();
    
    vector<vector<double>> m_dif(num_etfs), m_dea(num_etfs), m_hist(num_etfs);
    vector<vector<double>> w_dif(num_etfs), w_dea(num_etfs), w_hist(num_etfs);
    vector<vector<int>> d_to_w(num_etfs), d_to_m(num_etfs);
    
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].monthly.size() < 5 || etfs[i].weekly.size() < 10) continue;
        compute_macd(etfs[i].monthly, params.m_fast, params.m_slow, params.m_signal, m_dif[i], m_dea[i], m_hist[i]);
        compute_macd(etfs[i].weekly, params.w_fast, params.w_slow, params.w_signal, w_dif[i], w_dea[i], w_hist[i]);
        
        d_to_w[i].assign(global_dates.size(), -1);
        for (size_t w = 0; w < etfs[i].weekly.size(); ++w) {
            for (int d = etfs[i].weekly[w].first_daily_idx; d <= etfs[i].weekly[w].last_daily_idx; ++d) {
                if (d < (int)etfs[i].daily.size()) {
                    auto it = g_date_to_idx.find(etfs[i].daily[d].date);
                    if (it != g_date_to_idx.end()) d_to_w[i][it->second] = w;
                }
            }
        }
        
        d_to_m[i].assign(global_dates.size(), -1);
        for (size_t m = 0; m < etfs[i].monthly.size(); ++m) {
            for (int d = etfs[i].monthly[m].first_daily_idx; d <= etfs[i].monthly[m].last_daily_idx; ++d) {
                if (d < (int)etfs[i].daily.size()) {
                    auto it = g_date_to_idx.find(etfs[i].daily[d].date);
                    if (it != g_date_to_idx.end()) d_to_m[i][it->second] = m;
                }
            }
        }
    }
    
    double capital = 1000000.0;
    double cash = capital;
    vector<Position> active_positions;
    BacktestDetail res;
    
    int start_idx = (int)global_dates.size(), end_idx = 0;
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].daily.empty()) continue;
        auto it_s = g_date_to_idx.find(etfs[i].daily.front().date);
        auto it_e = g_date_to_idx.find(etfs[i].daily.back().date);
        if (it_s != g_date_to_idx.end()) start_idx = min(start_idx, it_s->second);
        if (it_e != g_date_to_idx.end()) end_idx = max(end_idx, it_e->second);
    }
    
    if (start_idx >= end_idx) return res;
    
    for (int d = start_idx; d <= end_idx; ++d) {
        string cur_date = global_dates[d];
        
        // ============ SELL LOGIC ============
        vector<Position> remaining_positions;
        for (auto& pos : active_positions) {
            int e_idx = pos.etf_idx;
            const auto& etf = etfs[e_idx];
            double cur_close = etf.global_close[d];
            double cur_high = etf.global_high[d];
            
            if (cur_close <= 0) { remaining_positions.push_back(pos); continue; }
            if (cur_high > pos.highest_price) pos.highest_price = cur_high;
            
            bool sell = false;
            
            int mi = d_to_m[e_idx][d];
            if (mi > 0 && mi < (int)m_dif[e_idx].size()) {
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
            
            int wi = d_to_w[e_idx][d];
            if (!sell && wi > 0 && wi < (int)w_dif[e_idx].size()) {
                if (d == g_date_to_idx[etf.daily[etf.weekly[wi].last_daily_idx].date]) {
                    if (w_dif[e_idx][wi] < w_dea[e_idx][wi] && w_dif[e_idx][wi-1] >= w_dea[e_idx][wi-1]) {
                        if (mi > 0 && mi < (int)m_dif[e_idx].size() && m_dif[e_idx][mi] < m_dif[e_idx][mi > 0 ? mi-1 : mi]) {
                            sell = true;
                        }
                    }
                }
            }
            
            if (!sell && params.trailing_stop > 0) {
                if (cur_close <= pos.highest_price * (1.0 - params.trailing_stop / 100.0)) sell = true;
            }
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
                res.trades.push_back(t);
            } else {
                remaining_positions.push_back(pos);
            }
        }
        active_positions = remaining_positions;
        
        double daily_equity = cash;
        for (const auto& pos : active_positions) {
            double p = etfs[pos.etf_idx].global_close[d];
            if (p <= 0) p = pos.entry_price;
            daily_equity += pos.shares * p;
        }
        res.equity_curve.push_back(daily_equity);
        
        // ============ BUY LOGIC ============
        if ((int)active_positions.size() < params.max_positions) {
            struct BuyCandidate { int etf_idx; double strength; };
            vector<BuyCandidate> candidates;
            
            for (int i = 0; i < num_etfs; ++i) {
                if (m_dif[i].empty() || w_dif[i].empty()) continue;
                int mi = d_to_m[i][d];
                int wi = d_to_w[i][d];
                if (mi < 2 || wi < 2) continue;
                if (mi >= (int)m_dif[i].size() || wi >= (int)w_dif[i].size()) continue;
                
                const auto& etf = etfs[i];
                if (d != g_date_to_idx[etf.daily[etf.weekly[wi].last_daily_idx].date]) continue;
                
                bool already_held = false;
                for (const auto& pos : active_positions) {
                    if (pos.etf_idx == i) { already_held = true; break; }
                }
                if (already_held) continue;
                
                bool m_buy = false;
                if (params.buy_signal == 0) {
                    m_buy = (m_dif[i][mi] > m_dea[i][mi] && m_dif[i][mi-1] <= m_dea[i][mi-1]);
                } else if (params.buy_signal == 1) {
                    m_buy = (m_dif[i][mi] > m_dif[i][mi-1]);
                } else if (params.buy_signal == 2) {
                    m_buy = (m_hist[i][mi] > 0 && m_hist[i][mi-1] <= 0);
                } else if (params.buy_signal == 3) {
                    m_buy = (m_dif[i][mi] < 0 && m_dif[i][mi] > m_dea[i][mi] && m_dif[i][mi-1] <= m_dea[i][mi-1]);
                } else if (params.buy_signal == 4) {
                    m_buy = (m_dif[i][mi] > m_dea[i][mi]);
                }
                if (!m_buy) continue;
                
                bool w_ok = true;
                if (params.w_confirm == 1) {
                    w_ok = (w_dif[i][wi] > w_dea[i][wi] && w_dif[i][wi-1] <= w_dea[i][wi-1]);
                } else if (params.w_confirm == 2) {
                    w_ok = (w_dif[i][wi] > 0);
                } else if (params.w_confirm == 3) {
                    w_ok = (w_dif[i][wi] > w_dif[i][wi-1]);
                } else if (params.w_confirm == 4) {
                    w_ok = (w_dif[i][wi] > w_dea[i][wi]);
                }
                if (!w_ok) continue;
                
                double strength = m_dif[i][mi] - m_dif[i][mi-1];
                candidates.push_back({i, strength});
            }
            
            sort(candidates.begin(), candidates.end(), [](const BuyCandidate& a, const BuyCandidate& b) {
                return a.strength > b.strength;
            });
            
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
                if (alloc < 100) continue;
                
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
    
    // Metrics
    double final_equity = res.equity_curve.empty() ? capital : res.equity_curve.back();
    double yrs = (double)res.equity_curve.size() / 252.0;
    res.total_trades = res.trades.size();
    
    if (res.equity_curve.size() > 1 && res.total_trades > 0) {
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
        
        res.mdd = mdd;
        res.annualized = (pow(final_equity / capital, 1.0 / yrs) - 1.0) * 100.0;
        
        double mean_ret = sum_ret / daily_rets.size();
        double var = 0;
        for (double r : daily_rets) var += (r - mean_ret) * (r - mean_ret);
        var /= daily_rets.size();
        double stddev = sqrt(var);
        res.sharpe = stddev > 0 ? (mean_ret - 0.02 / 252.0) / stddev * sqrt(252.0) : 0;
        res.calmar = mdd > 0 ? res.annualized / (mdd * 100.0) : 0;
        
        int wins = 0;
        for (const auto& t : res.trades) { if (t.pnl_pct > 0) wins++; }
        res.win_rate = (double)wins / res.total_trades * 100.0;
    }
    
    return res;
}

// ============================================================
// Main — Run analysis
// ============================================================

int main() {
    string data_dir = "/ceph/dang_articles/yoj/market_data_qfq/";
    string cache_path = "/ceph/dang_articles/yoj/etf_list_cache.json";
    
    cout << "==========================================================\n";
    cout << "      全ETF MACD轮动策略 — 交易分析\n";
    cout << "==========================================================\n\n";
    
    auto etf_list = discover_etfs(data_dir, cache_path);
    
    vector<ETFData> etfs;
    for (const auto& [code, name] : etf_list) {
        string path = data_dir + code + ".csv";
        ifstream file(path);
        if (!file.is_open()) continue;
        
        string line;
        getline(file, line);
        
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
        
        if (etf.daily.size() < 60) continue;
        
        // Liquidity filter
        {
            size_t n = etf.daily.size();
            size_t lookback = min(n, (size_t)60);
            double total_turnover = 0.0;
            for (size_t i = n - lookback; i < n; i++) {
                total_turnover += etf.daily[i].volume * 100.0 * etf.daily[i].close;
            }
            double avg_turnover = total_turnover / lookback;
            if (avg_turnover < 1e8) continue;
        }
        
        // Non-equity filter
        {
            bool is_non_equity = false;
            if (name.find("货币") != string::npos) is_non_equity = true;
            if (name.find("债") != string::npos) is_non_equity = true;
            if (name.find("利率") != string::npos) is_non_equity = true;
            if (name.find("豆粕") != string::npos) is_non_equity = true;
            if (name.find("能源化工") != string::npos) is_non_equity = true;
            if (name.find("短融") != string::npos) is_non_equity = true;
            if (name.find("添益") != string::npos) is_non_equity = true;
            if (name.find("日利") != string::npos) is_non_equity = true;
            if (name.find("财富宝") != string::npos) is_non_equity = true;
            // 黄金/白银ETF保留在池中(用户要求)
            // 原黄金/上海金/金ETF排除规则已移除
            if (is_non_equity) continue;
        }
        
        etf.monthly = aggregate_monthly(etf.daily, 0, etf.daily.size());
        etf.weekly = aggregate_weekly(etf.daily, 0, etf.daily.size());
        etfs.push_back(etf);
    }
    
    cout << "已加载 " << etfs.size() << " 只权益类ETF\n\n";
    
    // Build global dates
    set<string> all_dates_set;
    for (const auto& etf : etfs) {
        for (const auto& b : etf.daily) all_dates_set.insert(b.date);
    }
    global_dates.assign(all_dates_set.begin(), all_dates_set.end());
    for (int i = 0; i < (int)global_dates.size(); ++i) {
        g_date_to_idx[global_dates[i]] = i;
    }
    
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
        for (int i = 0; i < (int)global_dates.size(); ++i) {
            if (etf.global_close[i] > 0) last_close = etf.global_close[i];
            else etf.global_close[i] = last_close;
        }
    }
    
    // Run with best params: M(6,15,3) buy=0 sell=2 wc=0 ts=0 tp=30 mp=3
    // (buy=0 and buy=2 gave same results per CSV, use buy=2)
    StrategyParams best;
    best.m_fast = 6; best.m_slow = 15; best.m_signal = 3;
    best.buy_signal = 2; best.sell_signal = 2;
    best.w_fast = 8; best.w_slow = 30; best.w_signal = 3;
    best.w_confirm = 0;
    best.trailing_stop = 0; best.take_profit = 30;
    best.max_positions = 3;
    
    cout << "运行参数: M(" << best.m_fast << "," << best.m_slow << "," << best.m_signal 
         << ") buy=" << best.buy_signal << " sell=" << best.sell_signal
         << " wc=" << best.w_confirm << " ts=" << best.trailing_stop 
         << " tp=" << best.take_profit << " mp=" << best.max_positions << "\n\n";
    
    BacktestDetail result = run_backtest_detailed(etfs, best);
    
    cout << "==========================================================\n";
    cout << "  策略总览\n";
    cout << "==========================================================\n";
    cout << "  年化收益: " << fixed << setprecision(2) << result.annualized << "%\n";
    cout << "  最大回撤: " << result.mdd * 100.0 << "%\n";
    cout << "  Sharpe: " << result.sharpe << "\n";
    cout << "  Calmar: " << result.calmar << "\n";
    cout << "  总交易: " << result.total_trades << "\n";
    cout << "  胜率: " << result.win_rate << "%\n\n";
    
    // ============================================================
    // ANALYSIS 1: Save all trades to CSV
    // ============================================================
    {
        ofstream csv("trade_analysis.csv");
        csv << "trade_no,etf_code,etf_name,entry_date,exit_date,entry_price,exit_price,pnl_pct,hold_days,entry_year,pnl_bucket\n";
        for (int i = 0; i < (int)result.trades.size(); ++i) {
            const auto& t = result.trades[i];
            string year = t.entry_date.substr(0, 4);
            string bucket;
            if (t.pnl_pct < -20) bucket = "< -20%";
            else if (t.pnl_pct < -10) bucket = "-20~-10%";
            else if (t.pnl_pct < -5) bucket = "-10~-5%";
            else if (t.pnl_pct < 0) bucket = "-5~0%";
            else if (t.pnl_pct < 5) bucket = "0~5%";
            else if (t.pnl_pct < 10) bucket = "5~10%";
            else if (t.pnl_pct < 20) bucket = "10~20%";
            else if (t.pnl_pct < 30) bucket = "20~30%";
            else bucket = "> 30%";
            
            csv << i+1 << "," << t.etf_code << "," << t.etf_name << ","
                << t.entry_date << "," << t.exit_date << ","
                << fixed << setprecision(4) << t.entry_price << "," << t.exit_price << ","
                << setprecision(2) << t.pnl_pct << "," << t.hold_days << ","
                << year << "," << bucket << "\n";
        }
        csv.close();
        cout << "交易记录已保存到 trade_analysis.csv\n\n";
    }
    
    // ============================================================
    // ANALYSIS 2: Per-ETF contribution
    // ============================================================
    {
        cout << "==========================================================\n";
        cout << "  各ETF贡献分析\n";
        cout << "==========================================================\n\n";
        
        struct ETFStat {
            string name;
            int trades = 0;
            int wins = 0;
            double total_pnl = 0;
            double max_pnl = -1e9;
            double min_pnl = 1e9;
            double total_hold_days = 0;
        };
        
        map<string, ETFStat> etf_stats;
        for (const auto& t : result.trades) {
            auto& s = etf_stats[t.etf_code];
            s.name = t.etf_name;
            s.trades++;
            if (t.pnl_pct > 0) s.wins++;
            s.total_pnl += t.pnl_pct;
            s.max_pnl = max(s.max_pnl, t.pnl_pct);
            s.min_pnl = min(s.min_pnl, t.pnl_pct);
            s.total_hold_days += t.hold_days;
        }
        
        vector<pair<string, ETFStat>> sorted_etfs(etf_stats.begin(), etf_stats.end());
        sort(sorted_etfs.begin(), sorted_etfs.end(), [](const auto& a, const auto& b) {
            return a.second.total_pnl > b.second.total_pnl;
        });
        
        // Summary stats
        int etfs_with_profit = 0, etfs_with_loss = 0;
        double total_profit_from_top10 = 0, total_profit_all = 0;
        for (const auto& [code, s] : sorted_etfs) {
            if (s.total_pnl > 0) etfs_with_profit++;
            else etfs_with_loss++;
            total_profit_all += s.total_pnl;
        }
        
        cout << "  交易过的ETF总数: " << sorted_etfs.size() << "\n";
        cout << "  盈利ETF数: " << etfs_with_profit << "  亏损ETF数: " << etfs_with_loss << "\n\n";
        
        cout << "  >>> Top 15 盈利ETF <<<\n";
        cout << "  " << setw(20) << "ETF名称" << setw(8) << "交易" << setw(8) << "胜率" 
             << setw(12) << "累计收益%" << setw(10) << "最大盈%" << setw(10) << "最大亏%" 
             << setw(10) << "均持天" << "\n";
        cout << "  " << string(78, '-') << "\n";
        
        for (int i = 0; i < min(15, (int)sorted_etfs.size()); ++i) {
            const auto& s = sorted_etfs[i].second;
            double wr = s.trades > 0 ? 100.0 * s.wins / s.trades : 0;
            double avg_hold = s.trades > 0 ? s.total_hold_days / s.trades : 0;
            cout << "  " << setw(20) << s.name << setw(8) << s.trades 
                 << setw(7) << fixed << setprecision(0) << wr << "%"
                 << setw(11) << setprecision(1) << s.total_pnl << "%"
                 << setw(9) << setprecision(1) << s.max_pnl << "%"
                 << setw(9) << setprecision(1) << s.min_pnl << "%"
                 << setw(10) << setprecision(0) << avg_hold << "\n";
            if (i < 10) total_profit_from_top10 += s.total_pnl;
        }
        
        cout << "\n  >>> Bottom 10 亏损ETF <<<\n";
        cout << "  " << setw(20) << "ETF名称" << setw(8) << "交易" << setw(8) << "胜率" 
             << setw(12) << "累计收益%" << setw(10) << "最大盈%" << setw(10) << "最大亏%" << "\n";
        cout << "  " << string(68, '-') << "\n";
        
        for (int i = max(0, (int)sorted_etfs.size() - 10); i < (int)sorted_etfs.size(); ++i) {
            const auto& s = sorted_etfs[i].second;
            double wr = s.trades > 0 ? 100.0 * s.wins / s.trades : 0;
            cout << "  " << setw(20) << s.name << setw(8) << s.trades 
                 << setw(7) << fixed << setprecision(0) << wr << "%"
                 << setw(11) << setprecision(1) << s.total_pnl << "%"
                 << setw(9) << setprecision(1) << s.max_pnl << "%"
                 << setw(9) << setprecision(1) << s.min_pnl << "%"
                 << "\n";
        }
        
        // Concentration analysis
        cout << "\n  收益集中度分析:\n";
        cout << "    Top 10 ETF贡献总收益: " << fixed << setprecision(1) << total_profit_from_top10 << "%\n";
        cout << "    全部ETF总收益: " << total_profit_all << "%\n";
        double concentration = total_profit_all != 0 ? total_profit_from_top10 / total_profit_all * 100 : 0;
        cout << "    Top 10 贡献占比: " << setprecision(1) << concentration << "%\n";
    }
    
    // ============================================================
    // ANALYSIS 3: PnL distribution
    // ============================================================
    {
        cout << "\n==========================================================\n";
        cout << "  收益分布分析\n";
        cout << "==========================================================\n\n";
        
        // Buckets
        map<string, pair<int, double>> buckets; // bucket -> (count, total_pnl)
        vector<string> bucket_order = {"< -20%", "-20~-10%", "-10~-5%", "-5~0%", "0~5%", "5~10%", "10~20%", "20~30%", "> 30%"};
        for (const auto& b : bucket_order) buckets[b] = {0, 0};
        
        vector<double> all_pnl;
        double sum_pnl = 0;
        for (const auto& t : result.trades) {
            all_pnl.push_back(t.pnl_pct);
            sum_pnl += t.pnl_pct;
            
            string bucket;
            if (t.pnl_pct < -20) bucket = "< -20%";
            else if (t.pnl_pct < -10) bucket = "-20~-10%";
            else if (t.pnl_pct < -5) bucket = "-10~-5%";
            else if (t.pnl_pct < 0) bucket = "-5~0%";
            else if (t.pnl_pct < 5) bucket = "0~5%";
            else if (t.pnl_pct < 10) bucket = "5~10%";
            else if (t.pnl_pct < 20) bucket = "10~20%";
            else if (t.pnl_pct < 30) bucket = "20~30%";
            else bucket = "> 30%";
            
            buckets[bucket].first++;
            buckets[bucket].second += t.pnl_pct;
        }
        
        sort(all_pnl.begin(), all_pnl.end());
        double median_pnl = all_pnl.size() > 0 ? all_pnl[all_pnl.size() / 2] : 0;
        double mean_pnl = all_pnl.size() > 0 ? sum_pnl / all_pnl.size() : 0;
        
        cout << "  单笔交易统计:\n";
        cout << "    总交易数: " << result.total_trades << "\n";
        cout << "    平均收益: " << fixed << setprecision(2) << mean_pnl << "%\n";
        cout << "    中位数收益: " << median_pnl << "%\n";
        cout << "    最大盈利: " << all_pnl.back() << "%\n";
        cout << "    最大亏损: " << all_pnl.front() << "%\n";
        cout << "    标准差: ";
        double var = 0;
        for (double p : all_pnl) var += (p - mean_pnl) * (p - mean_pnl);
        cout << sqrt(var / all_pnl.size()) << "%\n\n";
        
        cout << "  收益分桶:\n";
        cout << "  " << setw(12) << "区间" << setw(8) << "笔数" << setw(8) << "占比" << setw(14) << "贡献收益%\n";
        cout << "  " << string(42, '-') << "\n";
        for (const auto& b : bucket_order) {
            int cnt = buckets[b].first;
            double pct = 100.0 * cnt / result.total_trades;
            cout << "  " << setw(12) << b << setw(8) << cnt 
                 << setw(7) << fixed << setprecision(1) << pct << "%"
                 << setw(12) << setprecision(1) << buckets[b].second << "%\n";
        }
        
        // Profit factor: sum of wins / abs(sum of losses)
        double total_wins = 0, total_losses = 0;
        for (double p : all_pnl) {
            if (p > 0) total_wins += p;
            else total_losses += fabs(p);
        }
        cout << "\n  盈亏比分析:\n";
        cout << "    总盈利: " << fixed << setprecision(1) << total_wins << "%\n";
        cout << "    总亏损: " << total_losses << "%\n";
        cout << "    盈亏比(Profit Factor): " << setprecision(2) << (total_losses > 0 ? total_wins / total_losses : 999) << "\n";
    }
    
    // ============================================================
    // ANALYSIS 4: Year-by-year breakdown
    // ============================================================
    {
        cout << "\n==========================================================\n";
        cout << "  逐年交易分析\n";
        cout << "==========================================================\n\n";
        
        map<string, vector<const Trade*>> yearly_trades;
        for (const auto& t : result.trades) {
            string year = t.entry_date.substr(0, 4);
            yearly_trades[year].push_back(&t);
        }
        
        cout << "  " << setw(6) << "年份" << setw(8) << "交易" << setw(8) << "胜率" 
             << setw(12) << "平均收益%" << setw(12) << "总收益%" << setw(10) << "最大盈%" << setw(10) << "最大亏%" 
             << setw(10) << "均持天" << "\n";
        cout << "  " << string(76, '-') << "\n";
        
        for (const auto& [year, trades] : yearly_trades) {
            int wins = 0;
            double total = 0, max_p = -1e9, min_p = 1e9, total_hold = 0;
            for (const auto* t : trades) {
                if (t->pnl_pct > 0) wins++;
                total += t->pnl_pct;
                max_p = max(max_p, t->pnl_pct);
                min_p = min(min_p, t->pnl_pct);
                total_hold += t->hold_days;
            }
            double wr = 100.0 * wins / trades.size();
            double avg = total / trades.size();
            double avg_hold = total_hold / trades.size();
            cout << "  " << setw(6) << year << setw(8) << trades.size()
                 << setw(7) << fixed << setprecision(0) << wr << "%"
                 << setw(11) << setprecision(1) << avg << "%"
                 << setw(11) << setprecision(1) << total << "%"
                 << setw(9) << setprecision(1) << max_p << "%"
                 << setw(9) << setprecision(1) << min_p << "%"
                 << setw(10) << setprecision(0) << avg_hold << "\n";
        }
    }
    
    // ============================================================
    // ANALYSIS 5: Take-profit hit analysis  
    // ============================================================
    {
        cout << "\n==========================================================\n";
        cout << "  止盈触发分析 (tp=30%)\n";
        cout << "==========================================================\n\n";
        
        int tp_hits = 0, normal_sells = 0, loss_sells = 0;
        double tp_pnl = 0, normal_pnl = 0, loss_pnl = 0;
        
        for (const auto& t : result.trades) {
            // Take profit ~ 30% (allow some slippage)
            if (t.pnl_pct >= 28.0) {
                tp_hits++;
                tp_pnl += t.pnl_pct;
            } else if (t.pnl_pct > 0) {
                normal_sells++;
                normal_pnl += t.pnl_pct;
            } else {
                loss_sells++;
                loss_pnl += t.pnl_pct;
            }
        }
        
        cout << "  止盈出场 (>=28%): " << tp_hits << " 笔, 总贡献 " 
             << fixed << setprecision(1) << tp_pnl << "%\n";
        cout << "  正常盈利出场: " << normal_sells << " 笔, 总贡献 " << normal_pnl << "%\n";
        cout << "  亏损出场: " << loss_sells << " 笔, 总贡献 " << loss_pnl << "%\n\n";
        
        cout << "  止盈占总盈利比例: " << setprecision(1) 
             << (tp_pnl + normal_pnl > 0 ? tp_pnl / (tp_pnl + normal_pnl) * 100 : 0) << "%\n";
    }
    
    // ============================================================
    // ANALYSIS 6: Holding period analysis
    // ============================================================
    {
        cout << "\n==========================================================\n";
        cout << "  持仓时间分析\n";
        cout << "==========================================================\n\n";
        
        map<string, pair<int, double>> hold_buckets;
        vector<string> hold_order = {"1-7天", "8-14天", "15-30天", "31-60天", "61-90天", ">90天"};
        for (const auto& b : hold_order) hold_buckets[b] = {0, 0};
        
        double total_invested_days = 0;
        for (const auto& t : result.trades) {
            total_invested_days += t.hold_days;
            string bucket;
            if (t.hold_days <= 7) bucket = "1-7天";
            else if (t.hold_days <= 14) bucket = "8-14天";
            else if (t.hold_days <= 30) bucket = "15-30天";
            else if (t.hold_days <= 60) bucket = "31-60天";
            else if (t.hold_days <= 90) bucket = "61-90天";
            else bucket = ">90天";
            
            hold_buckets[bucket].first++;
            hold_buckets[bucket].second += t.pnl_pct;
        }
        
        cout << "  " << setw(10) << "持仓时间" << setw(8) << "笔数" << setw(8) << "占比" << setw(14) << "贡献收益%\n";
        cout << "  " << string(40, '-') << "\n";
        for (const auto& b : hold_order) {
            int cnt = hold_buckets[b].first;
            double pct = 100.0 * cnt / result.total_trades;
            cout << "  " << setw(10) << b << setw(8) << cnt 
                 << setw(7) << fixed << setprecision(1) << pct << "%"
                 << setw(12) << setprecision(1) << hold_buckets[b].second << "%\n";
        }
        
        // Cash idle time
        double total_calendar_days = result.equity_curve.size();
        // max_positions=3, so max invested time = 3 * calendar_days
        double utilization = total_invested_days / (total_calendar_days * 3) * 100;
        cout << "\n  总日历天数: " << (int)total_calendar_days << "\n";
        cout << "  总持仓天数(所有仓位合计): " << (int)total_invested_days << "\n";
        cout << "  仓位利用率(vs 3仓位): " << fixed << setprecision(1) << utilization << "%\n";
    }
    
    // ============================================================
    // ANALYSIS 7: Consecutive wins/losses  
    // ============================================================
    {
        cout << "\n==========================================================\n";
        cout << "  连续盈亏分析\n";
        cout << "==========================================================\n\n";
        
        int max_consec_wins = 0, max_consec_losses = 0;
        int cur_wins = 0, cur_losses = 0;
        
        for (const auto& t : result.trades) {
            if (t.pnl_pct > 0) {
                cur_wins++;
                cur_losses = 0;
                max_consec_wins = max(max_consec_wins, cur_wins);
            } else {
                cur_losses++;
                cur_wins = 0;
                max_consec_losses = max(max_consec_losses, cur_losses);
            }
        }
        
        cout << "  最长连续盈利: " << max_consec_wins << " 笔\n";
        cout << "  最长连续亏损: " << max_consec_losses << " 笔\n";
    }
    
    // ============================================================
    // ANALYSIS 8: Remove top N trades — robustness check
    // ============================================================
    {
        cout << "\n==========================================================\n";
        cout << "  去除大赢家鲁棒性分析\n";
        cout << "==========================================================\n\n";
        
        vector<double> all_pnl;
        for (const auto& t : result.trades) all_pnl.push_back(t.pnl_pct);
        sort(all_pnl.begin(), all_pnl.end(), greater<>());
        
        double total_pnl_sum = 0;
        for (double p : all_pnl) total_pnl_sum += p;
        
        cout << "  如果去掉最赚钱的N笔交易，策略是否仍然盈利？\n\n";
        cout << "  " << setw(14) << "去掉" << setw(14) << "剩余总收益%" << setw(14) << "仍盈利?\n";
        cout << "  " << string(42, '-') << "\n";
        
        double removed = 0;
        for (int n : {0, 1, 3, 5, 10, 15, 20, 30}) {
            if (n >= (int)all_pnl.size()) break;
            double rem = 0;
            for (int i = 0; i < n; ++i) rem += all_pnl[i];
            double remain = total_pnl_sum - rem;
            cout << "  " << setw(10) << ("Top " + to_string(n)) << " 笔"
                 << setw(13) << fixed << setprecision(1) << remain << "%"
                 << setw(10) << (remain > 0 ? "是" : "否") << "\n";
        }
    }
    
    cout << "\n==========================================================\n";
    cout << "  分析完毕\n";
    cout << "==========================================================\n";
    
    return 0;
}
