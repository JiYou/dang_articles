/**
 * 全部ETF轮动策略 — Walk-Forward验证
 * 
 * 基于 all_etf_rotation.cpp（300只权益类ETF，前复权数据），
 * 增加 cyclical_etf_rotation_wf.cpp 的Walk-Forward验证结构。
 * 
 * 关键差异：
 * - 自动发现 market_data_qfq/ 下的所有ETF
 * - 流动性过滤（成交额 < 1亿元剔除）
 * - 非权益类ETF过滤（债券/货币/商品）
 * - 滚动窗口: 2年训练 + 1年测试
 * - 训练期网格搜索最优参数 → 测试期验证
 * - 参数网格针对WF缩小（避免过长运行时间）
 * 
 * 编译: g++ -O3 -std=c++17 -o all_etf_rotation_wf all_etf_rotation_wf.cpp -lpthread
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
    int m_fast, m_slow, m_signal;
    int buy_signal;    // 0=golden cross, 1=DIF turning up, 2=MACD hist > 0, 3=underwater golden cross, 4=DIF>DEA
    int sell_signal;   // 0=death cross, 1=DIF turning down, 2=DIF < 0
    
    int w_fast, w_slow, w_signal;
    int w_confirm;     // 0=none, 3=weekly DIF rising, 4=weekly DIF>DEA
    
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

struct BacktestResult {
    double sharpe = -100;
    double calmar = -100;
    double combined_score = -100;
    double mdd = 0;
    double annualized = 0;
    int trades = 0;
    double win_rate = 0;
    double avg_hold_days = 0;
    double final_equity = 0;
    vector<Trade> trade_list;
    vector<double> equity_curve;
};

struct WindowResult {
    string train_start, train_end, test_start, test_end;
    StrategyParams best_params;
    double train_combined;
    double train_annualized;
    double train_mdd;
    double train_sharpe;
    double train_calmar;
    int train_trades;
    double test_annualized;
    double test_mdd;
    double test_sharpe;
    double test_calmar;
    int test_trades;
    double test_win_rate;
};

// Raw ETF for initial loading (before walk-forward window processing)
struct RawETF {
    string code, name;
    vector<DailyBar> daily;
};

// ============================================================
// Globals
// ============================================================

const double RISK_FREE_RATE = 0.02;

// ============================================================
// Auto-discover ETFs from market_data_qfq directory
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
    if (!file.is_open()) {
        cerr << "警告: 无法打开ETF名称缓存: " << cache_path << "\n";
        return names;
    }
    
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
    cout << "从缓存加载了 " << name_map.size() << " 个ETF名称\n";
    
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
        if (it != name_map.end()) {
            name = it->second;
        }
        
        etfs.push_back({code, name});
    }
    
    sort(etfs.begin(), etfs.end());
    
    cout << "发现 " << etfs.size() << " 只ETF (已排除指数文件)\n";
    return etfs;
}

// ============================================================
// Non-equity filter
// ============================================================

bool is_non_equity(const string& name) {
    if (name.find("货币") != string::npos) return true;
    if (name.find("债") != string::npos) return true;
    if (name.find("利率") != string::npos) return true;
    if (name.find("豆粕") != string::npos) return true;
    if (name.find("能源化工") != string::npos) return true;
    if (name.find("短融") != string::npos) return true;
    if (name.find("添益") != string::npos) return true;
    if (name.find("日利") != string::npos) return true;
    if (name.find("财富宝") != string::npos) return true;
    // 黄金/金ETF排除，但黄金股ETF保留
    if (name.find("黄金") != string::npos && name.find("股") == string::npos) return true;
    if (name.find("上海金") != string::npos) return true;
    if (name == "金ETF") return true;
    return false;
}

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
// Portfolio Backtest (windowed)
// ============================================================

BacktestResult run_backtest_windowed(
    const vector<ETFData>& etfs,
    const StrategyParams& params,
    const vector<string>& global_dates,
    const unordered_map<string, int>& g_date_to_idx,
    int global_start_idx,
    int global_end_idx
) {
    int num_etfs = etfs.size();
    
    vector<vector<double>> m_dif(num_etfs), m_dea(num_etfs), m_hist(num_etfs);
    vector<vector<double>> w_dif(num_etfs), w_dea(num_etfs), w_hist(num_etfs);
    vector<vector<int>> d_to_w(num_etfs);
    vector<vector<int>> d_to_m(num_etfs);
    
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].monthly.size() < 5 || etfs[i].weekly.size() < 10) continue;
        
        compute_macd(etfs[i].monthly, params.m_fast, params.m_slow, params.m_signal, m_dif[i], m_dea[i], m_hist[i]);
        compute_macd(etfs[i].weekly, params.w_fast, params.w_slow, params.w_signal, w_dif[i], w_dea[i], w_hist[i]);
        
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
    
    res.equity_curve.reserve(global_end_idx - global_start_idx + 1);
    
    for (int d = global_start_idx; d <= global_end_idx; ++d) {
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
            
            int mi = d_to_m[e_idx][d];
            if (mi > 0 && mi < (int)m_dif[e_idx].size()) {
                if (d == g_date_to_idx.at(etf.daily[etf.monthly[mi].last_daily_idx].date)) {
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
                if (d == g_date_to_idx.at(etf.daily[etf.weekly[wi].last_daily_idx].date)) {
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
            
            if (sell || d == global_end_idx) {
                double exit_price = cur_close;
                if (d < global_end_idx && sell) {
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
        
        double daily_equity = cash;
        for (const auto& pos : active_positions) {
            double p = etfs[pos.etf_idx].global_close[d];
            if (p <= 0) p = pos.entry_price;
            daily_equity += pos.shares * p;
        }
        res.equity_curve.push_back(daily_equity);
        
        // ============ BUY LOGIC ============
        if ((int)active_positions.size() < params.max_positions) {
            struct BuyCandidate {
                int etf_idx;
                double strength;
            };
            vector<BuyCandidate> candidates;
            
            for (int i = 0; i < num_etfs; ++i) {
                if (m_dif[i].empty() || w_dif[i].empty()) continue;
                
                int mi = d_to_m[i][d];
                int wi = d_to_w[i][d];
                if (mi < 2 || wi < 2) continue;
                if (mi >= (int)m_dif[i].size() || wi >= (int)w_dif[i].size()) continue;
                
                const auto& etf = etfs[i];
                if (d != g_date_to_idx.at(etf.daily[etf.weekly[wi].last_daily_idx].date)) continue;
                
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
                if (d < global_end_idx) {
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
                p.entry_date = (d < global_end_idx) ? global_dates[d + 1] : cur_date;
                p.shares = shares;
                p.entry_daily_idx = d;
                active_positions.push_back(p);
            }
        }
    }
    
    // ============ Metrics ============
    double final_eq = res.equity_curve.empty() ? capital : res.equity_curve.back();
    res.final_equity = final_eq;
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
        
        double ann_ret = pow(final_eq / capital, 1.0 / yrs) - 1.0;
        double ann_vol = std_ret * sqrt(252.0);
        
        res.sharpe = ann_vol > 0 ? (ann_ret - RISK_FREE_RATE) / ann_vol : 0;
        res.mdd = mdd;
        res.calmar = mdd > 0 ? ann_ret / mdd : 0;
        if (mdd == 0 && ann_ret > 0) res.calmar = 100.0;
        res.annualized = ann_ret * 100.0;
        res.combined_score = 0.5 * res.sharpe + 0.5 * min(res.calmar, 10.0);
        
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
    string data_dir = "/ceph/dang_articles/yoj/market_data_qfq/";
    string cache_path = "/ceph/dang_articles/yoj/etf_list_cache.json";
    
    cout << "==========================================================\n";
    cout << "  全部ETF轮动策略 — Walk-Forward验证\n";
    cout << "  数据源: market_data_qfq/ (前复权)\n";
    cout << "==========================================================\n\n";
    
    // Auto-discover ETFs
    auto etf_list = discover_etfs(data_dir, cache_path);
    
    // Load raw daily data with filtering
    vector<RawETF> raw_etfs;
    int skipped_liquidity = 0, skipped_equity = 0, skipped_data = 0;
    
    for (const auto& [code, name] : etf_list) {
        string path = data_dir + code + ".csv";
        ifstream file(path);
        if (!file.is_open()) continue;
        
        string line;
        getline(file, line); // header
        
        RawETF etf;
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
            skipped_data++;
            continue;
        }
        
        // 流动性过滤: 最近60个交易日平均日成交额 >= 1亿元
        // volume单位是手(1手=100股), 成交额 = volume * 100 * close
        {
            size_t n = etf.daily.size();
            size_t lookback = min(n, (size_t)60);
            double total_turnover = 0.0;
            for (size_t i = n - lookback; i < n; i++) {
                total_turnover += etf.daily[i].volume * 100.0 * etf.daily[i].close;
            }
            double avg_turnover = total_turnover / lookback;
            if (avg_turnover < 1e8) {
                skipped_liquidity++;
                continue;
            }
        }
        
        // 非权益类ETF过滤
        if (is_non_equity(name)) {
            skipped_equity++;
            continue;
        }
        
        raw_etfs.push_back(etf);
    }
    
    cout << "\n已加载 " << raw_etfs.size() << " 只权益类ETF"
         << " (跳过: 数据不足=" << skipped_data
         << " 流动性不足=" << skipped_liquidity
         << " 非权益=" << skipped_equity << ")\n\n";
    
    if (raw_etfs.empty()) {
        cerr << "错误: 没有加载到任何ETF数据!\n";
        return 1;
    }
    
    // Print loaded ETFs summary
    cout << "ETF池示例 (前10只):\n";
    for (int i = 0; i < min(10, (int)raw_etfs.size()); ++i) {
        cout << "  " << raw_etfs[i].name << " (" << raw_etfs[i].code << "): "
             << raw_etfs[i].daily.size() << " bars ("
             << raw_etfs[i].daily.front().date << " ~ " << raw_etfs[i].daily.back().date << ")\n";
    }
    if ((int)raw_etfs.size() > 10) {
        cout << "  ... 共 " << raw_etfs.size() << " 只\n";
    }
    cout << "\n";
    
    // Build global date union from ALL data
    set<string> all_dates_set;
    for (const auto& etf : raw_etfs) {
        for (const auto& b : etf.daily) all_dates_set.insert(b.date);
    }
    vector<string> global_dates(all_dates_set.begin(), all_dates_set.end());
    unordered_map<string, int> g_date_to_idx;
    for (int i = 0; i < (int)global_dates.size(); ++i) {
        g_date_to_idx[global_dates[i]] = i;
    }
    
    cout << "全局日期范围: " << global_dates.front() << " ~ " << global_dates.back()
         << " (" << global_dates.size() << " 个交易日)\n\n";
    
    // Helper: build ETFData array for a specific date window
    // Uses ALL data up to end_date for MACD warmup (no lookahead)
    auto build_etfs_for_window = [&](int date_start_idx, int date_end_idx) -> vector<ETFData> {
        vector<ETFData> etfs;
        string end_date = global_dates[date_end_idx];
        
        for (const auto& raw : raw_etfs) {
            ETFData etf;
            etf.code = raw.code;
            etf.name = raw.name;
            
            // Include ALL daily bars up to end_date for MACD warmup
            for (const auto& bar : raw.daily) {
                if (bar.date > end_date) break;
                etf.daily.push_back(bar);
            }
            
            if (etf.daily.size() < 60) continue;
            
            // Aggregate using all available data up to end_date
            etf.monthly = aggregate_monthly(etf.daily, 0, etf.daily.size());
            etf.weekly = aggregate_weekly(etf.daily, 0, etf.daily.size());
            
            // Build global price arrays
            etf.global_close.assign(global_dates.size(), 0.0);
            etf.global_open.assign(global_dates.size(), 0.0);
            etf.global_high.assign(global_dates.size(), 0.0);
            
            for (const auto& b : etf.daily) {
                auto it = g_date_to_idx.find(b.date);
                if (it != g_date_to_idx.end()) {
                    etf.global_close[it->second] = b.close;
                    etf.global_open[it->second] = b.open;
                    etf.global_high[it->second] = b.high;
                }
            }
            double last_close = 0.0;
            for (int i = 0; i <= date_end_idx; ++i) {
                if (etf.global_close[i] > 0) last_close = etf.global_close[i];
                else etf.global_close[i] = last_close;
            }
            
            etfs.push_back(etf);
        }
        return etfs;
    };
    
    // ============================================================
    // Parameter Grid (Reduced for Walk-Forward)
    // ============================================================
    // Full grid is 121,500 — too slow with 300 ETFs per window.
    // Use reduced grid (~3,888 params) covering key dimensions.
    
    vector<StrategyParams> grid;
    
    vector<int> m_fasts = {6, 10, 14};
    vector<int> m_slows = {15, 20, 28};
    vector<int> m_sigs = {3, 5};
    vector<int> buy_signals = {0, 1, 2};
    vector<int> sell_signals = {0, 1, 2};
    vector<int> w_confirms = {0, 3};
    vector<double> trailing_stops = {0.0, 15.0};
    vector<double> take_profits = {0.0, 30.0};
    vector<int> max_positions_opts = {1, 3, 5};
    
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
    
    cout << "参数网格大小: " << grid.size() << " (Walk-Forward缩减版)\n";
    cout << "  月MACD: fast=" << "6,10,14" << " slow=" << "15,20,28" << " signal=" << "3,5\n";
    cout << "  买入信号: 0,1,2  卖出信号: 0,1,2  周线确认: 0,3\n";
    cout << "  移动止损: 0,15%  止盈: 0,30%  持仓: 1,3,5\n\n";
    
    // ============================================================
    // Walk-Forward Windows
    // ============================================================
    
    int first_year = stoi(global_dates.front().substr(0, 4));
    int last_year = stoi(global_dates.back().substr(0, 4));
    
    cout << "==========================================================\n";
    cout << "  Walk-Forward 验证 (滚动窗口: 2年训练 + 1年测试)\n";
    cout << "==========================================================\n\n";
    
    vector<WindowResult> wf_results;
    
    for (int start_y = first_year; start_y + 3 <= last_year + 1; ++start_y) {
        string train_start = to_string(start_y) + "-01-01";
        string train_end = to_string(start_y + 2) + "-01-01";
        string test_start = train_end;
        string test_end = to_string(start_y + 3) + "-01-01";
        
        int train_start_idx = -1, train_end_idx = -1;
        int test_start_idx = -1, test_end_idx = -1;
        
        for (int i = 0; i < (int)global_dates.size(); ++i) {
            if (train_start_idx == -1 && global_dates[i] >= train_start) train_start_idx = i;
            if (train_end_idx == -1 && global_dates[i] >= train_end) train_end_idx = i - 1;
            if (test_start_idx == -1 && global_dates[i] >= test_start) test_start_idx = i;
            if (test_end_idx == -1 && global_dates[i] >= test_end) test_end_idx = i - 1;
        }
        if (train_end_idx == -1) train_end_idx = global_dates.size() - 1;
        if (test_start_idx == -1) continue;
        if (test_end_idx == -1) test_end_idx = global_dates.size() - 1;
        
        if (train_end_idx - train_start_idx < 200) continue;
        if (test_end_idx - test_start_idx < 50) continue;
        
        cout << "窗口 " << wf_results.size() + 1 << ": 训练 [" << global_dates[train_start_idx].substr(0,7)
             << " ~ " << global_dates[train_end_idx].substr(0,7)
             << "] 测试 [" << global_dates[test_start_idx].substr(0,7)
             << " ~ " << global_dates[test_end_idx].substr(0,7) << "]\n";
        
        // Build ETFs for TRAINING (data only up to train_end)
        auto train_etfs = build_etfs_for_window(train_start_idx, train_end_idx);
        
        cout << "  训练ETF数量: " << train_etfs.size() << "\n";
        
        if (train_etfs.size() < 3) {
            cout << "  [SKIP] ETF数量不足\n\n";
            continue;
        }
        
        // Grid search on training data
        vector<pair<StrategyParams, BacktestResult>> train_results(grid.size());
        atomic<int> idx(0);
        atomic<int> progress(0);
        int total = grid.size();
        
        vector<thread> threads;
        int num_threads = thread::hardware_concurrency();
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                while (true) {
                    int i = idx++;
                    if (i >= total) break;
                    auto r = run_backtest_windowed(train_etfs, grid[i], global_dates, g_date_to_idx,
                                                   train_start_idx, train_end_idx);
                    train_results[i] = {grid[i], r};
                    int p = ++progress;
                    if (p % 500 == 0) {
                        cout << "    训练进度: " << p << "/" << total << " (" 
                             << fixed << setprecision(1) << (100.0 * p / total) << "%)\r" << flush;
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
        cout << "    训练完成: " << total << "/" << total << "\n";
        
        // Find best training params by combined score
        int best_idx = 0;
        double best_combined = -1e9;
        for (int i = 0; i < (int)train_results.size(); ++i) {
            if (train_results[i].second.trades >= 3 && train_results[i].second.combined_score > best_combined) {
                best_combined = train_results[i].second.combined_score;
                best_idx = i;
            }
        }
        
        auto& best_train = train_results[best_idx];
        
        cout << "  最优训练参数: M(" << best_train.first.m_fast << "," << best_train.first.m_slow << "," << best_train.first.m_signal
             << ") buy=" << best_train.first.buy_signal << " sell=" << best_train.first.sell_signal
             << " wc=" << best_train.first.w_confirm << " ts=" << best_train.first.trailing_stop
             << " tp=" << best_train.first.take_profit << " mp=" << best_train.first.max_positions << "\n";
        cout << "  训练表现: 年化=" << fixed << setprecision(2) << best_train.second.annualized << "%"
             << " 回撤=" << best_train.second.mdd * 100 << "%"
             << " Sharpe=" << best_train.second.sharpe
             << " Calmar=" << best_train.second.calmar
             << " 交易=" << best_train.second.trades << "\n";
        
        // Test: build ETFs with data up to test_end, run with best params on test period
        auto test_etfs = build_etfs_for_window(test_start_idx, test_end_idx);
        
        cout << "  测试ETF数量: " << test_etfs.size() << "\n";
        
        auto test_result = run_backtest_windowed(test_etfs, best_train.first, global_dates, g_date_to_idx,
                                                  test_start_idx, test_end_idx);
        
        cout << "  测试表现: 年化=" << fixed << setprecision(2) << test_result.annualized << "%"
             << " 回撤=" << test_result.mdd * 100 << "%"
             << " Sharpe=" << test_result.sharpe
             << " Calmar=" << test_result.calmar
             << " 交易=" << test_result.trades
             << " 胜率=" << test_result.win_rate << "%\n";
        
        // Top-5 median test performance (robustness check)
        vector<pair<int, double>> ranked_train;
        for (int i = 0; i < (int)train_results.size(); ++i) {
            if (train_results[i].second.trades >= 3) {
                ranked_train.push_back({i, train_results[i].second.combined_score});
            }
        }
        sort(ranked_train.begin(), ranked_train.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        
        if (ranked_train.size() >= 5) {
            vector<double> top5_test_anns;
            for (int k = 0; k < 5; ++k) {
                auto r = run_backtest_windowed(test_etfs, train_results[ranked_train[k].first].first,
                                               global_dates, g_date_to_idx, test_start_idx, test_end_idx);
                top5_test_anns.push_back(r.annualized);
            }
            sort(top5_test_anns.begin(), top5_test_anns.end());
            double median5 = top5_test_anns[2];
            cout << "  Top-5 训练参数测试中位数: " << fixed << setprecision(2) << median5 << "%\n";
        }
        
        // Store window result
        WindowResult wr;
        wr.train_start = global_dates[train_start_idx];
        wr.train_end = global_dates[train_end_idx];
        wr.test_start = global_dates[test_start_idx];
        wr.test_end = global_dates[test_end_idx];
        wr.best_params = best_train.first;
        wr.train_combined = best_train.second.combined_score;
        wr.train_annualized = best_train.second.annualized;
        wr.train_mdd = best_train.second.mdd;
        wr.train_sharpe = best_train.second.sharpe;
        wr.train_calmar = best_train.second.calmar;
        wr.train_trades = best_train.second.trades;
        wr.test_annualized = test_result.annualized;
        wr.test_mdd = test_result.mdd;
        wr.test_sharpe = test_result.sharpe;
        wr.test_calmar = test_result.calmar;
        wr.test_trades = test_result.trades;
        wr.test_win_rate = test_result.win_rate;
        wf_results.push_back(wr);
        
        cout << "\n";
    }
    
    // ============================================================
    // Fixed-params walk-forward with the best known params
    // ============================================================
    
    cout << "==========================================================\n";
    cout << "  固定参数 Walk-Forward (用全样本最优参数逐年测试)\n";
    cout << "==========================================================\n\n";
    
    // Best params from all_etf_rotation.cpp (300 ETF, equity-only, 前复权)
    StrategyParams fixed_best;
    fixed_best.m_fast = 6; fixed_best.m_slow = 15; fixed_best.m_signal = 3;
    fixed_best.buy_signal = 0; fixed_best.sell_signal = 2;
    fixed_best.w_fast = 8; fixed_best.w_slow = 30; fixed_best.w_signal = 3;
    fixed_best.w_confirm = 0;
    fixed_best.trailing_stop = 0.0; fixed_best.take_profit = 30.0;
    fixed_best.max_positions = 3;
    
    cout << "固定参数: M(6,15,3) buy=0 sell=2 wc=0 ts=0 tp=30 mp=3\n";
    cout << "(全样本最优: 49.95% 年化, 13.30% 回撤, Sharpe 2.52, Calmar 3.76)\n\n";
    
    vector<pair<string, double>> yearly_returns;
    
    for (int y = first_year; y <= last_year; ++y) {
        string y_start = to_string(y) + "-01-01";
        string y_end = to_string(y + 1) + "-01-01";
        
        int y_start_idx = -1, y_end_idx = -1;
        for (int i = 0; i < (int)global_dates.size(); ++i) {
            if (y_start_idx == -1 && global_dates[i] >= y_start) y_start_idx = i;
            if (y_end_idx == -1 && global_dates[i] >= y_end) y_end_idx = i - 1;
        }
        if (y_start_idx == -1) continue;
        if (y_end_idx == -1) y_end_idx = global_dates.size() - 1;
        if (y_end_idx - y_start_idx < 50) continue;
        
        auto year_etfs = build_etfs_for_window(y_start_idx, y_end_idx);
        if (year_etfs.size() < 3) continue;
        
        auto r = run_backtest_windowed(year_etfs, fixed_best, global_dates, g_date_to_idx,
                                        y_start_idx, y_end_idx);
        
        cout << "  " << y << ": 年化=" << fixed << setprecision(2) << r.annualized << "%"
             << "  回撤=" << r.mdd * 100 << "%"
             << "  Sharpe=" << r.sharpe
             << "  交易=" << r.trades
             << "  胜率=" << r.win_rate << "%\n";
        
        yearly_returns.push_back({to_string(y), r.annualized});
    }
    
    // ============================================================
    // Summary
    // ============================================================
    
    cout << "\n==========================================================\n";
    cout << "  Walk-Forward 总结\n";
    cout << "==========================================================\n\n";
    
    if (!wf_results.empty()) {
        cout << ">>> 滚动窗口 Walk-Forward (2年训练 + 1年测试) <<<\n\n";
        
        cout << setw(10) << "窗口" << setw(15) << "训练年化" << setw(15) << "测试年化"
             << setw(12) << "测试回撤" << setw(12) << "测试Sharpe" << setw(12) << "测试Calmar"
             << setw(10) << "测试交易" << "\n";
        cout << string(86, '-') << "\n";
        
        vector<double> test_anns, test_mdds, test_sharpes;
        int pos_windows = 0;
        
        for (int i = 0; i < (int)wf_results.size(); ++i) {
            const auto& wr = wf_results[i];
            cout << setw(10) << ("W" + to_string(i+1))
                 << setw(14) << fixed << setprecision(2) << wr.train_annualized << "%"
                 << setw(14) << wr.test_annualized << "%"
                 << setw(11) << wr.test_mdd * 100 << "%"
                 << setw(12) << setprecision(2) << wr.test_sharpe
                 << setw(12) << wr.test_calmar
                 << setw(10) << wr.test_trades << "\n";
            test_anns.push_back(wr.test_annualized);
            test_mdds.push_back(wr.test_mdd);
            test_sharpes.push_back(wr.test_sharpe);
            if (wr.test_annualized > 0) pos_windows++;
        }
        
        sort(test_anns.begin(), test_anns.end());
        double median_ann = test_anns.size() % 2 == 0
            ? (test_anns[test_anns.size()/2-1] + test_anns[test_anns.size()/2]) / 2.0
            : test_anns[test_anns.size()/2];
        double avg_ann = accumulate(test_anns.begin(), test_anns.end(), 0.0) / test_anns.size();
        double worst_ann = test_anns.front();
        double best_ann = test_anns.back();
        double max_test_mdd = *max_element(test_mdds.begin(), test_mdds.end());
        double avg_sharpe = accumulate(test_sharpes.begin(), test_sharpes.end(), 0.0) / test_sharpes.size();
        
        cout << "\n统计:\n";
        cout << "  窗口数量: " << wf_results.size() << "\n";
        cout << "  正收益窗口: " << pos_windows << "/" << wf_results.size()
             << " (" << fixed << setprecision(1) << (100.0 * pos_windows / wf_results.size()) << "%)\n";
        cout << "  测试年化中位数: " << fixed << setprecision(2) << median_ann << "%\n";
        cout << "  测试年化均值: " << avg_ann << "%\n";
        cout << "  最差窗口: " << worst_ann << "%\n";
        cout << "  最好窗口: " << best_ann << "%\n";
        cout << "  最大测试回撤: " << max_test_mdd * 100 << "%\n";
        cout << "  测试Sharpe均值: " << avg_sharpe << "\n";
    }
    
    // Fixed-params yearly summary
    if (!yearly_returns.empty()) {
        cout << "\n>>> 固定参数逐年收益 <<<\n";
        int pos_years = 0;
        double sum_ann = 0;
        for (const auto& [year, ann] : yearly_returns) {
            if (ann > 0) pos_years++;
            sum_ann += ann;
        }
        cout << "  正收益年份: " << pos_years << "/" << yearly_returns.size()
             << " (" << fixed << setprecision(1) << (100.0 * pos_years / yearly_returns.size()) << "%)\n";
        cout << "  年化均值: " << fixed << setprecision(2) << (sum_ann / yearly_returns.size()) << "%\n";
    }
    
    cout << "\n>>> 对比 <<<\n";
    cout << "  全样本最优: 49.95% 年化, 13.30% 回撤, Sharpe 2.52, Calmar 3.76\n";
    if (!wf_results.empty()) {
        vector<double> ta;
        for (const auto& wr : wf_results) ta.push_back(wr.test_annualized);
        sort(ta.begin(), ta.end());
        double med = ta.size() % 2 == 0 ? (ta[ta.size()/2-1] + ta[ta.size()/2]) / 2.0 : ta[ta.size()/2];
        cout << "  Walk-Forward测试中位数: " << fixed << setprecision(2) << med << "% 年化\n";
        double wf_efficiency = med / 49.95 * 100.0;
        cout << "  WF效率 (中位数/全样本): " << fixed << setprecision(1) << wf_efficiency << "%\n";
        
        // Verdict
        if (med > 20.0) {
            cout << "\n  ✅ Walk-Forward验证通过: 测试中位数 > 20% 年化\n";
        } else if (med > 10.0) {
            cout << "\n  ⚠️ Walk-Forward部分通过: 测试中位数 10-20%，策略有一定前瞻偏差\n";
        } else if (med > 0.0) {
            cout << "\n  ⚠️ Walk-Forward勉强通过: 测试中位数 > 0 但远低于全样本\n";
        } else {
            cout << "\n  ❌ Walk-Forward未通过: 测试中位数 < 0，策略过拟合\n";
        }
    }
    
    // Save CSV
    ofstream csv("all_etf_rotation_wf_results.csv");
    csv << "window,train_start,train_end,test_start,test_end,m_fast,m_slow,m_signal,buy_signal,sell_signal,w_confirm,trailing_stop,take_profit,max_positions,train_annualized,train_mdd,train_sharpe,train_calmar,train_trades,test_annualized,test_mdd,test_sharpe,test_calmar,test_trades,test_win_rate\n";
    for (int i = 0; i < (int)wf_results.size(); ++i) {
        const auto& wr = wf_results[i];
        csv << "W" << i+1 << "," << wr.train_start << "," << wr.train_end
            << "," << wr.test_start << "," << wr.test_end
            << "," << wr.best_params.m_fast << "," << wr.best_params.m_slow << "," << wr.best_params.m_signal
            << "," << wr.best_params.buy_signal << "," << wr.best_params.sell_signal << "," << wr.best_params.w_confirm
            << "," << wr.best_params.trailing_stop << "," << wr.best_params.take_profit << "," << wr.best_params.max_positions
            << "," << fixed << setprecision(4) << wr.train_annualized << "," << wr.train_mdd * 100
            << "," << wr.train_sharpe << "," << wr.train_calmar << "," << wr.train_trades
            << "," << wr.test_annualized << "," << wr.test_mdd * 100
            << "," << wr.test_sharpe << "," << wr.test_calmar << "," << wr.test_trades
            << "," << wr.test_win_rate << "\n";
    }
    csv.close();
    
    cout << "\n结果已保存到 all_etf_rotation_wf_results.csv\n";
    
    return 0;
}
