/**
 * 每日信号工具 (Daily Signal Tool)
 * 
 * 基于全ETF MACD轮动策略的最优参数，每天运行一次，输出当日买卖信号。
 * 
 * 功能:
 *   1. 从 market_data_qfq/ 加载所有ETF的历史CSV数据
 *   2. 通过腾讯财经API更新到最新交易日的数据
 *   3. 运行策略模拟到今天，确定当前持仓状态
 *   4. 输出今日买入/卖出信号 + 当前持仓一览
 * 
 * 策略参数 (已验证最优):
 *   月线MACD: M(6,15,3) buy_signal=2 sell_signal=2
 *   周线: W(8,30,3) w_confirm=0
 *   风控: 移动止损=0% 止盈=30% 最大持仓=3
 *   优化: 同类去重=YES
 * 
 * 用法:
 *   ./daily_signal              # 使用缓存数据 + 自动更新
 *   ./daily_signal --no-update  # 仅使用缓存数据，不联网更新
 *   ./daily_signal --update-all # 更新所有ETF数据到最新
 * 
 * 编译: g++ -O3 -std=c++17 -o daily_signal daily_signal.cpp -lpthread
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
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <array>

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Data Structures (from all_etf_rotation.cpp)
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

struct Position {
    int etf_idx;
    double entry_price;
    double highest_price;
    string entry_date;
    double shares;
    int entry_daily_idx;
};

struct Trade {
    string etf_code, etf_name;
    string entry_date, exit_date;
    double entry_price, exit_price;
    double pnl_pct;
    int hold_days;
};

struct Signal {
    string type;      // "BUY" or "SELL"
    string etf_code;
    string etf_name;
    string category;
    double ref_price;  // reference price (latest close)
    string reason;
    double strength;   // signal strength for ranking
    // For sell signals
    double entry_price;
    string entry_date;
    double unrealized_pnl_pct;
};

// ============================================================
// Globals
// ============================================================

vector<string> global_dates;
unordered_map<string, int> g_date_to_idx;
unordered_map<string, string> etf_category_map;
unordered_map<string, string> etf_sector_map;

// Strategy params (proven best)
struct {
    int m_fast = 6, m_slow = 15, m_signal = 3;
    int buy_signal = 2;   // MACD hist turns positive
    int sell_signal = 2;  // DIF < 0
    int w_fast = 8, w_slow = 30, w_signal = 3;
    int w_confirm = 0;    // no weekly confirmation
    double trailing_stop = 0;
    double take_profit = 30;
    int max_positions = 3;
    double max_5d_rally = 100.0;  // no anti-chase filter
    bool dedup_category = true;   // same-category dedup (best optimization)
} PARAMS;

// ============================================================
// Utility: Run shell command and capture output
// ============================================================

string exec_cmd(const string& cmd) {
    array<char, 4096> buffer;
    string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

// ============================================================
// Get today's date as YYYY-MM-DD
// ============================================================

string get_today() {
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_buf);
    return string(buf);
}

// ============================================================
// Load ETF names from cache
// ============================================================

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

// ============================================================
// Load category map
// ============================================================

void load_category_map(const string& json_path) {
    ifstream file(json_path);
    if (!file.is_open()) {
        cerr << "警告: 无法打开ETF分类映射: " << json_path << "\n";
        return;
    }
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    size_t pos = 0;
    while (pos < content.size()) {
        size_t code_start = content.find('"', pos);
        if (code_start == string::npos) break;
        size_t code_end = content.find('"', code_start + 1);
        if (code_end == string::npos) break;
        string code = content.substr(code_start + 1, code_end - code_start - 1);
        
        size_t obj_start = content.find('{', code_end);
        if (obj_start == string::npos) break;
        size_t obj_end = content.find('}', obj_start);
        if (obj_end == string::npos) break;
        
        string obj_str = content.substr(obj_start, obj_end - obj_start);
        
        // Parse category
        size_t cat_key = obj_str.find("\"category\"");
        if (cat_key != string::npos) {
            size_t cat_colon = obj_str.find(':', cat_key);
            if (cat_colon != string::npos) {
                size_t cat_start = obj_str.find('"', cat_colon);
                if (cat_start != string::npos) {
                    size_t cat_end = obj_str.find('"', cat_start + 1);
                    if (cat_end != string::npos) {
                        etf_category_map[code] = obj_str.substr(cat_start + 1, cat_end - cat_start - 1);
                    }
                }
            }
        }
        
        // Parse sector
        size_t sec_key = obj_str.find("\"sector\"");
        if (sec_key != string::npos) {
            size_t sec_colon = obj_str.find(':', sec_key);
            if (sec_colon != string::npos) {
                size_t sec_start = obj_str.find('"', sec_colon);
                if (sec_start != string::npos) {
                    size_t sec_end = obj_str.find('"', sec_start + 1);
                    if (sec_end != string::npos) {
                        etf_sector_map[code] = obj_str.substr(sec_start + 1, sec_end - sec_start - 1);
                    }
                }
            }
        }
        
        pos = obj_end + 1;
    }
}

// ============================================================
// Update ETF data from Tencent API
// ============================================================

// Parse Tencent API qfqday response and return new daily bars
// Tencent column order: [date, open, CLOSE, high, low, volume]
vector<DailyBar> parse_tencent_bars(const string& json_str) {
    vector<DailyBar> bars;
    
    // Find qfqday array
    size_t key_pos = json_str.find("\"qfqday\"");
    if (key_pos == string::npos) {
        // Try "day" key as fallback
        key_pos = json_str.find("\"day\"");
    }
    if (key_pos == string::npos) return bars;
    
    size_t arr_start = json_str.find('[', key_pos);
    if (arr_start == string::npos) return bars;
    
    // Parse each sub-array: ["2025-03-03","3.784","3.770","3.816","3.755","11457686.000"]
    size_t pos = arr_start;
    while (pos < json_str.size()) {
        size_t sub_start = json_str.find('[', pos + 1);
        if (sub_start == string::npos) break;
        
        // Check if we hit the end of the outer array
        size_t next_close = json_str.find(']', pos + 1);
        if (next_close != string::npos && next_close < sub_start) break;
        
        size_t sub_end = json_str.find(']', sub_start);
        if (sub_end == string::npos) break;
        
        string sub = json_str.substr(sub_start + 1, sub_end - sub_start - 1);
        
        // Parse 6 comma-separated values
        vector<string> vals;
        stringstream ss(sub);
        string token;
        while (getline(ss, token, ',')) {
            // Remove quotes
            size_t q1 = token.find('"');
            size_t q2 = token.rfind('"');
            if (q1 != string::npos && q2 != string::npos && q2 > q1) {
                token = token.substr(q1 + 1, q2 - q1 - 1);
            }
            vals.push_back(token);
        }
        
        if (vals.size() >= 6) {
            try {
                DailyBar bar;
                bar.date = vals[0];
                bar.open = stod(vals[1]);
                bar.close = stod(vals[2]); // Tencent: close is index 2
                bar.high = stod(vals[3]);
                bar.low = stod(vals[4]);
                bar.volume = stod(vals[5]);
                bars.push_back(bar);
            } catch (...) {}
        }
        
        pos = sub_end;
    }
    
    return bars;
}

// Update a single ETF's data from Tencent API
int update_etf_data(const string& code, const string& data_dir, const string& last_date) {
    string today = get_today();
    if (last_date >= today) return 0; // already up to date
    
    // Calculate start date: day after last_date
    // Simple approach: just request from last_date (API will include it, we'll skip duplicates)
    string url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=" 
                 + code + ",day," + last_date + "," + today + ",500,qfq";
    
    string cmd = "curl -s --connect-timeout 5 --max-time 10 '" + url + "' 2>/dev/null";
    string response = exec_cmd(cmd);
    
    if (response.empty()) return -1;
    
    auto new_bars = parse_tencent_bars(response);
    if (new_bars.empty()) return 0;
    
    // Filter out bars we already have (date <= last_date)
    vector<DailyBar> truly_new;
    for (const auto& bar : new_bars) {
        if (bar.date > last_date) {
            truly_new.push_back(bar);
        }
    }
    
    if (truly_new.empty()) return 0;
    
    // Append to CSV file
    string csv_path = data_dir + code + ".csv";
    ofstream file(csv_path, ios::app);
    if (!file.is_open()) return -1;
    
    for (const auto& bar : truly_new) {
        file << bar.date << "," << fixed << setprecision(3) 
             << bar.open << "," << bar.high << "," << bar.low << "," 
             << bar.close << "," << setprecision(1) << bar.volume << "\n";
    }
    file.close();
    
    return truly_new.size();
}

// ============================================================
// ETF Discovery and Loading
// ============================================================

bool is_index_file(const string& code) {
    if (code.size() < 8) return false;
    string prefix = code.substr(0, 2);
    string num = code.substr(2);
    if (prefix == "sh" && num.size() == 6 && num[0] == '0' && num[1] == '0' && num[2] == '0') return true;
    if (prefix == "sz" && num.size() == 6 && num[0] == '3' && num[1] == '9' && num[2] == '9') return true;
    return false;
}

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
    return false;
}

// ============================================================
// Bar Aggregation (from all_etf_rotation.cpp)
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
// MACD Calculation (from all_etf_rotation.cpp)
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
// Strategy Simulation with Signal Detection
// ============================================================

struct SimResult {
    vector<Position> active_positions;
    double cash;
    double equity;
    vector<Trade> trade_list;
    vector<Signal> today_signals; // signals generated on the last day
};

SimResult run_simulation(const vector<ETFData>& etfs) {
    int num_etfs = etfs.size();
    
    // Precompute MACD for all ETFs
    vector<vector<double>> m_dif(num_etfs), m_dea(num_etfs), m_hist(num_etfs);
    vector<vector<double>> w_dif(num_etfs), w_dea(num_etfs), w_hist(num_etfs);
    vector<vector<int>> w_to_m(num_etfs);
    vector<vector<int>> d_to_w(num_etfs);
    vector<vector<int>> d_to_m(num_etfs);
    
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].monthly.size() < 5 || etfs[i].weekly.size() < 10) continue;
        
        compute_macd(etfs[i].monthly, PARAMS.m_fast, PARAMS.m_slow, PARAMS.m_signal, m_dif[i], m_dea[i], m_hist[i]);
        compute_macd(etfs[i].weekly, PARAMS.w_fast, PARAMS.w_slow, PARAMS.w_signal, w_dif[i], w_dea[i], w_hist[i]);
        
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
    SimResult sim;
    
    // Find date range
    int start_idx = (int)global_dates.size(), end_idx = 0;
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].daily.empty()) continue;
        auto it_s = g_date_to_idx.find(etfs[i].daily.front().date);
        auto it_e = g_date_to_idx.find(etfs[i].daily.back().date);
        if (it_s != g_date_to_idx.end()) start_idx = min(start_idx, it_s->second);
        if (it_e != g_date_to_idx.end()) end_idx = max(end_idx, it_e->second);
    }
    
    if (start_idx >= end_idx) {
        sim.cash = cash;
        sim.equity = capital;
        return sim;
    }
    
    for (int d = start_idx; d <= end_idx; ++d) {
        string cur_date = global_dates[d];
        bool is_last_day = (d == end_idx);
        
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
            string sell_reason;
            
            // Monthly MACD sell signal — check at month end
            int mi = d_to_m[e_idx][d];
            if (mi > 0 && mi < (int)m_dif[e_idx].size()) {
                if (d == g_date_to_idx[etf.daily[etf.monthly[mi].last_daily_idx].date]) {
                    if (PARAMS.sell_signal == 2) {
                        if (m_dif[e_idx][mi] < 0) {
                            sell = true;
                            sell_reason = "月线DIF<0 (趋势转空)";
                        }
                    }
                }
            }
            
            // Weekly death cross + monthly DIF declining
            int wi = d_to_w[e_idx][d];
            if (!sell && wi > 0 && wi < (int)w_dif[e_idx].size()) {
                if (d == g_date_to_idx[etf.daily[etf.weekly[wi].last_daily_idx].date]) {
                    if (w_dif[e_idx][wi] < w_dea[e_idx][wi] && w_dif[e_idx][wi-1] >= w_dea[e_idx][wi-1]) {
                        if (mi > 0 && mi < (int)m_dif[e_idx].size() && m_dif[e_idx][mi] < m_dif[e_idx][mi > 0 ? mi-1 : mi]) {
                            sell = true;
                            sell_reason = "周线死叉 + 月线DIF下行";
                        }
                    }
                }
            }
            
            // Take profit
            if (!sell && PARAMS.take_profit > 0) {
                if (cur_close >= pos.entry_price * (1.0 + PARAMS.take_profit / 100.0)) {
                    sell = true;
                    sell_reason = "止盈 (涨幅达" + to_string((int)PARAMS.take_profit) + "%)";
                }
            }
            
            if (sell && !is_last_day) {
                // Signal: sell tomorrow at open
                // In simulation, we execute at next day open
                double exit_price = etf.global_open[d + 1];
                if (exit_price <= 0) exit_price = cur_close;
                
                double value = pos.shares * exit_price;
                cash += value;
                
                // Record trade
                Trade t;
                t.etf_code = etf.code;
                t.etf_name = etf.name;
                t.entry_date = pos.entry_date;
                t.exit_date = global_dates[d + 1];
                t.entry_price = pos.entry_price;
                t.exit_price = exit_price;
                t.pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0;
                t.hold_days = d - pos.entry_daily_idx;
                sim.trade_list.push_back(t);
            } else if (sell && is_last_day) {
                // This is a TODAY sell signal — tomorrow's action
                Signal sig;
                sig.type = "SELL";
                sig.etf_code = etf.code;
                sig.etf_name = etf.name;
                string raw_code = etf.code.size() > 2 ? etf.code.substr(2) : etf.code;
                auto cat_it = etf_category_map.find(raw_code);
                sig.category = (cat_it != etf_category_map.end()) ? cat_it->second : "未知";
                sig.ref_price = cur_close;
                sig.reason = sell_reason;
                sig.entry_price = pos.entry_price;
                sig.entry_date = pos.entry_date;
                sig.unrealized_pnl_pct = (cur_close - pos.entry_price) / pos.entry_price * 100.0;
                sim.today_signals.push_back(sig);
                
                // Don't actually sell yet — keep in positions for display
                remaining_positions.push_back(pos);
            } else {
                remaining_positions.push_back(pos);
            }
        }
        active_positions = remaining_positions;
        
        // Calculate equity
        double daily_equity = cash;
        for (const auto& pos : active_positions) {
            double p = etfs[pos.etf_idx].global_close[d];
            if (p <= 0) p = pos.entry_price;
            daily_equity += pos.shares * p;
        }
        
        // ============ BUY LOGIC ============
        if ((int)active_positions.size() < PARAMS.max_positions) {
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
                
                // Only check at end of week
                const auto& etf = etfs[i];
                if (d != g_date_to_idx[etf.daily[etf.weekly[wi].last_daily_idx].date]) continue;
                
                // Already holding?
                bool already_held = false;
                for (const auto& pos : active_positions) {
                    if (pos.etf_idx == i) { already_held = true; break; }
                }
                if (already_held) continue;
                
                // Monthly MACD buy signal (buy_signal=2: MACD hist turns positive)
                bool m_buy = (m_hist[i][mi] > 0 && m_hist[i][mi-1] <= 0);
                if (!m_buy) continue;
                
                // Anti-chase filter
                if (PARAMS.max_5d_rally < 100.0) {
                    double cur_close_val = etfs[i].global_close[d];
                    double close_5d_ago = 0;
                    int count_back = 0;
                    for (int dd = d - 1; dd >= 0 && count_back < 5; --dd) {
                        double c = etfs[i].global_close[dd];
                        if (c > 0) { close_5d_ago = c; count_back++; }
                    }
                    if (close_5d_ago > 0 && cur_close_val > 0) {
                        double rally_5d = (cur_close_val - close_5d_ago) / close_5d_ago * 100.0;
                        if (rally_5d > PARAMS.max_5d_rally) continue;
                    }
                }
                
                double strength = m_dif[i][mi] - m_dif[i][mi-1];
                candidates.push_back({i, strength});
            }
            
            // Sort by strength
            sort(candidates.begin(), candidates.end(), [](const BuyCandidate& a, const BuyCandidate& b) {
                return a.strength > b.strength;
            });
            
            // Category dedup
            if (PARAMS.dedup_category) {
                unordered_map<string, bool> seen_category;
                for (const auto& pos : active_positions) {
                    string pos_code = etfs[pos.etf_idx].code;
                    if (pos_code.size() > 2) pos_code = pos_code.substr(2);
                    auto cat_it = etf_category_map.find(pos_code);
                    if (cat_it != etf_category_map.end()) {
                        seen_category[cat_it->second] = true;
                    }
                }
                vector<BuyCandidate> deduped;
                for (const auto& cand : candidates) {
                    string cand_code = etfs[cand.etf_idx].code;
                    if (cand_code.size() > 2) cand_code = cand_code.substr(2);
                    auto cat_it = etf_category_map.find(cand_code);
                    string cat = (cat_it != etf_category_map.end()) ? cat_it->second : "";
                    if (cat.empty() || !seen_category.count(cat)) {
                        deduped.push_back(cand);
                        if (!cat.empty()) seen_category[cat] = true;
                    }
                }
                candidates = deduped;
            }
            
            // Execute buys (or generate signals on last day)
            for (const auto& cand : candidates) {
                if ((int)active_positions.size() >= PARAMS.max_positions) break;
                
                int i = cand.etf_idx;
                const auto& etf = etfs[i];
                
                if (is_last_day) {
                    // TODAY buy signal — action tomorrow
                    Signal sig;
                    sig.type = "BUY";
                    sig.etf_code = etf.code;
                    sig.etf_name = etf.name;
                    string raw_code = etf.code.size() > 2 ? etf.code.substr(2) : etf.code;
                    auto cat_it = etf_category_map.find(raw_code);
                    sig.category = (cat_it != etf_category_map.end()) ? cat_it->second : "未知";
                    sig.ref_price = etf.global_close[d];
                    sig.reason = "月线MACD柱由负转正 (行业底部信号)";
                    sig.strength = cand.strength;
                    sim.today_signals.push_back(sig);
                    
                    // Mark position as taken for dedup purposes
                    Position p;
                    p.etf_idx = i;
                    p.entry_price = etf.global_close[d];
                    p.highest_price = etf.global_close[d];
                    p.entry_date = cur_date;
                    p.shares = 0; // placeholder
                    p.entry_daily_idx = d;
                    active_positions.push_back(p);
                } else {
                    double entry_price = etf.global_close[d];
                    if (d < end_idx) {
                        double nxt_open = etf.global_open[d + 1];
                        if (nxt_open > 0) entry_price = nxt_open;
                    }
                    if (entry_price <= 0) continue;
                    
                    double alloc = daily_equity / PARAMS.max_positions;
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
    }
    
    // Remove placeholder positions (buy signals that haven't been executed yet)
    vector<Position> real_positions;
    for (const auto& pos : active_positions) {
        if (pos.shares > 0) real_positions.push_back(pos);
    }
    
    sim.active_positions = real_positions;
    sim.cash = cash;
    
    // Calculate final equity
    sim.equity = cash;
    int end_d = end_idx;
    for (const auto& pos : real_positions) {
        double p = etfs[pos.etf_idx].global_close[end_d];
        if (p <= 0) p = pos.entry_price;
        sim.equity += pos.shares * p;
    }
    
    return sim;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    bool do_update = true;
    bool update_all = false;
    
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--no-update") do_update = false;
        else if (arg == "--update-all") update_all = true;
        else if (arg == "--help" || arg == "-h") {
            cout << "用法: ./daily_signal [选项]\n";
            cout << "  --no-update   不更新数据，仅使用缓存CSV\n";
            cout << "  --update-all  强制更新所有ETF数据\n";
            cout << "  --help        显示帮助\n";
            return 0;
        }
    }
    
    string data_dir = "/ceph/dang_articles/yoj/market_data_qfq/";
    string cache_path = "/ceph/dang_articles/yoj/etf_list_cache.json";
    string category_path = "/ceph/dang_articles/yoj/etf_category_map.json";
    
    string today = get_today();
    
    cout << "╔════════════════════════════════════════════════════════════╗\n";
    cout << "║           ETF轮动策略 — 每日信号工具                     ║\n";
    cout << "╠════════════════════════════════════════════════════════════╣\n";
    cout << "║  策略: 月线MACD M(6,15,3) 柱转正买入 / DIF<0卖出        ║\n";
    cout << "║  优化: 同类去重 | 止盈30% | 最大3仓                     ║\n";
    cout << "║  验证: 年化42.11% 回撤14.17% Sharpe 2.41               ║\n";
    cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    cout << "  当前日期: " << today << "\n\n";
    
    // Load ETF names
    auto name_map = load_etf_names(cache_path);
    cout << "  加载ETF名称缓存: " << name_map.size() << " 条\n";
    
    // Load category map
    load_category_map(category_path);
    cout << "  加载ETF分类映射: " << etf_category_map.size() << " 条\n\n";
    
    // Discover ETFs
    vector<pair<string, string>> etf_list;
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
        etf_list.push_back({code, name});
    }
    sort(etf_list.begin(), etf_list.end());
    cout << "  发现ETF文件: " << etf_list.size() << " 只\n";

    // ============================================================
    // Load all ETF data FIRST (to know which ones pass filters)
    // ============================================================
    cout << "\n  加载ETF数据...\n";
    vector<ETFData> etfs;
    vector<pair<int, string>> etfs_needing_update; // {index in etfs, last_date}
    
    for (const auto& [code, name] : etf_list) {
        string path = data_dir + code + ".csv";
        ifstream file(path);
        if (!file.is_open()) continue;
        
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
        
        if (etf.daily.size() < 60) continue;
        
        // Liquidity filter: avg daily turnover >= 1亿
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
        if (is_non_equity(name)) continue;
        
        // Track if this ETF needs updating
        if (do_update && !etf.daily.empty()) {
            string last_date = etf.daily.back().date;
            if (last_date < today) {
                etfs_needing_update.push_back({(int)etfs.size(), last_date});
            }
        }
        
        etfs.push_back(etf);
    }
    
    cout << "  已加载 " << etfs.size() << " 只ETF (已过滤)\n";
    
    // ============================================================
    // Update ONLY filtered ETFs from Tencent API
    // ============================================================
    if (do_update && !etfs_needing_update.empty()) {
        cout << "\n  正在更新 " << etfs_needing_update.size() << " 只ETF数据...\n";
        int updated = 0, failed = 0;
        
        for (const auto& [idx, last_date] : etfs_needing_update) {
            auto& etf = etfs[idx];
            int result = update_etf_data(etf.code, data_dir, last_date);
            if (result > 0) {
                // Reload this ETF's data from the updated CSV
                string path = data_dir + etf.code + ".csv";
                ifstream file(path);
                if (file.is_open()) {
                    etf.daily.clear();
                    string line;
                    getline(file, line); // header
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
                }
                updated++;
                if (updated % 50 == 0) {
                    cout << "    已更新 " << updated << "/" << etfs_needing_update.size() << "...\n";
                }
                this_thread::sleep_for(chrono::milliseconds(20));
            } else if (result < 0) {
                failed++;
            }
        }
        
        cout << "  更新完成: " << updated << " 只更新, " << failed << " 只失败\n";
    } else if (do_update) {
        cout << "\n  所有ETF数据已是最新\n";
    }

    // Aggregate bars (must be done after data update)
    for (auto& etf : etfs) {
        etf.monthly = aggregate_monthly(etf.daily, 0, etf.daily.size());
        etf.weekly = aggregate_weekly(etf.daily, 0, etf.daily.size());
    }
    
    cout << "\n  ETF池大小: " << etfs.size() << " 只\n\n";
    
    if (etfs.empty()) {
        cerr << "错误: 没有加载到任何ETF数据!\n";
        return 1;
    }
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
        for (int i = 0; i < (int)global_dates.size(); ++i) {
            if (etf.global_close[i] > 0) {
                last_close = etf.global_close[i];
            } else {
                etf.global_close[i] = last_close;
            }
        }
    }
    
    // ============================================================
    // Run simulation
    // ============================================================
    cout << "  运行策略模拟...\n\n";
    auto sim = run_simulation(etfs);
    
    // ============================================================
    // Output: Current date context
    // ============================================================
    
    // Determine what kind of day today is
    string last_global_date = global_dates.back();
    
    // Check if last day is end of week
    auto get_weekday = [](const string& date) -> int {
        int y = stoi(date.substr(0, 4));
        int m = stoi(date.substr(5, 2));
        int d = stoi(date.substr(8, 2));
        if (m <= 2) { y--; m += 12; }
        return (d + 13*(m+1)/5 + y + y/4 - y/100 + y/400) % 7; // 0=Sat, 6=Fri
    };
    
    int last_wd = get_weekday(last_global_date);
    // In the Zeller formula: 0=Sat, 1=Sun, 2=Mon, 3=Tue, 4=Wed, 5=Thu, 6=Fri
    string day_names[] = {"六", "日", "一", "二", "三", "四", "五"};
    
    string last_month = last_global_date.substr(0, 7);
    
    cout << "══════════════════════════════════════════════════════════════\n";
    cout << "                    当日信号报告\n";
    cout << "══════════════════════════════════════════════════════════════\n\n";
    cout << "  数据截至: " << last_global_date << " (周" << day_names[last_wd] << ")\n";
    cout << "  当月: " << last_month << "\n\n";
    
    // ============================================================
    // Output: Current Portfolio
    // ============================================================
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    cout << "  【当前持仓】\n";
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
    
    int end_d = (int)global_dates.size() - 1;
    
    if (sim.active_positions.empty()) {
        cout << "  （空仓）\n\n";
    } else {
        double total_position_value = 0;
        cout << "  " << setw(16) << left << "ETF名称" 
             << setw(12) << "ETF代码"
             << setw(16) << "分类"
             << setw(12) << "买入日期" 
             << setw(10) << right << "买入价" 
             << setw(10) << "现价" 
             << setw(10) << "持仓市值"
             << setw(10) << "浮动盈亏" 
             << setw(8) << "持有天" << "\n";
        cout << "  " << string(104, '-') << "\n";
        
        for (const auto& pos : sim.active_positions) {
            const auto& etf = etfs[pos.etf_idx];
            double cur_price = etf.global_close[end_d];
            if (cur_price <= 0) cur_price = pos.entry_price;
            double pnl = (cur_price - pos.entry_price) / pos.entry_price * 100.0;
            double value = pos.shares * cur_price;
            total_position_value += value;
            
            string raw_code = etf.code.size() > 2 ? etf.code.substr(2) : etf.code;
            string cat = "未知";
            auto cat_it = etf_category_map.find(raw_code);
            if (cat_it != etf_category_map.end()) cat = cat_it->second;
            
            // Trim category for display
            if (cat.size() > 16) cat = cat.substr(0, 14) + "..";
            
            int hold_days = end_d - pos.entry_daily_idx;
            
            cout << "  " << setw(16) << left << etf.name 
                 << setw(12) << etf.code
                 << setw(16) << cat
                 << setw(12) << pos.entry_date
                 << setw(10) << right << fixed << setprecision(4) << pos.entry_price
                 << setw(10) << cur_price
                 << setw(10) << setprecision(0) << value
                 << setw(9) << setprecision(2) << pnl << "%"
                 << setw(7) << hold_days << "\n";
        }
        
        cout << "\n  总持仓市值: " << fixed << setprecision(0) << total_position_value << " 元\n";
        cout << "  可用现金:   " << sim.cash << " 元\n";
        cout << "  账户总值:   " << sim.equity << " 元\n";
        cout << "  总收益率:   " << fixed << setprecision(2) << ((sim.equity / 1000000.0 - 1.0) * 100.0) << "%\n\n";
    }
    
    int available_slots = PARAMS.max_positions - (int)sim.active_positions.size();
    
    // ============================================================
    // Output: Today's Signals
    // ============================================================
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    cout << "  【今日信号】\n";
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
    
    // Separate buy and sell signals
    vector<Signal> buy_signals, sell_signals;
    for (const auto& sig : sim.today_signals) {
        if (sig.type == "BUY") buy_signals.push_back(sig);
        else sell_signals.push_back(sig);
    }
    
    // Sell signals
    if (!sell_signals.empty()) {
        cout << "  🔴 卖出信号 (明日开盘执行):\n\n";
        for (const auto& sig : sell_signals) {
            cout << "    ▶ " << sig.etf_name << " (" << sig.etf_code << ")\n";
            cout << "      分类: " << sig.category << "\n";
            cout << "      原因: " << sig.reason << "\n";
            cout << "      买入价: " << fixed << setprecision(4) << sig.entry_price 
                 << "  买入日: " << sig.entry_date << "\n";
            cout << "      现价: " << sig.ref_price 
                 << "  浮动盈亏: " << setprecision(2) << sig.unrealized_pnl_pct << "%\n";
            cout << "      → 建议: 明日开盘价附近卖出\n\n";
        }
    }
    
    // Buy signals
    if (!buy_signals.empty()) {
        cout << "  🟢 买入信号 (明日开盘执行):\n\n";
        
        int slot_idx = 0;
        for (const auto& sig : buy_signals) {
            slot_idx++;
            string status = (slot_idx <= available_slots) ? "✅ 可执行" : "⚠️ 仓位已满,备选";
            
            cout << "    ▶ " << sig.etf_name << " (" << sig.etf_code << ") [" << status << "]\n";
            cout << "      分类: " << sig.category << "\n";
            cout << "      原因: " << sig.reason << "\n";
            cout << "      参考价: " << fixed << setprecision(4) << sig.ref_price 
                 << " (昨收)\n";
            cout << "      信号强度: " << setprecision(4) << sig.strength << "\n";
            
            if (slot_idx <= available_slots) {
                double alloc = sim.equity / PARAMS.max_positions;
                int est_shares = (int)(alloc / sig.ref_price / 100) * 100; // round to 100
                cout << "      建议仓位: 约 " << fixed << setprecision(0) << alloc << " 元"
                     << " (约 " << est_shares << " 股)\n";
            }
            cout << "\n";
        }
    }
    
    if (buy_signals.empty() && sell_signals.empty()) {
        cout << "  📊 今日无信号\n\n";
        
        // Show context about when next signal might come
        cout << "  说明:\n";
        cout << "    - 买入信号在每周最后一个交易日检查 (月线MACD柱转正)\n";
        cout << "    - 卖出信号在每月最后一个交易日检查 (月线DIF<0) 或每周检查 (周线死叉+月线下行)\n";
        cout << "    - 止盈信号每日检查 (涨幅达30%)\n";
        cout << "    - 可用仓位: " << available_slots << "/" << PARAMS.max_positions << "\n\n";
    }
    
    // ============================================================
    // Output: Recent trades (last 10)
    // ============================================================
    if (!sim.trade_list.empty()) {
        cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        cout << "  【近期交易记录】 (最近10笔)\n";
        cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        
        cout << "  " << setw(16) << left << "ETF" 
             << setw(12) << "买入日期" << setw(12) << "卖出日期"
             << setw(10) << right << "买入价" << setw(10) << "卖出价" 
             << setw(10) << "收益%" << setw(8) << "天数" << "\n";
        cout << "  " << string(78, '-') << "\n";
        
        int start = max(0, (int)sim.trade_list.size() - 10);
        for (int i = start; i < (int)sim.trade_list.size(); ++i) {
            const auto& t = sim.trade_list[i];
            cout << "  " << setw(16) << left << t.etf_name 
                 << setw(12) << t.entry_date << setw(12) << t.exit_date
                 << setw(10) << right << fixed << setprecision(4) << t.entry_price 
                 << setw(10) << t.exit_price
                 << setw(9) << setprecision(2) << t.pnl_pct << "%"
                 << setw(7) << t.hold_days << "\n";
        }
        
        // Summary stats
        int total_trades = sim.trade_list.size();
        int wins = 0;
        double total_pnl = 0;
        for (const auto& t : sim.trade_list) {
            if (t.pnl_pct > 0) wins++;
            total_pnl += t.pnl_pct;
        }
        
        cout << "\n  总交易: " << total_trades << " 笔"
             << "  胜率: " << fixed << setprecision(1) << (100.0 * wins / total_trades) << "%"
             << "  累计收益: " << setprecision(2) << total_pnl << "%\n\n";
    }
    
    // ============================================================
    // Output: Market context (MACD status of current positions)
    // ============================================================
    if (!sim.active_positions.empty()) {
        cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        cout << "  【持仓MACD状态】\n";
        cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        
        for (const auto& pos : sim.active_positions) {
            const auto& etf = etfs[pos.etf_idx];
            
            // Compute MACD for display
            vector<double> m_dif, m_dea, m_hist;
            vector<double> w_dif, w_dea, w_hist;
            compute_macd(etf.monthly, PARAMS.m_fast, PARAMS.m_slow, PARAMS.m_signal, m_dif, m_dea, m_hist);
            compute_macd(etf.weekly, PARAMS.w_fast, PARAMS.w_slow, PARAMS.w_signal, w_dif, w_dea, w_hist);
            
            cout << "  " << etf.name << " (" << etf.code << "):\n";
            
            if (!m_dif.empty()) {
                int mi = m_dif.size() - 1;
                cout << "    月线 DIF=" << fixed << setprecision(4) << m_dif[mi] 
                     << " DEA=" << m_dea[mi]
                     << " MACD柱=" << m_hist[mi];
                if (m_dif[mi] > 0) cout << " ⬆️ DIF>0";
                else cout << " ⚠️ DIF<0";
                if (m_hist[mi] > 0) cout << " 柱>0";
                else cout << " 柱<0";
                cout << "\n";
            }
            if (!w_dif.empty()) {
                int wi = w_dif.size() - 1;
                cout << "    周线 DIF=" << fixed << setprecision(4) << w_dif[wi]
                     << " DEA=" << w_dea[wi]
                     << " MACD柱=" << w_hist[wi];
                if (w_dif[wi] > w_dea[wi]) cout << " 金叉中";
                else cout << " 死叉中";
                cout << "\n";
            }
            cout << "\n";
        }
    }
    
    // ============================================================
    // Save portfolio state to JSON
    // ============================================================
    {
        string state_path = "/ceph/dang_articles/yoj/stock_data_bfq/portfolio_state.json";
        ofstream state_file(state_path);
        if (state_file.is_open()) {
            state_file << "{\n";
            state_file << "  \"date\": \"" << last_global_date << "\",\n";
            state_file << "  \"cash\": " << fixed << setprecision(2) << sim.cash << ",\n";
            state_file << "  \"equity\": " << sim.equity << ",\n";
            state_file << "  \"positions\": [\n";
            for (size_t i = 0; i < sim.active_positions.size(); ++i) {
                const auto& pos = sim.active_positions[i];
                const auto& etf = etfs[pos.etf_idx];
                double cur_price = etf.global_close[end_d];
                state_file << "    {\n";
                state_file << "      \"code\": \"" << etf.code << "\",\n";
                state_file << "      \"name\": \"" << etf.name << "\",\n";
                state_file << "      \"entry_date\": \"" << pos.entry_date << "\",\n";
                state_file << "      \"entry_price\": " << setprecision(4) << pos.entry_price << ",\n";
                state_file << "      \"current_price\": " << cur_price << ",\n";
                state_file << "      \"shares\": " << setprecision(2) << pos.shares << ",\n";
                state_file << "      \"pnl_pct\": " << setprecision(2) << ((cur_price - pos.entry_price) / pos.entry_price * 100.0) << "\n";
                state_file << "    }" << (i + 1 < sim.active_positions.size() ? "," : "") << "\n";
            }
            state_file << "  ],\n";
            state_file << "  \"signals\": [\n";
            for (size_t i = 0; i < sim.today_signals.size(); ++i) {
                const auto& sig = sim.today_signals[i];
                state_file << "    {\n";
                state_file << "      \"type\": \"" << sig.type << "\",\n";
                state_file << "      \"code\": \"" << sig.etf_code << "\",\n";
                state_file << "      \"name\": \"" << sig.etf_name << "\",\n";
                state_file << "      \"category\": \"" << sig.category << "\",\n";
                state_file << "      \"ref_price\": " << setprecision(4) << sig.ref_price << ",\n";
                state_file << "      \"reason\": \"" << sig.reason << "\"\n";
                state_file << "    }" << (i + 1 < sim.today_signals.size() ? "," : "") << "\n";
            }
            state_file << "  ]\n";
            state_file << "}\n";
            state_file.close();
            cout << "  持仓状态已保存: " << state_path << "\n";
        }
    }
    
    cout << "\n══════════════════════════════════════════════════════════════\n";
    cout << "  提示: 信号基于收盘数据，操作在次日开盘执行\n";
    cout << "══════════════════════════════════════════════════════════════\n";
    
    return 0;
}
