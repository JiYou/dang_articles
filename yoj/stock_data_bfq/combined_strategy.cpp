/**
 * 组合策略: 宏观择时 + MACD行业轮动
 * 
 * 将中信+纳指宏观择时策略与全ETF MACD轮动策略组合：
 * 
 *   1. 上证 < 2900 → 全仓中信证券 (低位抄底)
 *   2. 上证 > 3200 → 全仓纳指ETF (高位避险)
 *   3. 2900 ≤ 上证 ≤ 3200 → MACD月线/周线轮动 (正常行情)
 * 
 * 判断使用昨日上证收盘价，交易使用今日开盘价。
 * 状态切换时先清空旧持仓，再建立新持仓。
 * 
 * 数据源:
 *   - 上证指数: market_data/sh000001.csv (不需复权)
 *   - 中信证券: stock_data_bfq/600030.csv (date,code,open,close,high,low,volume)
 *   - 纳指ETF:  market_data_qfq/sh513100.csv (前复权)
 *   - 全ETF池:  market_data_qfq/ (前复权, 300只权益类)
 * 
 * 编译: g++ -O3 -std=c++17 -o combined_strategy combined_strategy.cpp -lpthread
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
    int buy_signal;
    int sell_signal;
    int w_fast, w_slow, w_signal;
    int w_confirm;
    double trailing_stop;
    double take_profit;
    int max_positions;
};

struct Trade {
    string asset_code, asset_name;
    string entry_date, exit_date;
    double entry_price, exit_price;
    double pnl_pct;
    int hold_days;
    string reason;  // "宏观低位", "宏观高位", "MACD轮动", "区间切换"
};

struct Position {
    int etf_idx;       // -1 for 中信/纳指 macro positions
    string code, name;
    double entry_price;
    double highest_price;
    string entry_date;
    double shares;
    int entry_daily_idx;
};

// ============================================================
// ETF Discovery & Filtering (from all_etf_rotation.cpp)
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
    // 黄金/白银ETF保留在池中(用户要求)
    // 原黄金/上海金/金ETF排除规则已移除
    return false;
}

// ============================================================
// CSV Loaders
// ============================================================

// market_data format: date,open,high,low,close,volume
bool load_market_csv(const string& path, vector<DailyBar>& bars) {
    ifstream file(path);
    if (!file.is_open()) return false;
    string line;
    getline(file, line); // header
    bars.clear();
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string d, o, h, l, c, v;
        getline(ss, d, ',');
        getline(ss, o, ',');
        getline(ss, h, ',');
        getline(ss, l, ',');
        getline(ss, c, ',');
        getline(ss, v, ',');
        try {
            DailyBar b;
            b.date = d; b.open = stod(o); b.high = stod(h);
            b.low = stod(l); b.close = stod(c); b.volume = stod(v);
            if (b.close > 0) bars.push_back(b);
        } catch (...) {}
    }
    sort(bars.begin(), bars.end(), [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });
    return !bars.empty();
}

// stock_data_bfq format: date,code,open,close,high,low,volume,...
bool load_stock_csv(const string& path, vector<DailyBar>& bars) {
    ifstream file(path);
    if (!file.is_open()) return false;
    string line;
    getline(file, line); // header
    bars.clear();
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string d, code, o, c, h, l, v;
        getline(ss, d, ',');
        getline(ss, code, ',');
        getline(ss, o, ',');
        getline(ss, c, ',');
        getline(ss, h, ',');
        getline(ss, l, ',');
        getline(ss, v, ',');
        try {
            DailyBar b;
            b.date = d; b.open = stod(o); b.close = stod(c);
            b.high = stod(h); b.low = stod(l); b.volume = stod(v);
            if (b.close > 0) bars.push_back(b);
        } catch (...) {}
    }
    sort(bars.begin(), bars.end(), [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });
    return !bars.empty();
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
// Main
// ============================================================

int main() {
    string data_dir = "/ceph/dang_articles/yoj/market_data/";
    string qfq_dir = "/ceph/dang_articles/yoj/market_data_qfq/";
    string stock_dir = "/ceph/dang_articles/yoj/stock_data_bfq/";
    string cache_path = "/ceph/dang_articles/yoj/etf_list_cache.json";
    
    const double THRESHOLD_LOW = 2900.0;
    const double THRESHOLD_HIGH = 3200.0;
    
    cout << "==========================================================\n";
    cout << "  组合策略: 宏观择时 + MACD行业轮动\n";
    cout << "  上证 < " << THRESHOLD_LOW << " → 中信证券\n";
    cout << "  上证 > " << THRESHOLD_HIGH << " → 纳指ETF\n";
    cout << "  其他 → MACD月线轮动 (300只权益类ETF)\n";
    cout << "==========================================================\n\n";
    
    // ============================================================
    // 1. Load macro data
    // ============================================================
    
    vector<DailyBar> index_bars, citic_bars, nasdaq_bars;
    
    cout << "加载宏观数据...\n";
    if (!load_market_csv(data_dir + "sh000001.csv", index_bars)) {
        cerr << "错误: 无法加载上证指数\n"; return 1;
    }
    cout << "  上证指数: " << index_bars.size() << " bars ("
         << index_bars.front().date << " ~ " << index_bars.back().date << ")\n";
    
    if (!load_stock_csv(stock_dir + "600030.csv", citic_bars)) {
        cerr << "错误: 无法加载中信证券\n"; return 1;
    }
    cout << "  中信证券: " << citic_bars.size() << " bars ("
         << citic_bars.front().date << " ~ " << citic_bars.back().date << ")\n";
    
    if (!load_market_csv(qfq_dir + "sh513100.csv", nasdaq_bars)) {
        cerr << "错误: 无法加载纳指ETF\n"; return 1;
    }
    cout << "  纳指ETF(513100前复权): " << nasdaq_bars.size() << " bars ("
         << nasdaq_bars.front().date << " ~ " << nasdaq_bars.back().date << ")\n\n";
    
    // Build date -> price maps for macro assets
    map<string, double> idx_close_map, citic_open_map, citic_close_map, nasdaq_open_map, nasdaq_close_map;
    for (const auto& b : index_bars) idx_close_map[b.date] = b.close;
    for (const auto& b : citic_bars) { citic_open_map[b.date] = b.open; citic_close_map[b.date] = b.close; }
    for (const auto& b : nasdaq_bars) { nasdaq_open_map[b.date] = b.open; nasdaq_close_map[b.date] = b.close; }
    
    // ============================================================
    // 2. Load ETF rotation pool
    // ============================================================
    
    cout << "加载ETF轮动池...\n";
    auto name_map = load_etf_names(cache_path);
    cout << "  缓存ETF名称: " << name_map.size() << "\n";
    
    vector<ETFData> etfs;
    int skipped_liq = 0, skipped_eq = 0, skipped_data = 0;
    
    for (const auto& entry : fs::directory_iterator(qfq_dir)) {
        if (!entry.is_regular_file()) continue;
        string filename = entry.path().filename().string();
        if (filename.size() < 5 || filename.substr(filename.size() - 4) != ".csv") continue;
        string code = filename.substr(0, filename.size() - 4);
        if (is_index_file(code)) continue;
        
        string numeric_code = code.substr(2);
        string name = numeric_code;
        auto it = name_map.find(numeric_code);
        if (it != name_map.end()) name = it->second;
        
        // Load data
        ifstream file(entry.path());
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
        
        if (etf.daily.size() < 60) { skipped_data++; continue; }
        
        // Liquidity filter
        {
            size_t n = etf.daily.size();
            size_t lookback = min(n, (size_t)60);
            double total_turnover = 0.0;
            for (size_t i = n - lookback; i < n; i++) {
                total_turnover += etf.daily[i].volume * 100.0 * etf.daily[i].close;
            }
            if (total_turnover / lookback < 1e8) { skipped_liq++; continue; }
        }
        
        // Non-equity filter
        if (is_non_equity(name)) { skipped_eq++; continue; }
        
        // Aggregate
        etf.monthly = aggregate_monthly(etf.daily, 0, etf.daily.size());
        etf.weekly = aggregate_weekly(etf.daily, 0, etf.daily.size());
        
        etfs.push_back(etf);
    }
    
    // Sort for deterministic order
    sort(etfs.begin(), etfs.end(), [](const ETFData& a, const ETFData& b) { return a.code < b.code; });
    
    cout << "  已加载 " << etfs.size() << " 只权益类ETF"
         << " (跳过: 数据不足=" << skipped_data
         << " 流动性不足=" << skipped_liq
         << " 非权益=" << skipped_eq << ")\n\n";
    
    // ============================================================
    // 3. Build global date array
    // ============================================================
    
    // Only use dates where index, citic, AND nasdaq all have data
    set<string> all_dates_set;
    for (const auto& b : index_bars) {
        if (citic_close_map.count(b.date) && nasdaq_close_map.count(b.date)) {
            all_dates_set.insert(b.date);
        }
    }
    // Also include ETF dates for mapping
    for (const auto& etf : etfs) {
        for (const auto& b : etf.daily) all_dates_set.insert(b.date);
    }
    
    vector<string> global_dates(all_dates_set.begin(), all_dates_set.end());
    unordered_map<string, int> g_date_to_idx;
    for (int i = 0; i < (int)global_dates.size(); ++i) {
        g_date_to_idx[global_dates[i]] = i;
    }
    
    // Build global price arrays for ETFs
    for (auto& etf : etfs) {
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
        for (int i = 0; i < (int)global_dates.size(); ++i) {
            if (etf.global_close[i] > 0) last_close = etf.global_close[i];
            else etf.global_close[i] = last_close;
        }
    }
    
    cout << "全局日期范围: " << global_dates.front() << " ~ " << global_dates.back()
         << " (" << global_dates.size() << " 个交易日)\n\n";
    
    // ============================================================
    // 4. Precompute MACD for all ETFs (fixed best params)
    // ============================================================
    
    StrategyParams macd_params;
    macd_params.m_fast = 6; macd_params.m_slow = 15; macd_params.m_signal = 3;
    macd_params.buy_signal = 0; macd_params.sell_signal = 2;
    macd_params.w_fast = 8; macd_params.w_slow = 30; macd_params.w_signal = 3;
    macd_params.w_confirm = 0;
    macd_params.trailing_stop = 0.0; macd_params.take_profit = 30.0;
    macd_params.max_positions = 3;
    
    cout << "MACD轮动参数: M(" << macd_params.m_fast << "," << macd_params.m_slow << "," << macd_params.m_signal
         << ") buy=" << macd_params.buy_signal << " sell=" << macd_params.sell_signal
         << " wc=" << macd_params.w_confirm << " ts=" << macd_params.trailing_stop
         << " tp=" << macd_params.take_profit << " mp=" << macd_params.max_positions << "\n\n";
    
    int num_etfs = etfs.size();
    vector<vector<double>> m_dif(num_etfs), m_dea(num_etfs), m_hist(num_etfs);
    vector<vector<double>> w_dif(num_etfs), w_dea(num_etfs), w_hist(num_etfs);
    vector<vector<int>> d_to_w(num_etfs), d_to_m(num_etfs);
    
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].monthly.size() < 5 || etfs[i].weekly.size() < 10) continue;
        
        compute_macd(etfs[i].monthly, macd_params.m_fast, macd_params.m_slow, macd_params.m_signal, m_dif[i], m_dea[i], m_hist[i]);
        compute_macd(etfs[i].weekly, macd_params.w_fast, macd_params.w_slow, macd_params.w_signal, w_dif[i], w_dea[i], w_hist[i]);
        
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
    
    // ============================================================
    // 5. Combined Strategy Backtest
    // ============================================================
    
    cout << "==========================================================\n";
    cout << "  开始回测组合策略\n";
    cout << "==========================================================\n\n";
    
    enum MacroState { STATE_INIT, STATE_CITIC, STATE_NASDAQ, STATE_MACD };
    
    double capital = 1000000.0;
    double cash = capital;
    MacroState state = STATE_INIT;
    
    // Macro position tracking
    double macro_shares = 0;
    double macro_entry_price = 0;
    string macro_entry_date;
    
    // MACD rotation positions
    vector<Position> macd_positions;
    
    vector<Trade> all_trades;
    vector<double> equity_curve;
    
    // Find common start: first date where all three assets have data
    int start_idx = -1, end_idx = -1;
    for (int i = 0; i < (int)global_dates.size(); ++i) {
        const string& d = global_dates[i];
        if (idx_close_map.count(d) && citic_close_map.count(d) && nasdaq_close_map.count(d)) {
            if (start_idx == -1) start_idx = i;
            end_idx = i;
        }
    }
    
    if (start_idx == -1) {
        cerr << "错误: 没有找到三者重叠的日期!\n";
        return 1;
    }
    
    cout << "回测区间: " << global_dates[start_idx] << " ~ " << global_dates[end_idx] << "\n\n";
    
    int state_switches = 0;
    
    for (int d = start_idx; d <= end_idx; ++d) {
        string cur_date = global_dates[d];
        
        // Check if all macro data available
        bool has_macro = idx_close_map.count(cur_date) && citic_close_map.count(cur_date) && nasdaq_close_map.count(cur_date);
        if (!has_macro) {
            // No macro data this day — forward-fill with previous equity value
            if (!equity_curve.empty()) {
                equity_curve.push_back(equity_curve.back());
            } else {
                equity_curve.push_back(capital);
            }
            continue;
        }
        
        double idx_yesterday = (d > start_idx && idx_close_map.count(global_dates[d-1])) 
                               ? idx_close_map[global_dates[d-1]] : 0;
        
        double citic_open_today = citic_open_map.count(cur_date) ? citic_open_map[cur_date] : 0;
        double citic_close_today = citic_close_map[cur_date];
        double nasdaq_open_today = nasdaq_open_map.count(cur_date) ? nasdaq_open_map[cur_date] : 0;
        double nasdaq_close_today = nasdaq_close_map[cur_date];
        
        // ============ DETERMINE TARGET STATE ============
        MacroState target_state = state;
        if (idx_yesterday > 0 && d > start_idx) {
            if (idx_yesterday < THRESHOLD_LOW) {
                target_state = STATE_CITIC;
            } else if (idx_yesterday > THRESHOLD_HIGH) {
                target_state = STATE_NASDAQ;
            } else {
                if (state == STATE_INIT) target_state = STATE_MACD;
                else if (state == STATE_CITIC || state == STATE_NASDAQ) {
                    // Transition from macro to MACD only when leaving macro zone
                    target_state = STATE_MACD;
                }
                // If already in MACD, stay in MACD
            }
        } else if (state == STATE_INIT) {
            // Day 0 or no yesterday data — start with appropriate state based on today
            double idx_today = idx_close_map[cur_date];
            if (idx_today < THRESHOLD_LOW) target_state = STATE_CITIC;
            else if (idx_today > THRESHOLD_HIGH) target_state = STATE_NASDAQ;
            else target_state = STATE_MACD;
        }
        
        // ============ STATE TRANSITION ============
        if (target_state != state && target_state != STATE_INIT) {
            state_switches++;
            
            // Close old positions
            if (state == STATE_CITIC && macro_shares > 0) {
                double exit_price = citic_open_today;
                cash = macro_shares * exit_price;
                Trade t;
                t.asset_code = "600030"; t.asset_name = "中信证券";
                t.entry_date = macro_entry_date; t.exit_date = cur_date;
                t.entry_price = macro_entry_price; t.exit_price = exit_price;
                t.pnl_pct = (exit_price - macro_entry_price) / macro_entry_price * 100.0;
                t.hold_days = d - g_date_to_idx[macro_entry_date];
                t.reason = "区间切换";
                all_trades.push_back(t);
                macro_shares = 0;
            } else if (state == STATE_NASDAQ && macro_shares > 0) {
                double exit_price = nasdaq_open_today;
                cash = macro_shares * exit_price;
                Trade t;
                t.asset_code = "513100"; t.asset_name = "纳指ETF";
                t.entry_date = macro_entry_date; t.exit_date = cur_date;
                t.entry_price = macro_entry_price; t.exit_price = exit_price;
                t.pnl_pct = (exit_price - macro_entry_price) / macro_entry_price * 100.0;
                t.hold_days = d - g_date_to_idx[macro_entry_date];
                t.reason = "区间切换";
                all_trades.push_back(t);
                macro_shares = 0;
            } else if (state == STATE_MACD) {
                // Close all MACD positions
                for (auto& pos : macd_positions) {
                    double exit_price = etfs[pos.etf_idx].global_open[d];
                    if (exit_price <= 0) exit_price = etfs[pos.etf_idx].global_close[d];
                    if (exit_price <= 0) exit_price = pos.entry_price;
                    cash += pos.shares * exit_price;
                    Trade t;
                    t.asset_code = etfs[pos.etf_idx].code; t.asset_name = etfs[pos.etf_idx].name;
                    t.entry_date = pos.entry_date; t.exit_date = cur_date;
                    t.entry_price = pos.entry_price; t.exit_price = exit_price;
                    t.pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0;
                    t.hold_days = d - pos.entry_daily_idx;
                    t.reason = "区间切换";
                    all_trades.push_back(t);
                }
                macd_positions.clear();
            }
            
            // Open new positions
            if (target_state == STATE_CITIC && citic_open_today > 0) {
                macro_entry_price = citic_open_today;
                macro_shares = cash / macro_entry_price;
                macro_entry_date = cur_date;
                cash = 0;
            } else if (target_state == STATE_NASDAQ && nasdaq_open_today > 0) {
                macro_entry_price = nasdaq_open_today;
                macro_shares = cash / macro_entry_price;
                macro_entry_date = cur_date;
                cash = 0;
            }
            // For STATE_MACD: don't buy immediately, wait for MACD signals
            
            state = target_state;
        }
        
        // ============ MACD ROTATION LOGIC (within STATE_MACD) ============
        if (state == STATE_MACD) {
            // SELL check
            vector<Position> remaining;
            for (auto& pos : macd_positions) {
                int e_idx = pos.etf_idx;
                const auto& etf = etfs[e_idx];
                
                double cur_close = etf.global_close[d];
                double cur_high = etf.global_high[d];
                if (cur_close <= 0) { remaining.push_back(pos); continue; }
                if (cur_high > pos.highest_price) pos.highest_price = cur_high;
                
                bool sell = false;
                
                int mi = d_to_m[e_idx][d];
                if (mi > 0 && mi < (int)m_dif[e_idx].size()) {
                    if (d == g_date_to_idx[etf.daily[etf.monthly[mi].last_daily_idx].date]) {
                        if (macd_params.sell_signal == 0) {
                            sell = (m_dif[e_idx][mi] < m_dea[e_idx][mi] && m_dif[e_idx][mi-1] >= m_dea[e_idx][mi-1]);
                        } else if (macd_params.sell_signal == 1) {
                            sell = (m_dif[e_idx][mi] < m_dif[e_idx][mi-1]);
                        } else if (macd_params.sell_signal == 2) {
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
                
                if (!sell && macd_params.trailing_stop > 0) {
                    if (cur_close <= pos.highest_price * (1.0 - macd_params.trailing_stop / 100.0)) sell = true;
                }
                if (!sell && macd_params.take_profit > 0) {
                    if (cur_close >= pos.entry_price * (1.0 + macd_params.take_profit / 100.0)) sell = true;
                }
                
                if (sell) {
                    double exit_price = cur_close;
                    if (d < end_idx) {
                        double nxt_open = etf.global_open[d + 1];
                        if (nxt_open > 0) exit_price = nxt_open;
                    }
                    cash += pos.shares * exit_price;
                    Trade t;
                    t.asset_code = etf.code; t.asset_name = etf.name;
                    t.entry_date = pos.entry_date; t.exit_date = cur_date;
                    t.entry_price = pos.entry_price; t.exit_price = exit_price;
                    t.pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0;
                    t.hold_days = d - pos.entry_daily_idx;
                    t.reason = "MACD卖出";
                    all_trades.push_back(t);
                } else {
                    remaining.push_back(pos);
                }
            }
            macd_positions = remaining;
            
            // BUY check
            double total_equity = cash;
            for (const auto& pos : macd_positions) {
                double p = etfs[pos.etf_idx].global_close[d];
                if (p <= 0) p = pos.entry_price;
                total_equity += pos.shares * p;
            }
            
            if ((int)macd_positions.size() < macd_params.max_positions) {
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
                    if (d != g_date_to_idx[etf.daily[etf.weekly[wi].last_daily_idx].date]) continue;
                    
                    bool already_held = false;
                    for (const auto& pos : macd_positions) {
                        if (pos.etf_idx == i) { already_held = true; break; }
                    }
                    if (already_held) continue;
                    
                    bool m_buy = false;
                    if (macd_params.buy_signal == 0) {
                        m_buy = (m_dif[i][mi] > m_dea[i][mi] && m_dif[i][mi-1] <= m_dea[i][mi-1]);
                    } else if (macd_params.buy_signal == 1) {
                        m_buy = (m_dif[i][mi] > m_dif[i][mi-1]);
                    } else if (macd_params.buy_signal == 2) {
                        m_buy = (m_hist[i][mi] > 0 && m_hist[i][mi-1] <= 0);
                    } else if (macd_params.buy_signal == 3) {
                        m_buy = (m_dif[i][mi] < 0 && m_dif[i][mi] > m_dea[i][mi] && m_dif[i][mi-1] <= m_dea[i][mi-1]);
                    } else if (macd_params.buy_signal == 4) {
                        m_buy = (m_dif[i][mi] > m_dea[i][mi]);
                    }
                    if (!m_buy) continue;
                    
                    bool w_ok = true;
                    if (macd_params.w_confirm == 1) {
                        w_ok = (w_dif[i][wi] > w_dea[i][wi] && w_dif[i][wi-1] <= w_dea[i][wi-1]);
                    } else if (macd_params.w_confirm == 2) {
                        w_ok = (w_dif[i][wi] > 0);
                    } else if (macd_params.w_confirm == 3) {
                        w_ok = (w_dif[i][wi] > w_dif[i][wi-1]);
                    } else if (macd_params.w_confirm == 4) {
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
                    if ((int)macd_positions.size() >= macd_params.max_positions) break;
                    
                    int i = cand.etf_idx;
                    const auto& etf = etfs[i];
                    double entry_price = etf.global_close[d];
                    if (d < end_idx) {
                        double nxt_open = etf.global_open[d + 1];
                        if (nxt_open > 0) entry_price = nxt_open;
                    }
                    if (entry_price <= 0) continue;
                    
                    double alloc = total_equity / macd_params.max_positions;
                    if (alloc > cash) alloc = cash;
                    if (alloc < 100) continue;
                    
                    double shares = alloc / entry_price;
                    cash -= shares * entry_price;
                    
                    Position p;
                    p.etf_idx = i;
                    p.code = etf.code;
                    p.name = etf.name;
                    p.entry_price = entry_price;
                    p.highest_price = entry_price;
                    p.entry_date = (d < end_idx) ? global_dates[d + 1] : cur_date;
                    p.shares = shares;
                    p.entry_daily_idx = d;
                    macd_positions.push_back(p);
                }
            }
        }
        
        // ============ RECORD EQUITY ============
        double equity = cash;
        if (state == STATE_CITIC && macro_shares > 0) {
            equity = macro_shares * citic_close_today;
        } else if (state == STATE_NASDAQ && macro_shares > 0) {
            equity = macro_shares * nasdaq_close_today;
        } else if (state == STATE_MACD) {
            for (const auto& pos : macd_positions) {
                double p = etfs[pos.etf_idx].global_close[d];
                if (p <= 0) p = pos.entry_price;
                equity += pos.shares * p;
            }
        }
        equity_curve.push_back(equity);
    }
    
    // Close final positions
    string last_date = global_dates[end_idx];
    if (state == STATE_CITIC && macro_shares > 0) {
        double exit_price = citic_close_map[last_date];
        cash = macro_shares * exit_price;
        Trade t;
        t.asset_code = "600030"; t.asset_name = "中信证券";
        t.entry_date = macro_entry_date; t.exit_date = last_date;
        t.entry_price = macro_entry_price; t.exit_price = exit_price;
        t.pnl_pct = (exit_price - macro_entry_price) / macro_entry_price * 100.0;
        t.hold_days = end_idx - g_date_to_idx[macro_entry_date];
        t.reason = "回测结束";
        all_trades.push_back(t);
        macro_shares = 0;
    } else if (state == STATE_NASDAQ && macro_shares > 0) {
        double exit_price = nasdaq_close_map[last_date];
        cash = macro_shares * exit_price;
        Trade t;
        t.asset_code = "513100"; t.asset_name = "纳指ETF";
        t.entry_date = macro_entry_date; t.exit_date = last_date;
        t.entry_price = macro_entry_price; t.exit_price = exit_price;
        t.pnl_pct = (exit_price - macro_entry_price) / macro_entry_price * 100.0;
        t.hold_days = end_idx - g_date_to_idx[macro_entry_date];
        t.reason = "回测结束";
        all_trades.push_back(t);
        macro_shares = 0;
    } else if (state == STATE_MACD) {
        for (auto& pos : macd_positions) {
            double exit_price = etfs[pos.etf_idx].global_close[end_idx];
            if (exit_price <= 0) exit_price = pos.entry_price;
            cash += pos.shares * exit_price;
            Trade t;
            t.asset_code = etfs[pos.etf_idx].code; t.asset_name = etfs[pos.etf_idx].name;
            t.entry_date = pos.entry_date; t.exit_date = last_date;
            t.entry_price = pos.entry_price; t.exit_price = exit_price;
            t.pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0;
            t.hold_days = end_idx - pos.entry_daily_idx;
            t.reason = "回测结束";
            all_trades.push_back(t);
        }
        macd_positions.clear();
    }
    
    // ============================================================
    // 6. Calculate Metrics
    // ============================================================
    
    double final_equity = equity_curve.empty() ? capital : equity_curve.back();
    double years = (double)equity_curve.size() / 252.0;
    double ann_ret = pow(final_equity / capital, 1.0 / years) - 1.0;
    
    double max_eq = equity_curve[0];
    double mdd = 0;
    string mdd_peak_date, mdd_trough_date;
    double sum_ret = 0;
    vector<double> daily_rets;
    
    for (size_t i = 1; i < equity_curve.size(); ++i) {
        double r = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1];
        daily_rets.push_back(r);
        sum_ret += r;
        if (equity_curve[i] > max_eq) {
            max_eq = equity_curve[i];
            mdd_peak_date = global_dates[start_idx + i];
        }
        double dd = (max_eq - equity_curve[i]) / max_eq;
        if (dd > mdd) {
            mdd = dd;
            mdd_trough_date = global_dates[start_idx + i];
        }
    }
    
    
    double mean_ret = daily_rets.empty() ? 0 : sum_ret / daily_rets.size();
    double var = 0;
    for (double r : daily_rets) var += (r - mean_ret) * (r - mean_ret);
    if (!daily_rets.empty()) var /= daily_rets.size();
    double std_ret = sqrt(var);
    double ann_vol = std_ret * sqrt(252.0);
    double sharpe = ann_vol > 0 ? (ann_ret - 0.02) / ann_vol : 0;
    double calmar = mdd > 0 ? ann_ret / mdd : 0;
    
    int wins = 0;
    int macro_trades = 0, macd_trades = 0;
    for (const auto& t : all_trades) {
        if (t.pnl_pct > 0) wins++;
        if (t.asset_name == "中信证券" || t.asset_name == "纳指ETF") macro_trades++;
        else macd_trades++;
    }
    double win_rate = all_trades.empty() ? 0 : (double)wins / all_trades.size() * 100.0;
    
    // ============================================================
    // 7. Print Results
    // ============================================================
    
    cout << "\n==========================================================\n";
    cout << "  组合策略回测结果\n";
    cout << "==========================================================\n\n";
    
    cout << "回测区间: " << global_dates[start_idx] << " ~ " << global_dates[end_idx]
         << " (" << fixed << setprecision(1) << years << " 年)\n";
    cout << "初始资金: ¥" << fixed << setprecision(0) << capital << "\n";
    cout << "最终资金: ¥" << final_equity << "\n";
    cout << "累计收益: " << fixed << setprecision(2) << (final_equity / capital - 1.0) * 100 << "%\n";
    cout << "年化收益: " << ann_ret * 100 << "%\n";
    cout << "最大回撤: " << mdd * 100 << "%\n";
    cout << "Sharpe:   " << sharpe << "\n";
    cout << "Calmar:   " << calmar << "\n";
    cout << "总交易:   " << all_trades.size() << " (宏观=" << macro_trades << " MACD=" << macd_trades << ")\n";
    cout << "胜率:     " << win_rate << "%\n";
    cout << "状态切换: " << state_switches << " 次\n";
    
    // Trade details
    cout << "\n--- 交易明细 ---\n";
    cout << setw(4) << "序" << setw(16) << "资产" << setw(12) << "买入日期" << setw(12) << "卖出日期"
         << setw(10) << "买入价" << setw(10) << "卖出价" << setw(10) << "收益%" << setw(8) << "天数"
         << setw(12) << "原因" << "\n";
    cout << string(94, '-') << "\n";
    for (int i = 0; i < (int)all_trades.size(); ++i) {
        const auto& t = all_trades[i];
        cout << setw(4) << (i+1) << setw(16) << t.asset_name
             << setw(12) << t.entry_date << setw(12) << t.exit_date
             << setw(10) << fixed << setprecision(4) << t.entry_price
             << setw(10) << t.exit_price
             << setw(9) << setprecision(2) << t.pnl_pct << "%"
             << setw(7) << t.hold_days
             << setw(12) << t.reason << "\n";
    }
    
    // State time breakdown
    cout << "\n--- 各状态持仓时间分析 ---\n";
    int days_citic = 0, days_nasdaq = 0, days_macd = 0, days_cash = 0;
    {
        MacroState s = STATE_INIT;
        for (int d = start_idx; d <= end_idx; ++d) {
            string cur_date = global_dates[d];
            if (!idx_close_map.count(cur_date)) continue;
            
            double idx_yesterday = (d > start_idx && idx_close_map.count(global_dates[d-1]))
                                   ? idx_close_map[global_dates[d-1]] : 0;
            
            if (d > start_idx && idx_yesterday > 0) {
                if (idx_yesterday < THRESHOLD_LOW) s = STATE_CITIC;
                else if (idx_yesterday > THRESHOLD_HIGH) s = STATE_NASDAQ;
                else if (s == STATE_INIT || s == STATE_CITIC || s == STATE_NASDAQ) s = STATE_MACD;
            } else if (s == STATE_INIT) {
                double idx_today = idx_close_map[cur_date];
                if (idx_today < THRESHOLD_LOW) s = STATE_CITIC;
                else if (idx_today > THRESHOLD_HIGH) s = STATE_NASDAQ;
                else s = STATE_MACD;
            }
            
            if (s == STATE_CITIC) days_citic++;
            else if (s == STATE_NASDAQ) days_nasdaq++;
            else if (s == STATE_MACD) days_macd++;
            else days_cash++;
        }
    }
    int total_days = days_citic + days_nasdaq + days_macd + days_cash;
    cout << "  中信证券(低位): " << days_citic << " 天 (" << fixed << setprecision(1) << (100.0 * days_citic / total_days) << "%)\n";
    cout << "  纳指ETF(高位):  " << days_nasdaq << " 天 (" << (100.0 * days_nasdaq / total_days) << "%)\n";
    cout << "  MACD轮动(正常): " << days_macd << " 天 (" << (100.0 * days_macd / total_days) << "%)\n";
    
    // ============================================================
    // 8. Comparison Table
    // ============================================================
    
    cout << "\n==========================================================\n";
    cout << "  三策略对比\n";
    cout << "==========================================================\n\n";
    
    cout << setw(40) << "策略" << setw(12) << "年化" << setw(12) << "最大回撤"
         << setw(10) << "Sharpe" << setw(10) << "Calmar" << setw(8) << "交易" << "\n";
    cout << string(92, '-') << "\n";
    
    cout << setw(40) << "组合策略(宏观+MACD)" 
         << setw(11) << fixed << setprecision(2) << ann_ret * 100 << "%"
         << setw(11) << mdd * 100 << "%"
         << setw(10) << sharpe
         << setw(10) << calmar
         << setw(8) << all_trades.size() << "\n";
    
    cout << setw(40) << "中信+纳指ETF轮动 (3200/2900)"
         << setw(11) << "32.12%"
         << setw(11) << "24.01%"
         << setw(10) << "—"
         << setw(10) << "1.34"
         << setw(8) << "13" << "\n";
    
    cout << setw(40) << "全ETF MACD轮动 M(6,15,3)"
         << setw(11) << "49.95%"
         << setw(11) << "13.30%"
         << setw(10) << "2.52"
         << setw(10) << "3.76"
         << setw(8) << "—" << "\n";
    
    // Save trades to CSV
    ofstream csv("combined_strategy_trades.csv");
    csv << "seq,asset_code,asset_name,entry_date,exit_date,entry_price,exit_price,pnl_pct,hold_days,reason\n";
    for (int i = 0; i < (int)all_trades.size(); ++i) {
        const auto& t = all_trades[i];
        csv << i+1 << "," << t.asset_code << "," << t.asset_name
            << "," << t.entry_date << "," << t.exit_date
            << "," << fixed << setprecision(4) << t.entry_price << "," << t.exit_price
            << "," << setprecision(2) << t.pnl_pct << "," << t.hold_days
            << "," << t.reason << "\n";
    }
    csv.close();
    cout << "\n交易明细已保存到 combined_strategy_trades.csv\n";
    
    return 0;
}
