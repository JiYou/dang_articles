#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <iomanip>

using namespace std;

// Data structures
struct DailyBar {
    string date;
    double open, close, high, low, volume;
};

struct AggBar {
    string date;
    double open, close, high, low, volume;
    int first_daily_idx;
    int last_daily_idx;
};

struct StockData {
    string code, name;
    vector<DailyBar> daily;
    int split_idx;
    vector<AggBar> train_monthly, test_monthly;
    vector<AggBar> train_weekly, test_weekly;
};

struct Config {
    int m_fast, m_slow, m_signal, trend_mode;
    int w_fast, w_slow, w_signal, buy_mode, sell_mode;
    
    bool operator==(const Config& o) const {
        return m_fast == o.m_fast && m_slow == o.m_slow && m_signal == o.m_signal && trend_mode == o.trend_mode &&
               w_fast == o.w_fast && w_slow == o.w_slow && w_signal == o.w_signal && buy_mode == o.buy_mode && sell_mode == o.sell_mode;
    }
};

struct Result {
    Config cfg;
    double avg_annualized;
    int total_trades;
    int profitable_trades;
    int stocks_traded;
};

// Date helpers
int date_to_days(const string& d) {
    if (d.size() < 10) return 0;
    int y = stoi(d.substr(0, 4));
    int m = stoi(d.substr(5, 2));
    int day = stoi(d.substr(8, 2));
    if (m <= 2) { y--; m += 12; }
    return 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + day;
}

int date_to_week_id(const string& d) { return date_to_days(d) / 7; }

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    return (pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

double get_years(const string& d1, const string& d2) {
    if (d1.size() < 10 || d2.size() < 10) return 0;
    int y1 = stoi(d1.substr(0,4)), m1 = stoi(d1.substr(5,2)), day1 = stoi(d1.substr(8,2));
    int y2 = stoi(d2.substr(0,4)), m2 = stoi(d2.substr(5,2)), day2 = stoi(d2.substr(8,2));
    int days = (y2 - y1) * 365 + (m2 - m1) * 30 + (day2 - day1);
    return days / 365.25;
}

// MACD computation
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

// Aggregation
vector<AggBar> aggregate_monthly(const vector<DailyBar>& daily, int start, int end) {
    vector<AggBar> res;
    if (start >= end) return res;
    
    string current_month = daily[start].date.substr(0, 7);
    AggBar current_bar = {daily[start].date, daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    
    for (int i = start + 1; i < end; ++i) {
        string month = daily[i].date.substr(0, 7);
        if (month == current_month) {
            current_bar.close = daily[i].close;
            current_bar.high = max(current_bar.high, daily[i].high);
            current_bar.low = min(current_bar.low, daily[i].low);
            current_bar.volume += daily[i].volume;
            current_bar.last_daily_idx = i;
        } else {
            res.push_back(current_bar);
            current_month = month;
            current_bar = {daily[i].date, daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i};
        }
    }
    res.push_back(current_bar);
    return res;
}

vector<AggBar> aggregate_weekly(const vector<DailyBar>& daily, int start, int end) {
    vector<AggBar> res;
    if (start >= end) return res;
    
    int current_week = date_to_week_id(daily[start].date);
    AggBar current_bar = {daily[start].date, daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    
    for (int i = start + 1; i < end; ++i) {
        int week = date_to_week_id(daily[i].date);
        if (week == current_week) {
            current_bar.close = daily[i].close;
            current_bar.high = max(current_bar.high, daily[i].high);
            current_bar.low = min(current_bar.low, daily[i].low);
            current_bar.volume += daily[i].volume;
            current_bar.last_daily_idx = i;
        } else {
            res.push_back(current_bar);
            current_week = week;
            current_bar = {daily[i].date, daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i};
        }
    }
    res.push_back(current_bar);
    return res;
}

// Load Data
vector<StockData> load_data() {
    vector<StockData> stocks;
    string all_stock_path = "all_stock.csv";
    ifstream fs(all_stock_path);
    if (!fs.is_open()) {
        all_stock_path = "../all_stock.csv";
        fs.open(all_stock_path);
    }
    if (!fs.is_open()) {
        cerr << "Could not open all_stock.csv" << endl;
        return stocks;
    }
    
    string line;
    while (getline(fs, line)) {
        if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
            line = line.substr(3);
        if (line.empty()) continue;
        stringstream ss(line);
        string code, name;
        getline(ss, code, ',');
        getline(ss, name, ',');
        
        string csv_path = code + ".csv";
        ifstream sfs(csv_path);
        if (!sfs.is_open()) continue;
        
        StockData stock;
        stock.code = code;
        stock.name = name;
        
        string sline;
        bool header = true;
        while (getline(sfs, sline)) {
            if (header) { header = false; continue; }
            stringstream sss(sline);
            string token;
            vector<string> cols;
            while (getline(sss, token, ',')) cols.push_back(token);
            if (cols.size() < 7) continue;
            
            DailyBar bar;
            bar.date = cols[0];
            try {
                bar.open = stod(cols[2]);
                bar.close = stod(cols[3]);
                bar.high = stod(cols[4]);
                bar.low = stod(cols[5]);
                bar.volume = stod(cols[6]);
                stock.daily.push_back(bar);
            } catch (...) {}
        }
        
        if (stock.daily.size() >= 100) {
            stock.split_idx = stock.daily.size() * 0.4;
            stock.train_weekly = aggregate_weekly(stock.daily, 0, stock.split_idx);
            stock.train_monthly = aggregate_monthly(stock.daily, 0, stock.split_idx);
            stock.test_weekly = aggregate_weekly(stock.daily, stock.split_idx, stock.daily.size());
            stock.test_monthly = aggregate_monthly(stock.daily, stock.split_idx, stock.daily.size());
            stocks.push_back(stock);
        }
    }
    return stocks;
}

struct SimStats {
    double ratio = 1.0;
    int trades = 0;
    int wins = 0;
    string first_date = "";
    string last_date = "";
};

SimStats simulate_stock(const StockData& stock, const Config& cfg, bool is_train) {
    const auto& daily = stock.daily;
    const auto& monthly_bars = is_train ? stock.train_monthly : stock.test_monthly;
    const auto& weekly_bars = is_train ? stock.train_weekly : stock.test_weekly;
    
    SimStats stats;
    if (monthly_bars.size() < 5 || weekly_bars.size() < 5) return stats;
    
    vector<double> m_dif, m_dea, m_hist;
    compute_macd(monthly_bars, cfg.m_fast, cfg.m_slow, cfg.m_signal, m_dif, m_dea, m_hist);
    
    vector<double> w_dif, w_dea, w_hist;
    compute_macd(weekly_bars, cfg.w_fast, cfg.w_slow, cfg.w_signal, w_dif, w_dea, w_hist);
    
    vector<int> weekly_to_monthly(weekly_bars.size(), -1);
    int m_idx = 0;
    for (size_t w = 0; w < weekly_bars.size(); ++w) {
        while (m_idx + 1 < monthly_bars.size() && 
               monthly_bars[m_idx].last_daily_idx < weekly_bars[w].first_daily_idx)
            m_idx++;
        if (m_idx < monthly_bars.size() && monthly_bars[m_idx].first_daily_idx <= weekly_bars[w].last_daily_idx)
            weekly_to_monthly[w] = m_idx;
    }
    
    bool holding = false;
    double buy_price = 0;
    
    for (size_t i = 1; i < weekly_bars.size() - 1; ++i) {
        int m = weekly_to_monthly[i];
        if (m < 1) continue;
        
        bool monthly_bullish = false;
        if (cfg.trend_mode == 0) monthly_bullish = (m_dif[m] > 0);
        else if (cfg.trend_mode == 1) monthly_bullish = (m_dif[m] > m_dea[m]);
        else if (cfg.trend_mode == 2) monthly_bullish = (m_hist[m] > 0);
        else if (cfg.trend_mode == 3) monthly_bullish = (m_dif[m] > m_dif[m-1]);
        
        bool weekly_buy = false;
        if (cfg.buy_mode == 0) weekly_buy = (w_dif[i-1] <= w_dea[i-1] && w_dif[i] > w_dea[i]);
        else if (cfg.buy_mode == 1) weekly_buy = (w_hist[i-1] <= 0 && w_hist[i] > 0);
        
        bool weekly_sell = false;
        if (cfg.sell_mode == 0) weekly_sell = (w_dif[i-1] >= w_dea[i-1] && w_dif[i] < w_dea[i]);
        else if (cfg.sell_mode == 1) weekly_sell = (w_dif[i] < 0);
        else if (cfg.sell_mode == 2) weekly_sell = (w_hist[i] < 0);
        
        int exec_idx = weekly_bars[i+1].first_daily_idx;
        if (exec_idx >= daily.size()) continue;
        string exec_date = daily[exec_idx].date;
        double price = daily[exec_idx].open;
        
        if (!holding && weekly_buy && monthly_bullish) {
            holding = true;
            buy_price = price;
            if (stats.first_date.empty()) stats.first_date = exec_date;
        } else if (holding && weekly_sell) {
            holding = false;
            double tr = price / buy_price;
            stats.ratio *= tr;
            stats.trades++;
            if (tr > 1.0) stats.wins++;
            stats.last_date = exec_date;
        }
    }
    
    if (holding) {
        int last_idx = is_train ? stock.split_idx - 1 : daily.size() - 1;
        double tr = daily[last_idx].close / buy_price;
        stats.ratio *= tr;
        stats.trades++;
        if (tr > 1.0) stats.wins++;
        stats.last_date = daily[last_idx].date;
    }
    
    return stats;
}

Result evaluate_config(const vector<StockData>& stocks, const Config& cfg, bool is_train) {
    Result res;
    res.cfg = cfg;
    res.total_trades = 0;
    res.profitable_trades = 0;
    res.stocks_traded = 0;
    
    vector<double> all_annualized;
    
    for (const auto& stock : stocks) {
        SimStats stats = simulate_stock(stock, cfg, is_train);
        if (stats.trades > 0) {
            double years = get_years(stats.first_date, stats.last_date);
            double ann = compute_annualized(stats.ratio, years);
            all_annualized.push_back(ann);
            res.total_trades += stats.trades;
            res.profitable_trades += stats.wins;
            res.stocks_traded++;
        }
    }
    
    // Top 20 scoring (same as V5)
    if (res.stocks_traded >= 20) {
        sort(all_annualized.rbegin(), all_annualized.rend());
        double sum = 0;
        for (int i = 0; i < 20; ++i) sum += all_annualized[i];
        res.avg_annualized = sum / 20.0;
    } else {
        res.avg_annualized = -100.0;
    }
    return res;
}

int main() {
    cout << "Loading data..." << endl;
    vector<StockData> stocks = load_data();
    cout << "Loaded " << stocks.size() << " stocks." << endl;
    if (stocks.empty()) return 1;
    
    // Phase A: Monthly Trend Filter Parameters
    cout << "\n=== Phase A: Monthly Trend Optimization ===" << endl;
    vector<int> m_fasts = {8, 10, 12, 13, 14};
    vector<int> m_slows = {17, 19, 22, 24, 26};
    vector<int> m_signals = {3, 5};
    vector<int> t_modes = {0, 1, 2, 3};
    
    vector<Config> phaseA_configs;
    for (int mf : m_fasts) {
        for (int ms : m_slows) {
            if (mf >= ms) continue;
            for (int m_sig : m_signals) {
                for (int tm : t_modes) {
                    Config c;
                    c.m_fast = mf; c.m_slow = ms; c.m_signal = m_sig; c.trend_mode = tm;
                    c.w_fast = 16; c.w_slow = 39; c.w_signal = 6; c.buy_mode = 1; c.sell_mode = 1;
                    phaseA_configs.push_back(c);
                }
            }
        }
    }
    
    mutex mtx;
    vector<Result> phaseA_results;
    atomic<int> idx(0);
    
    auto workerA = [&]() {
        while (true) {
            int i = idx++;
            if (i >= phaseA_configs.size()) break;
            Result r = evaluate_config(stocks, phaseA_configs[i], true);
            lock_guard<mutex> lock(mtx);
            phaseA_results.push_back(r);
        }
    };
    
    int num_threads = thread::hardware_concurrency();
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) threads.emplace_back(workerA);
    for (auto& t : threads) t.join();
    
    sort(phaseA_results.begin(), phaseA_results.end(), [](const Result& a, const Result& b) {
        return a.avg_annualized > b.avg_annualized;
    });
    
    cout << "Top 5 Monthly Configs:" << endl;
    for (int i = 0; i < min(5, (int)phaseA_results.size()); ++i) {
        auto& r = phaseA_results[i];
        cout << i+1 << ". M=" << r.cfg.m_fast << "," << r.cfg.m_slow << "," << r.cfg.m_signal 
             << " tm=" << r.cfg.trend_mode << " | Ann: " << fixed << setprecision(2) << r.avg_annualized
             << "% | Trades: " << r.total_trades << endl;
    }
    
    // Phase B: Weekly Entry/Exit Optimization
    cout << "\n=== Phase B: Weekly Entry/Exit Optimization ===" << endl;
    vector<Config> phaseB_configs;
    for (int i = 0; i < min(5, (int)phaseA_results.size()); ++i) {
        auto best_m = phaseA_results[i].cfg;
        for (int wf = 8; wf <= 18; wf += 2) {
            for (int ws = 15; ws <= 45; ws += 3) {
                if (wf >= ws) continue;
                for (int wsig = 3; wsig <= 9; wsig += 2) {
                    for (int bm : {0, 1}) {
                        for (int sm : {0, 1, 2}) {
                            Config c = best_m;
                            c.w_fast = wf; c.w_slow = ws; c.w_signal = wsig; c.buy_mode = bm; c.sell_mode = sm;
                            phaseB_configs.push_back(c);
                        }
                    }
                }
            }
        }
    }
    
    vector<Result> phaseB_results;
    idx = 0;
    
    auto workerB = [&]() {
        while (true) {
            int i = idx++;
            if (i >= phaseB_configs.size()) break;
            Result r = evaluate_config(stocks, phaseB_configs[i], true);
            lock_guard<mutex> lock(mtx);
            phaseB_results.push_back(r);
        }
    };
    
    threads.clear();
    for (int i = 0; i < num_threads; ++i) threads.emplace_back(workerB);
    for (auto& t : threads) t.join();
    
    sort(phaseB_results.begin(), phaseB_results.end(), [](const Result& a, const Result& b) {
        return a.avg_annualized > b.avg_annualized;
    });
    
    cout << "Top 20 Dual-Timeframe Configs (Train):" << endl;
    for (int i = 0; i < min(20, (int)phaseB_results.size()); ++i) {
        auto& r = phaseB_results[i];
        cout << i+1 << ". M=" << r.cfg.m_fast << "," << r.cfg.m_slow << "," << r.cfg.m_signal 
             << " tm=" << r.cfg.trend_mode << " | W=" << r.cfg.w_fast << "," << r.cfg.w_slow << "," << r.cfg.w_signal 
             << " bm=" << r.cfg.buy_mode << " sm=" << r.cfg.sell_mode 
             << " | Ann: " << fixed << setprecision(2) << r.avg_annualized << "%" << endl;
    }
    
    // Phase C: Test Evaluation
    cout << "\n=== Phase C: Test Evaluation ===" << endl;
    Config best_cfg = phaseB_results[0].cfg;
    Result train_res = phaseB_results[0];
    Result test_res = evaluate_config(stocks, best_cfg, false);
    
    cout << "\nBest Config:" << endl;
    cout << "Monthly: " << best_cfg.m_fast << "," << best_cfg.m_slow << "," << best_cfg.m_signal << " tm=" << best_cfg.trend_mode << endl;
    cout << "Weekly:  " << best_cfg.w_fast << "," << best_cfg.w_slow << "," << best_cfg.w_signal << " bm=" << best_cfg.buy_mode << " sm=" << best_cfg.sell_mode << endl;
    
    cout << "\nVersion Comparison:" << endl;
    cout << "  Version        Train      Test  Config" << endl;
    cout << "  V4            51.10%    46.41%  10,24,3 Hist>0/DIF<0 monthly" << endl;
    cout << "  V5            51.07%    46.66%  14,17,3 GoldenX/DIF<0 monthly" << endl;
    cout << "  V6            51.30%    46.27%  13,19,3 GoldenX/DIF<0 rsi=7" << endl;
    cout << "  V7            " << fixed << setprecision(2) << train_res.avg_annualized << "%    " << test_res.avg_annualized << "%  Monthly M+W dual-timeframe" << endl;
    
    // Export Results
    ofstream ofs("macd_v7_results.csv");
    ofs << "code,name,trades,wins,win_rate,annualized\n";
    
    struct StockResult { string code, name; double ann; };
    vector<StockResult> s_results;
    
    for (const auto& stock : stocks) {
        SimStats stats = simulate_stock(stock, best_cfg, false);
        if (stats.trades > 0) {
            double years = get_years(stats.first_date, stats.last_date);
            double ann = compute_annualized(stats.ratio, years);
            double wr = (double)stats.wins / stats.trades * 100.0;
            ofs << stock.code << "," << stock.name << "," << stats.trades << "," 
                << stats.wins << "," << fixed << setprecision(2) << wr << "," << ann << "\n";
            s_results.push_back({stock.code, stock.name, ann});
        }
    }
    ofs.close();
    
    sort(s_results.begin(), s_results.end(), [](const StockResult& a, const StockResult& b){ return a.ann > b.ann; });
    
    ofstream tfs("macd_v7_top20.csv");
    tfs << "code,name,annualized\n";
    for(int i=0; i<min(20, (int)s_results.size()); ++i) {
        tfs << s_results[i].code << "," << s_results[i].name << "," << s_results[i].ann << "\n";
    }
    tfs.close();
    
    return 0;
}
