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

struct DailyBar {
    string date;
    double open, high, low, close, volume;
};

struct AggBar {
    double open, close, high, low, volume;
    int first_daily_idx;  // index into daily[] for execution
    int last_daily_idx;
};

struct ETFData {
    string code, name;
    vector<DailyBar> daily;
    // Window specific
    vector<AggBar> train_monthly, test_monthly;
    vector<AggBar> train_weekly, test_weekly;
};

struct StrategyParams {
    int m_fast, m_slow, m_signal;
    int trend_mode;  
    
    int w_fast, w_slow, w_signal;
    int buy_mode;   
    int sell_mode;  
    
    double trailing_stop;  
    double take_profit;    
};

struct Trade {
    string etf_code, etf_name;
    string entry_date, exit_date;
    double entry_price, exit_price;
    double pnl_pct;
    int hold_days;
};

struct BacktestResult {
    double total_pnl_pct = 0.0;
    double annualized = 0.0;
    int trades = 0;
    int win_trades = 0;
    vector<Trade> trade_list;
};

struct ScoreResult {
    StrategyParams params;
    double train_avg_annualized;
    int train_total_trades;
    double test_avg_annualized;
    int test_total_trades;
    unordered_map<string, BacktestResult> etf_results;
};

struct WindowResult {
    string train_start, train_end, test_end;
    ScoreResult best_score;
    double test_top20_avg;
    double window_median_return;
};

// --- Globals ---
unordered_map<string, string> g_etf_names = {
    {"sh000001", "上证指数"}, {"sh000016", "上证50"}, {"sh000300", "沪深300"},
    {"sh000852", "中证1000"}, {"sh000905", "中证500"},
    {"sz399001", "深证成指"}, {"sz399006", "创业板指"},
    {"sh000688", "科创50指数"}, {"sz399303", "国证2000"}, {"sz399673", "创业板50指数"},
    {"sz399986", "中证新能"}, {"sz399989", "中证医疗"},
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

// --- Date Helpers ---
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

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    if (years < 2.0) return (ratio - 1.0) * 100.0;
    return (pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

double compute_years(const string& d1, const string& d2) {
    if (d1.size() < 10 || d2.size() < 10) return 0;
    int y1 = stoi(d1.substr(0, 4)), m1 = stoi(d1.substr(5, 2)), day1 = stoi(d1.substr(8, 2));
    int y2 = stoi(d2.substr(0, 4)), m2 = stoi(d2.substr(5, 2)), day2 = stoi(d2.substr(8, 2));
    int days = (y2 - y1) * 365 + (m2 - m1) * 30 + (day2 - day1);
    return days / 365.25;
}

// --- Data Aggregation ---
vector<AggBar> aggregate_weekly(const vector<DailyBar>& daily, int start, int end) {
    vector<AggBar> bars;
    if (start >= end || start >= (int)daily.size()) return bars;
    int current_id = date_to_week_id(daily[start].date);
    AggBar b = {daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    
    for (int i = start + 1; i < end; ++i) {
        int id = date_to_week_id(daily[i].date);
        if (id != current_id) {
            bars.push_back(b);
            current_id = id;
            b = {daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i};
        } else {
            b.high = max(b.high, daily[i].high);
            b.low = min(b.low, daily[i].low);
            b.close = daily[i].close;
            b.volume += daily[i].volume;
            b.last_daily_idx = i;
        }
    }
    bars.push_back(b);
    return bars;
}

vector<AggBar> aggregate_monthly(const vector<DailyBar>& daily, int start, int end) {
    vector<AggBar> bars;
    if (start >= end || start >= (int)daily.size()) return bars;
    int current_id = date_to_month_id(daily[start].date);
    AggBar b = {daily[start].open, daily[start].close, daily[start].high, daily[start].low, daily[start].volume, start, start};
    
    for (int i = start + 1; i < end; ++i) {
        int id = date_to_month_id(daily[i].date);
        if (id != current_id) {
            bars.push_back(b);
            current_id = id;
            b = {daily[i].open, daily[i].close, daily[i].high, daily[i].low, daily[i].volume, i, i};
        } else {
            b.high = max(b.high, daily[i].high);
            b.low = min(b.low, daily[i].low);
            b.close = daily[i].close;
            b.volume += daily[i].volume;
            b.last_daily_idx = i;
        }
    }
    bars.push_back(b);
    return bars;
}

// --- MACD ---
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

// --- Backtesting Core ---
BacktestResult run_backtest(const ETFData& etf, const StrategyParams& params, bool is_test) {
    BacktestResult res;
    const auto& weekly = is_test ? etf.test_weekly : etf.train_weekly;
    const auto& monthly = is_test ? etf.test_monthly : etf.train_monthly;
    
    if (weekly.size() < 20 || monthly.size() < 5) return res;
    
    vector<double> m_dif, m_dea, m_hist;
    compute_macd(monthly, params.m_fast, params.m_slow, params.m_signal, m_dif, m_dea, m_hist);
    
    vector<double> w_dif, w_dea, w_hist;
    compute_macd(weekly, params.w_fast, params.w_slow, params.w_signal, w_dif, w_dea, w_hist);
    
    vector<int> weekly_to_monthly(weekly.size(), -1);
    int m_idx = 0;
    for (size_t w = 0; w < weekly.size(); ++w) {
        while (m_idx + 1 < (int)monthly.size() && 
               monthly[m_idx].last_daily_idx < weekly[w].first_daily_idx)
            m_idx++;
        if (m_idx < (int)monthly.size() &&
            monthly[m_idx].first_daily_idx <= weekly[w].last_daily_idx)
            weekly_to_monthly[w] = m_idx;
    }
    
    bool in_pos = false;
    double entry_price = 0, highest_price = 0;
    string entry_date;
    double capital = 10000;
    
    for (size_t i = max({(size_t)1, (size_t)params.w_slow, (size_t)params.m_slow}); i < weekly.size() - 1; ++i) {
        int mi = weekly_to_monthly[i];
        if (mi <= 0) continue;
        
        bool m_bull = false;
        if (params.trend_mode == 0) m_bull = m_dif[mi] > 0;
        else if (params.trend_mode == 1) m_bull = m_dif[mi] > m_dea[mi];
        else if (params.trend_mode == 2) m_bull = m_hist[mi] > 0;
        else if (params.trend_mode == 3) m_bull = m_dif[mi] > m_dif[mi-1];
        
        if (!in_pos) {
            bool w_buy = false;
            if (params.buy_mode == 0) w_buy = (w_dif[i] > w_dea[i] && w_dif[i-1] <= w_dea[i-1]);
            else if (params.buy_mode == 1) w_buy = (w_hist[i] > 0 && w_hist[i-1] <= 0);
            
            // NO SECTOR ROTATION in V9
            if (w_buy && m_bull) {
                in_pos = true;
                int nxt_idx = weekly[i+1].first_daily_idx;
                entry_price = etf.daily[nxt_idx].open;
                highest_price = entry_price;
                entry_date = etf.daily[nxt_idx].date;
            }
        } else {
            int curr_idx = weekly[i].last_daily_idx;
            double cur_close = etf.daily[curr_idx].close;
            double cur_high = etf.daily[curr_idx].high;
            highest_price = max(highest_price, cur_high);
            
            bool sell = false;
            if (params.sell_mode == 0) sell = (w_dif[i] < w_dea[i] && w_dif[i-1] >= w_dea[i-1]);
            else if (params.sell_mode == 1) sell = (w_dif[i] < 0);
            else if (params.sell_mode == 2) sell = (w_hist[i] < 0 && w_hist[i-1] >= 0);
            
            if (!sell && params.trailing_stop > 0) {
                if (cur_close <= highest_price * (1.0 - params.trailing_stop/100.0)) sell = true;
            }
            if (!sell && params.take_profit > 0) {
                if (cur_close >= entry_price * (1.0 + params.take_profit/100.0)) sell = true;
            }
            
            if (sell || i == weekly.size() - 2) {
                int exit_idx = (i == weekly.size() - 2) ? weekly[i+1].last_daily_idx : weekly[i+1].first_daily_idx;
                double exit_price = etf.daily[exit_idx].open;
                double pnl = (exit_price - entry_price) / entry_price * 100.0;
                capital *= (1.0 + pnl / 100.0);
                
                Trade t = {etf.code, etf.name, entry_date, etf.daily[exit_idx].date, entry_price, exit_price, pnl, exit_idx - weekly[i+1].first_daily_idx};
                res.trade_list.push_back(t);
                res.trades++;
                if (pnl > 0) res.win_trades++;
                
                in_pos = false;
            }
        }
    }
    
    if (res.trades > 0) {
        string d1 = etf.daily[weekly.front().first_daily_idx].date; string d2 = etf.daily[weekly.back().last_daily_idx].date; double yrs = compute_years(d1, d2);
        res.total_pnl_pct = (capital - 10000) / 100.0;
        res.annualized = compute_annualized(capital / 10000.0, yrs);
    }
    return res;
}

ScoreResult eval_params(const vector<ETFData>& etfs, const StrategyParams& params) {
    ScoreResult sr;
    sr.params = params;
    sr.train_total_trades = 0;
    sr.test_total_trades = 0;
    
    vector<double> train_anns, test_anns;
    
    for (const auto& etf : etfs) {
        if(etf.train_weekly.empty()) continue;
        
        auto train_res = run_backtest(etf, params, false);
        if (train_res.trades > 0) {
            double ann = max(-200.0, min(200.0, train_res.annualized));
            train_anns.push_back(ann);
            sr.train_total_trades += train_res.trades;
        }
        
        auto test_res = run_backtest(etf, params, true);
        if (test_res.trades > 0) {
            double ann = max(-200.0, min(200.0, test_res.annualized));
            test_res.annualized = ann;
            test_anns.push_back(ann);
            sr.test_total_trades += test_res.trades;
            sr.etf_results[etf.code] = test_res;
        }
    }
    
    sort(train_anns.rbegin(), train_anns.rend());
    int n_tr = min(20, (int)train_anns.size());
    if (n_tr >= 5) {
        if (n_tr % 2 == 0) sr.train_avg_annualized = (train_anns[n_tr/2 - 1] + train_anns[n_tr/2]) / 2.0;
        else sr.train_avg_annualized = train_anns[n_tr/2];
    } else {
        sr.train_avg_annualized = -100;
    }
    
    sort(test_anns.rbegin(), test_anns.rend());
    int n_te = min(20, (int)test_anns.size());
    if (n_te >= 5) {
        if (n_te % 2 == 0) sr.test_avg_annualized = (test_anns[n_te/2 - 1] + test_anns[n_te/2]) / 2.0;
        else sr.test_avg_annualized = test_anns[n_te/2];
    } else {
        sr.test_avg_annualized = -100;
    }
    
    return sr;
}

int main() {
    string data_dir = "/ceph/dang_articles/yoj/market_data/";
    vector<ETFData> all_etfs;
    
    // Load Data
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        string path = entry.path().string();
        if (path.find(".csv") == string::npos) continue;
        
        string code = entry.path().stem().string();
        if (code.find("sh000") == 0 || code.find("sz399") == 0) continue; 
        if (g_etf_names.find(code) == g_etf_names.end()) continue;
        
        ifstream file(path);
        string line;
        getline(file, line); 
        
        ETFData etf;
        etf.code = code;
        etf.name = g_etf_names[code];
        
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
        all_etfs.push_back(etf);
    }
    
    cout << "Loaded " << all_etfs.size() << " tradeable ETFs.\n";
    
    // Find global min and max dates
    string min_date = "9999-99-99", max_date = "0000-00-00";
    for(const auto& etf : all_etfs) {
        if(etf.daily.front().date < min_date) min_date = etf.daily.front().date;
        if(etf.daily.back().date > max_date) max_date = etf.daily.back().date;
    }
    
    int start_year = stoi(min_date.substr(0, 4));
    int end_year = stoi(max_date.substr(0, 4));
    
    cout << "Global data range: " << min_date << " to " << max_date << "\n";
    
    // Grid Generation
    vector<StrategyParams> full_grid;
    for (int mf : {8, 10, 12, 14}) {
        for (int ms : {15, 17, 20, 24}) {
            for (int msg : {3, 5}) {
                for (int tm : {0, 1, 2, 3}) {
                    for (int wf : {6, 8, 10, 12}) {
                        for (int ws : {25, 30, 35, 40}) {
                            for (int wsig : {3, 5}) {
                                for (int bm : {0, 1}) {
                                    for (int sm : {0, 1, 2}) {
                                        for (double tr : {0.0, 15.0, 20.0}) {
                                            for (double tp : {0.0, 20.0, 30.0, 50.0}) {
                                                StrategyParams p;
                                                p.m_fast = mf; p.m_slow = ms; p.m_signal = msg; p.trend_mode = tm;
                                                p.w_fast = wf; p.w_slow = ws; p.w_signal = wsig;
                                                p.buy_mode = bm; p.sell_mode = sm;
                                                p.trailing_stop = tr; p.take_profit = tp;
                                                full_grid.push_back(p);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    cout << "Total grid size per window: " << full_grid.size() << " combinations\n";

    vector<WindowResult> walk_forward_results;

    // Walk-forward loop
    for(int y = start_year; y <= end_year - 4; ++y) {
        string t_start = to_string(y) + "-01-01";
        string t_end = to_string(y + 3) + "-01-01";
        string test_end = to_string(y + 4) + "-01-01";
        
        vector<ETFData> window_etfs;
        for(auto& etf : all_etfs) {
            ETFData w_etf;
            w_etf.code = etf.code;
            w_etf.name = etf.name;
            
            int train_s = -1, train_e = -1, test_s = -1, test_e = -1;
            for(size_t i=0; i<etf.daily.size(); ++i) {
                if(train_s == -1 && etf.daily[i].date >= t_start) train_s = i;
                if(train_e == -1 && etf.daily[i].date >= t_end) { train_e = i; test_s = i; }
                if(test_e == -1 && etf.daily[i].date >= test_end) { test_e = i; break; }
            }
            if(train_e == -1) train_e = etf.daily.size();
            if(test_s == -1) test_s = etf.daily.size();
            if(test_e == -1) test_e = etf.daily.size();
            
            // Need at least ~100 bars for train
            if(train_e - train_s < 100) continue;
            
            w_etf.daily = etf.daily; // Keep full daily for indexing
            w_etf.train_weekly = aggregate_weekly(w_etf.daily, train_s, train_e);
            w_etf.train_monthly = aggregate_monthly(w_etf.daily, train_s, train_e);
            w_etf.test_weekly = aggregate_weekly(w_etf.daily, test_s, test_e);
            w_etf.test_monthly = aggregate_monthly(w_etf.daily, test_s, test_e);
            
            window_etfs.push_back(w_etf);
        }
        
        if(window_etfs.empty()) continue;
        
        if(walk_forward_results.empty()) cout << "\nWalk-Forward Windows:\n";
        cout << "  Window " << walk_forward_results.size() + 1 << ": Train [" << t_start.substr(0,7) << " to " << t_end.substr(0,7) 
             << "] Test [" << t_end.substr(0,7) << " to " << test_end.substr(0,7) << "], ETFs: " << window_etfs.size() << "\n";
             
        auto run_grid = [&](const vector<StrategyParams>& grid) {
            vector<ScoreResult> results(grid.size());
            atomic<int> idx(0);
            vector<thread> threads;
            for(int t=0; t<(int)thread::hardware_concurrency(); ++t) {
                threads.emplace_back([&](){
                    while(true) {
                        int i = idx++;
                        if (i >= (int)grid.size()) break;
                        results[i] = eval_params(window_etfs, grid[i]);
                    }
                });
            }
            for(auto& t : threads) t.join();
            sort(results.begin(), results.end(), [](const ScoreResult& a, const ScoreResult& b){
                return a.train_avg_annualized > b.train_avg_annualized;
            });
            return results;
        };

        auto res_grid = run_grid(full_grid);
        if(res_grid.empty()) continue;
        
        auto best = res_grid[0];
        
        cout << "    Best: M(" << best.params.m_fast << "," << best.params.m_slow << "," << best.params.m_signal 
             << " md" << best.params.trend_mode << ") W(" << best.params.w_fast << "," << best.params.w_slow << "," 
             << best.params.w_signal << " b" << best.params.buy_mode << " s" << best.params.sell_mode 
             << ") Tr=" << best.params.trailing_stop << " Tp=" << best.params.take_profit << "\n";
        cout << " -> Test Top20 Median: " << fixed << setprecision(2) << best.test_avg_annualized << "%\n";
        
        int test_traded_etfs = best.etf_results.size();
        int pos_etfs = 0;
        vector<double> best_test_anns;
        for (const auto& kv : best.etf_results) {
            best_test_anns.push_back(kv.second.annualized);
            if (kv.second.annualized > 0) pos_etfs++;
        }
        double med_test_ann = -100.0;
        if (!best_test_anns.empty()) {
            sort(best_test_anns.begin(), best_test_anns.end());
            int n = best_test_anns.size();
            if (n % 2 == 0) med_test_ann = (best_test_anns[n/2 - 1] + best_test_anns[n/2]) / 2.0;
            else med_test_ann = best_test_anns[n/2];
        }
        
        cout << "    [Window Stats] Traded ETFs: " << test_traded_etfs 
             << ", Median Test Return: " << med_test_ann 
             << "%, Positive ETFs: " << pos_etfs << "\n";
        
        WindowResult wr = {t_start, t_end, test_end, best, best.test_avg_annualized, med_test_ann};
        walk_forward_results.push_back(wr);
    }
    
    // Aggregation
    unordered_map<string, vector<double>> etf_all_test_returns;
    for(const auto& wr : walk_forward_results) {
        for(const auto& kv : wr.best_score.etf_results) {
            etf_all_test_returns[kv.first].push_back(kv.second.annualized);
        }
    }
    
    struct FinalETF { string code; string name; double avg_ann; int windows; double consistency; };
    vector<FinalETF> final_etfs;
    vector<double> all_avgs;
    
    for(const auto& kv : etf_all_test_returns) {
        if(kv.second.empty()) continue;
        double sum = 0;
        int pos = 0;
        for(double v : kv.second) {
            sum += v;
            if(v > 0) pos++;
        }
        double avg = sum / kv.second.size();
        double cons = (double)pos / kv.second.size() * 100.0;
        final_etfs.push_back({kv.first, g_etf_names[kv.first], avg, (int)kv.second.size(), cons});
        all_avgs.push_back(avg);
    }
    
    sort(all_avgs.rbegin(), all_avgs.rend());
    int n_top = min(20, (int)all_avgs.size());
    double final_top20_avg = n_top > 0 ? accumulate(all_avgs.begin(), all_avgs.begin()+n_top, 0.0) / n_top : 0.0;
    
    double sum_test = 0, sum_sq = 0, min_w = 9999, max_w = -9999;
    vector<double> window_scores;
    int pos_windows = 0;
    for(const auto& wr : walk_forward_results) {
        window_scores.push_back(wr.test_top20_avg);
        if (wr.window_median_return > 0) pos_windows++;
        sum_test += wr.test_top20_avg;
        sum_sq += wr.test_top20_avg * wr.test_top20_avg;
        if(wr.test_top20_avg < min_w) min_w = wr.test_top20_avg;
        if(wr.test_top20_avg > max_w) max_w = wr.test_top20_avg;
    }
    double n_w = walk_forward_results.size();
    double avg_w = n_w > 0 ? sum_test / n_w : 0;
    double std_w = n_w > 0 ? sqrt(sum_sq / n_w - avg_w * avg_w) : 0;
    
    double final_score_median = 0;
    if (!window_scores.empty()) {
        sort(window_scores.begin(), window_scores.end());
        int n = window_scores.size();
        if (n % 2 == 0) final_score_median = (window_scores[n/2 - 1] + window_scores[n/2]) / 2.0;
        else final_score_median = window_scores[n/2];
    }
    double pos_win_ratio = n_w > 0 ? (double)pos_windows / n_w * 100.0 : 0.0;
    
    cout << "\nWalk-Forward Summary:\n";
    cout << "  Median Test Top20 across windows: " << fixed << setprecision(2) << final_score_median << "%\n";
    cout << "  Positive Windows: " << pos_windows << " (" << fixed << setprecision(2) << pos_win_ratio << "%)\n";
    cout << "  Std Dev (of scores): " << fixed << setprecision(2) << std_w << "%\n";
    cout << "  Min window score: " << fixed << setprecision(2) << min_w << "%\n";
    cout << "  Max window score: " << fixed << setprecision(2) << max_w << "%\n\n";
    
    cout << "Comparison:\n";
    cout << "  V7      Test 31.14%  (fixed 40/60 split, multi-timeframe)\n";
    cout << "  V8      Test 33.84%  (fixed 40/60 split, + sector rotation)\n";
    cout << "  V9-WF   Test " << fixed << setprecision(2) << final_score_median << "%  (walk-forward, median across windows)\n";
    
    ofstream csv("etf_macd_v9_results.csv");
    csv << "Code,Name,WF_AvgAnnualized,NumWindows,Consistency(%)\n";
    for(const auto& f : final_etfs) {
        csv << f.code << "," << f.name << "," << fixed << setprecision(2) << f.avg_ann 
            << "," << f.windows << "," << f.consistency << "%\n";
    }
    
    return 0;
}
