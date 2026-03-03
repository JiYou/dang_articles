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
    string code, name, sector;
    int sector_id;
    vector<DailyBar> daily;
    int split_idx;
    vector<AggBar> train_monthly, test_monthly;
    vector<AggBar> train_weekly, test_weekly;
    vector<double> global_close;
};

struct StrategyParams {
    int m_fast, m_slow, m_signal;
    int trend_mode;  
    
    int w_fast, w_slow, w_signal;
    int buy_mode;   
    int sell_mode;  
    
    double trailing_stop;  
    double take_profit;    

    int lookback_idx;
    int top_k_idx;
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

unordered_map<string, string> g_etf_sectors = {
    {"sh510300", "sector_broad"}, {"sh510050", "sector_broad"}, {"sh510500", "sector_broad"}, {"sz159915", "sector_broad"}, {"sh588000", "sector_broad"}, {"sh515080", "sector_broad"}, {"sh561990", "sector_broad"},
    {"sh512000", "sector_finance"}, {"sh512800", "sector_finance"}, {"sh512070", "sector_finance"},
    {"sz159928", "sector_consumer"}, {"sh512690", "sector_consumer"}, {"sh515170", "sector_consumer"},
    {"sh512010", "sector_healthcare"}, {"sh512170", "sector_healthcare"}, {"sh512290", "sector_healthcare"}, {"sh562800", "sector_healthcare"},
    {"sh512480", "sector_tech"}, {"sh512760", "sector_tech"}, {"sz159995", "sector_tech"}, {"sh515000", "sector_tech"}, {"sh515710", "sector_tech"}, {"sh515880", "sector_tech"}, {"sh516780", "sector_tech"},
    {"sh516160", "sector_energy"}, {"sh515790", "sector_energy"}, {"sh515030", "sector_energy"}, {"sh515860", "sector_energy"}, {"sh562510", "sector_energy"},
    {"sh512400", "sector_materials"}, {"sh515650", "sector_materials"}, {"sh516150", "sector_materials"}, {"sh515210", "sector_materials"}, {"sh515220", "sector_materials"},
    {"sh512660", "sector_military"},
    {"sh512200", "sector_realestate"}, {"sh516950", "sector_realestate"},
    {"sh512980", "sector_media"},
    {"sh513100", "sector_overseas"}, {"sz159941", "sector_overseas"}, {"sh513060", "sector_overseas"}, {"sz159920", "sector_overseas"}, {"sh513050", "sector_overseas"}, {"sh513030", "sector_overseas"}, {"sh513080", "sector_overseas"}, {"sh513520", "sector_overseas"}, {"sh513000", "sector_overseas"}, {"sh513880", "sector_overseas"}, {"sz159866", "sector_overseas"}, {"sh513090", "sector_overseas"}, {"sh513010", "sector_overseas"}, {"sh513180", "sector_overseas"}, {"sh513330", "sector_overseas"}, {"sh513550", "sector_overseas"}, {"sz159607", "sector_overseas"},
    {"sh518880", "sector_commodity"}, {"sz159934", "sector_commodity"}, {"sh518800", "sector_commodity"}, {"sz159981", "sector_commodity"}, {"sz159985", "sector_commodity"}, {"sh518660", "sector_commodity"}, {"sz159869", "sector_commodity"},
    {"sh510880", "sector_dividend"}, {"sh515180", "sector_dividend"}, {"sh512890", "sector_dividend"}, {"sh510090", "sector_dividend"}
};

// Global Momentum state
vector<int> g_lookbacks = {21, 42, 63, 84, 126};
vector<int> g_top_ks = {3, 5, 7, 10, 99};
unordered_map<string, int> g_date_to_idx;
vector<vector<vector<vector<bool>>>> g_is_top_k; // [lookback_idx][top_k_idx][sector_id][date_idx]

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
            
            int curr_daily_idx = weekly[i].last_daily_idx;
            string date_str = etf.daily[curr_daily_idx].date;
            int d_idx = g_date_to_idx[date_str];
            bool sector_ok = g_is_top_k[params.lookback_idx][params.top_k_idx][etf.sector_id][d_idx];
            
            if (w_buy && m_bull && sector_ok) {
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
        double yrs = compute_years(res.trade_list.front().entry_date, res.trade_list.back().exit_date);
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
        auto train_res = run_backtest(etf, params, false);
        if (train_res.trades > 0) {
            train_anns.push_back(train_res.annualized);
            sr.train_total_trades += train_res.trades;
        }
        
        auto test_res = run_backtest(etf, params, true);
        if (test_res.trades > 0) {
            test_anns.push_back(test_res.annualized);
            sr.test_total_trades += test_res.trades;
            sr.etf_results[etf.code] = test_res;
        }
    }
    
    sort(train_anns.rbegin(), train_anns.rend());
    int n_tr = min(20, (int)train_anns.size());
    sr.train_avg_annualized = n_tr >= 5 ? accumulate(train_anns.begin(), train_anns.begin()+n_tr, 0.0)/n_tr : -100;
    
    sort(test_anns.rbegin(), test_anns.rend());
    int n_te = min(20, (int)test_anns.size());
    sr.test_avg_annualized = n_te >= 5 ? accumulate(test_anns.begin(), test_anns.begin()+n_te, 0.0)/n_te : -100;
    
    return sr;
}

int main() {
    string data_dir = "/ceph/dang_articles/yoj/market_data/";
    vector<ETFData> etfs;
    
    unordered_map<string, int> sector_name_to_id;
    vector<string> id_to_sector_name;
    int num_sectors = 0;

    // Load Data
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        string path = entry.path().string();
        if (path.find(".csv") == string::npos) continue;
        
        string code = entry.path().stem().string();
        if (code.find("sh000") == 0 || code.find("sz399") == 0) continue; 
        if (g_etf_sectors.find(code) == g_etf_sectors.end()) continue;
        
        ifstream file(path);
        string line;
        getline(file, line); 
        
        ETFData etf;
        etf.code = code;
        etf.name = g_etf_names.count(code) ? g_etf_names[code] : code;
        etf.sector = g_etf_sectors[code];
        
        if (sector_name_to_id.find(etf.sector) == sector_name_to_id.end()) {
            sector_name_to_id[etf.sector] = num_sectors++;
            id_to_sector_name.push_back(etf.sector);
        }
        etf.sector_id = sector_name_to_id[etf.sector];
        
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
        etf.train_weekly = aggregate_weekly(etf.daily, 0, etf.split_idx);
        etf.train_monthly = aggregate_monthly(etf.daily, 0, etf.split_idx);
        etf.test_weekly = aggregate_weekly(etf.daily, etf.split_idx, etf.daily.size());
        etf.test_monthly = aggregate_monthly(etf.daily, etf.split_idx, etf.daily.size());
        
        etfs.push_back(etf);
    }
    
    cout << "Loaded " << etfs.size() << " tradeable ETFs across " << num_sectors << " sectors.\n";
    
    // Build global dates and precompute sector momentum
    set<string> all_dates_set;
    for (const auto& etf : etfs) {
        for (const auto& b : etf.daily) {
            all_dates_set.insert(b.date);
        }
    }
    vector<string> global_dates(all_dates_set.begin(), all_dates_set.end());
    for (int i = 0; i < (int)global_dates.size(); ++i) {
        g_date_to_idx[global_dates[i]] = i;
    }

    for (auto& etf : etfs) {
        etf.global_close.assign(global_dates.size(), 0.0);
        double last_close = 0.0;
        for (const auto& b : etf.daily) {
            etf.global_close[g_date_to_idx[b.date]] = b.close;
        }
        for (int i = 0; i < (int)global_dates.size(); ++i) {
            if (etf.global_close[i] > 0) {
                last_close = etf.global_close[i];
            } else {
                etf.global_close[i] = last_close; 
            }
        }
    }

    g_is_top_k.resize(g_lookbacks.size(), vector<vector<vector<bool>>>(
        g_top_ks.size(), vector<vector<bool>>(
            num_sectors, vector<bool>(global_dates.size(), false)
        )
    ));

    for (size_t l_idx = 0; l_idx < g_lookbacks.size(); ++l_idx) {
        int D = g_lookbacks[l_idx];
        for (int d_idx = 0; d_idx < (int)global_dates.size(); ++d_idx) {
            vector<pair<double, int>> sector_moms; 
            for (int s_id = 0; s_id < num_sectors; ++s_id) {
                double sum_ret = 0;
                int count = 0;
                for (const auto& etf : etfs) {
                    if (etf.sector_id == s_id) {
                        if (d_idx >= D) {
                            double c_curr = etf.global_close[d_idx];
                            double c_prev = etf.global_close[d_idx - D];
                            if (c_curr > 0 && c_prev > 0) {
                                sum_ret += (c_curr / c_prev - 1.0);
                                count++;
                            }
                        }
                    }
                }
                double mom = count > 0 ? (sum_ret / count) : -999999.0;
                sector_moms.push_back({mom, s_id});
            }
            
            sort(sector_moms.rbegin(), sector_moms.rend());
            
            for (size_t k_idx = 0; k_idx < g_top_ks.size(); ++k_idx) {
                int K = g_top_ks[k_idx];
                for (int i = 0; i < min(K, num_sectors); ++i) {
                    if (sector_moms[i].first != -999999.0) { 
                        g_is_top_k[l_idx][k_idx][sector_moms[i].second][d_idx] = true;
                    }
                }
                if (K == 99) {
                    for(int i = 0; i < num_sectors; ++i) {
                        g_is_top_k[l_idx][k_idx][i][d_idx] = true; 
                    }
                }
            }
        }
    }
    
    // Thread pool lambda
    auto run_grid = [&](const vector<StrategyParams>& grid) {
        vector<ScoreResult> results(grid.size());
        atomic<int> idx(0);
        vector<thread> threads;
        for(int t=0; t<(int)thread::hardware_concurrency(); ++t) {
            threads.emplace_back([&](){
                while(true) {
                    int i = idx++;
                    if (i >= (int)grid.size()) break;
                    results[i] = eval_params(etfs, grid[i]);
                }
            });
        }
        for(auto& t : threads) t.join();
        sort(results.begin(), results.end(), [](const ScoreResult& a, const ScoreResult& b){
            return a.train_avg_annualized > b.train_avg_annualized;
        });
        return results;
    };
    
    // Phase A
    cout << "\n--- Phase A: Monthly Trend + Fixed Weekly Baseline + Sector Rotation ---\n";
    vector<StrategyParams> grid_A;
    vector<int> m_fasts = {8, 10, 12, 14};
    vector<int> m_slows = {17, 20, 24, 28};
    vector<int> m_sigs = {3, 5};
    vector<int> t_modes = {0, 1, 2, 3};
    
    for (int mf : m_fasts) {
        for (int ms : m_slows) {
            for (int msg : m_sigs) {
                for (int tm : t_modes) {
                    for (int l_idx = 0; l_idx < (int)g_lookbacks.size(); ++l_idx) {
                        for (int k_idx = 0; k_idx < (int)g_top_ks.size(); ++k_idx) {
                            StrategyParams p;
                            p.m_fast = mf; p.m_slow = ms; p.m_signal = msg; p.trend_mode = tm;
                            p.w_fast = 8; p.w_slow = 30; p.w_signal = 3;
                            p.buy_mode = 1; p.sell_mode = 2; 
                            p.trailing_stop = 0.0; p.take_profit = 30.0;
                            p.lookback_idx = l_idx; p.top_k_idx = k_idx;
                            grid_A.push_back(p);
                        }
                    }
                }
            }
        }
    }
    
    auto res_A = run_grid(grid_A);
    for (int i=0; i<min(5, (int)res_A.size()); ++i) {
        cout << "Top " << i+1 << " Monthly: M(" << res_A[i].params.m_fast << "," 
             << res_A[i].params.m_slow << "," << res_A[i].params.m_signal << ") mode=" 
             << res_A[i].params.trend_mode << " Lookback=" << g_lookbacks[res_A[i].params.lookback_idx] << "d"
             << " TopK=" << g_top_ks[res_A[i].params.top_k_idx]
             << " -> Train: " << fixed << setprecision(2) 
             << res_A[i].train_avg_annualized << "%\n";
    }
    
    // Phase B
    cout << "\n--- Phase B: Weekly Signal Optimization ---\n";
    vector<StrategyParams> grid_B;
    for (int i=0; i<min(5, (int)res_A.size()); ++i) {
        auto p = res_A[i].params;
        for (int wf : {6, 8, 10, 12, 14, 16, 18}) {
            for (int ws : {15, 20, 25, 30, 35, 40, 45}) {
                for (int wsig : {3, 5, 7, 9}) {
                    for (int bm : {0, 1}) {
                        for (int sm : {0, 1, 2}) {
                            for (double tr : {0.0, 15.0, 20.0}) {
                                for (double tp : {0.0, 20.0, 30.0, 50.0}) {
                                    p.w_fast = wf; p.w_slow = ws; p.w_signal = wsig;
                                    p.buy_mode = bm; p.sell_mode = sm; p.trailing_stop = tr; p.take_profit = tp;
                                    grid_B.push_back(p);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    auto res_B = run_grid(grid_B);
    for (int i=0; i<min(20, (int)res_B.size()); ++i) {
        cout << "Top " << i+1 << " Overall -> Train: " << fixed << setprecision(2) 
             << res_B[i].train_avg_annualized << "% | Test: " << res_B[i].test_avg_annualized << "%\n"
             << "  M(" << res_B[i].params.m_fast << "," << res_B[i].params.m_slow << "," << res_B[i].params.m_signal 
             << " md" << res_B[i].params.trend_mode << ") W(" << res_B[i].params.w_fast << "," 
             << res_B[i].params.w_slow << "," << res_B[i].params.w_signal << " b" << res_B[i].params.buy_mode 
             << " s" << res_B[i].params.sell_mode << ") Tr=" << res_B[i].params.trailing_stop << " Tp=" << res_B[i].params.take_profit 
             << " Lk=" << g_lookbacks[res_B[i].params.lookback_idx] << "d K=" << g_top_ks[res_B[i].params.top_k_idx] << "\n";
    }
    
    // Output and CSV
    if (res_B.empty()) return 0;
    auto best = res_B[0];
    
    ofstream csv("etf_macd_v8_results.csv");
    csv << "Code,Name,Sector,TestAnnualized,TestTrades,TestWinRate\n";
    for (const auto& kv : best.etf_results) {
        double wr = kv.second.trades > 0 ? (double)kv.second.win_trades / kv.second.trades * 100.0 : 0;
        string sec = "unknown";
        for(auto& e: etfs) { if(e.code == kv.first) { sec = e.sector; break; } }
        csv << kv.first << "," << kv.second.trade_list.front().etf_name << "," << sec << ","
            << fixed << setprecision(2) << kv.second.annualized << "," 
            << kv.second.trades << "," << wr << "%\n";
    }
    
    cout << "\nComparison:\n";
    cout << "  V5-W    Train 25.55%  Test 16.00%  (single weekly MACD)\n";
    cout << "  V7      Train 199%*   Test 31.14%  (monthly trend + weekly)\n";
    cout << "  V8      Train " << fixed << setprecision(2) << best.train_avg_annualized << "%  Test " << fixed << setprecision(2) << best.test_avg_annualized << "%  (V7 + sector rotation)\n";

    return 0;
}
