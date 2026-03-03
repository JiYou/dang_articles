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
    int first_daily_idx;
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
    vector<double> global_open;
    vector<double> global_high;
};

struct StrategyParams {
    int m_fast, m_slow, m_signal;
    int trend_mode;  
    
    int w_fast, w_slow, w_signal;
    int buy_mode;   
    int sell_mode;  
    
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

struct BacktestResult {
    double sharpe;
    double calmar;
    double combined_score;
    double mdd;
    double total_pnl_pct = 0.0;
    double annualized = 0.0;
    int trades = 0;
    vector<Trade> trade_list;
    vector<double> equity_curve;
};

struct ScoreResult {
    StrategyParams params;
    BacktestResult train_res;
    BacktestResult test_res;
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

unordered_map<string, int> g_date_to_idx;
vector<string> global_dates;

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

// --- Position State ---
struct Position {
    int etf_idx;
    double entry_price;
    double highest_price;
    string entry_date;
    double shares;
    int entry_daily_idx;
};

const double RISK_FREE_RATE = 0.02;

// --- Backtest Evaluation ---
BacktestResult run_portfolio_backtest(const vector<ETFData>& etfs, const StrategyParams& params, bool is_test) {
    int num_etfs = etfs.size();
    
    // Precompute MACD signals
    vector<vector<double>> w_dif(num_etfs), w_dea(num_etfs), w_hist(num_etfs);
    vector<vector<double>> m_dif(num_etfs), m_dea(num_etfs), m_hist(num_etfs);
    vector<vector<int>> w_to_m(num_etfs);
    vector<vector<int>> d_to_w(num_etfs);
    
    for (int i = 0; i < num_etfs; ++i) {
        const auto& weekly = is_test ? etfs[i].test_weekly : etfs[i].train_weekly;
        const auto& monthly = is_test ? etfs[i].test_monthly : etfs[i].train_monthly;
        if (weekly.size() < 20 || monthly.size() < 5) continue;
        
        compute_macd(monthly, params.m_fast, params.m_slow, params.m_signal, m_dif[i], m_dea[i], m_hist[i]);
        compute_macd(weekly, params.w_fast, params.w_slow, params.w_signal, w_dif[i], w_dea[i], w_hist[i]);
        
        w_to_m[i].assign(weekly.size(), -1);
        int m_idx = 0;
        for (size_t w = 0; w < weekly.size(); ++w) {
            while (m_idx + 1 < (int)monthly.size() && 
                   monthly[m_idx].last_daily_idx < weekly[w].first_daily_idx)
                m_idx++;
            if (m_idx < (int)monthly.size() &&
                monthly[m_idx].first_daily_idx <= weekly[w].last_daily_idx)
                w_to_m[i][w] = m_idx;
        }
        
        d_to_w[i].assign(global_dates.size(), -1);
        for(size_t w = 0; w < weekly.size(); ++w) {
            for(int d = weekly[w].first_daily_idx; d <= weekly[w].last_daily_idx; ++d) {
                int glob_d = g_date_to_idx[etfs[i].daily[d].date];
                d_to_w[i][glob_d] = w;
            }
        }
    }
    
    double capital = 1000000.0;
    double cash = capital;
    vector<Position> active_positions;
    BacktestResult res;
    
    int start_idx = global_dates.size(), end_idx = 0;
    for (int i = 0; i < num_etfs; ++i) {
        if (etfs[i].split_idx < etfs[i].daily.size()) {
            int first_d = is_test ? etfs[i].split_idx : 0;
            int last_d = is_test ? etfs[i].daily.size() - 1 : etfs[i].split_idx - 1;
            if(first_d <= last_d) {
                start_idx = min(start_idx, g_date_to_idx[etfs[i].daily[first_d].date]);
                end_idx = max(end_idx, g_date_to_idx[etfs[i].daily[last_d].date]);
            }
        }
    }
    
    if (start_idx >= end_idx) return res;
    
    res.equity_curve.reserve(end_idx - start_idx + 1);
    
    for (int d = start_idx; d <= end_idx; ++d) {
        string cur_date = global_dates[d];
        
        // Update highest prices and handle sells
        vector<Position> remaining_positions;
        for (auto& pos : active_positions) {
            int e_idx = pos.etf_idx;
            const auto& etf = etfs[e_idx];
            int w_idx = d_to_w[e_idx][d];
            
            if (w_idx == -1) {
                // Keep holding if no data, update with global_high
                if (etf.global_high[d] > pos.highest_price) pos.highest_price = etf.global_high[d];
                remaining_positions.push_back(pos);
                continue;
            }
            
            const auto& weekly = is_test ? etf.test_weekly : etf.train_weekly;
            if (w_idx < max(params.w_slow, params.m_slow)) {
                remaining_positions.push_back(pos);
                continue;
            }
            
            // Find current close for SL/TP evaluation
            double cur_close = etf.global_close[d];
            double cur_high = etf.global_high[d];
            if (cur_high > pos.highest_price) pos.highest_price = cur_high;
            
            bool sell = false;
            // End of week check
            if (d == g_date_to_idx[etf.daily[weekly[w_idx].last_daily_idx].date]) {
                if (params.sell_mode == 0) sell = (w_dif[e_idx][w_idx] < w_dea[e_idx][w_idx] && w_dif[e_idx][w_idx-1] >= w_dea[e_idx][w_idx-1]);
                else if (params.sell_mode == 1) sell = (w_dif[e_idx][w_idx] < 0);
                else if (params.sell_mode == 2) sell = (w_hist[e_idx][w_idx] < 0 && w_hist[e_idx][w_idx-1] >= 0);
            }
            
            if (!sell && params.trailing_stop > 0) {
                if (cur_close <= pos.highest_price * (1.0 - params.trailing_stop/100.0)) sell = true;
            }
            if (!sell && params.take_profit > 0) {
                if (cur_close >= pos.entry_price * (1.0 + params.take_profit/100.0)) sell = true;
            }
            
            if (sell || d == end_idx) {
                // Execute next day open (or current close if last day)
                double exit_price = cur_close; 
                if (d < end_idx) {
                    double nxt_open = etf.global_open[d+1];
                    if (nxt_open > 0) exit_price = nxt_open;
                }
                
                double value = pos.shares * exit_price;
                cash += value;
                
                Trade t;
                t.etf_code = etf.code;
                t.etf_name = etf.name;
                t.entry_date = pos.entry_date;
                t.exit_date = cur_date; // Approx
                t.entry_price = pos.entry_price;
                t.exit_price = exit_price;
                t.pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0;
                res.trade_list.push_back(t);
                res.trades++;
            } else {
                remaining_positions.push_back(pos);
            }
        }
        active_positions = remaining_positions;
        
        // Record equity today
        double daily_equity = cash;
        for (const auto& pos : active_positions) {
            double p = etfs[pos.etf_idx].global_close[d];
            if (p <= 0) p = pos.entry_price; // Fallback
            daily_equity += pos.shares * p;
        }
        res.equity_curve.push_back(daily_equity);
        
        // Look for buys
        if (active_positions.size() < params.max_positions) {
            for (int i = 0; i < num_etfs && active_positions.size() < params.max_positions; ++i) {
                int w_idx = d_to_w[i][d];
                if (w_idx <= max(params.w_slow, params.m_slow)) continue;
                
                const auto& etf = etfs[i];
                const auto& weekly = is_test ? etf.test_weekly : etf.train_weekly;
                
                // Only buy on the last day of the week
                if (d != g_date_to_idx[etf.daily[weekly[w_idx].last_daily_idx].date]) continue;
                
                bool already_held = false;
                for (const auto& pos : active_positions) {
                    if (pos.etf_idx == i) { already_held = true; break; }
                }
                if (already_held) continue;
                
                int mi = w_to_m[i][w_idx];
                if (mi <= 0) continue;
                
                bool m_bull = false;
                if (params.trend_mode == 0) m_bull = m_dif[i][mi] > 0;
                else if (params.trend_mode == 1) m_bull = m_dif[i][mi] > m_dea[i][mi];
                else if (params.trend_mode == 2) m_bull = m_hist[i][mi] > 0;
                else if (params.trend_mode == 3) m_bull = m_dif[i][mi] > m_dif[i][mi-1];
                
                bool w_buy = false;
                if (params.buy_mode == 0) w_buy = (w_dif[i][w_idx] > w_dea[i][w_idx] && w_dif[i][w_idx-1] <= w_dea[i][w_idx-1]);
                else if (params.buy_mode == 1) w_buy = (w_hist[i][w_idx] > 0 && w_hist[i][w_idx-1] <= 0);
                
                if (m_bull && w_buy) {
                    double entry_price = etf.global_close[d];
                    if (d < end_idx) {
                        double nxt_open = etf.global_open[d+1];
                        if (nxt_open > 0) entry_price = nxt_open;
                    }
                    if (entry_price <= 0) continue;
                    
                    double alloc = daily_equity / params.max_positions;
                    if (alloc > cash) alloc = cash; // Cannot use margin
                    
                    double shares = alloc / entry_price;
                    cash -= shares * entry_price;
                    
                    Position p;
                    p.etf_idx = i;
                    p.entry_price = entry_price;
                    p.highest_price = entry_price;
                    p.entry_date = (d < end_idx) ? global_dates[d+1] : cur_date;
                    p.shares = shares;
                    active_positions.push_back(p);
                }
            }
        }
    }
    
    // Evaluate metrics
    double final_equity = res.equity_curve.empty() ? capital : res.equity_curve.back();
    double yrs = (double)res.equity_curve.size() / 252.0;
    
    if (res.equity_curve.size() > 1 && res.trades > 0) {
        vector<double> daily_rets(res.equity_curve.size() - 1);
        double max_eq = res.equity_curve[0];
        double mdd = 0;
        double sum_ret = 0;
        
        for (size_t i = 1; i < res.equity_curve.size(); ++i) {
            double r = (res.equity_curve[i] - res.equity_curve[i-1]) / res.equity_curve[i-1];
            daily_rets[i-1] = r;
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
        
        double ann_ret = (pow(final_equity / capital, 1.0 / yrs) - 1.0);
        double ann_vol = std_ret * sqrt(252.0);
        
        res.sharpe = ann_vol > 0 ? (ann_ret - RISK_FREE_RATE) / ann_vol : 0;
        res.mdd = mdd;
        res.calmar = mdd > 0 ? ann_ret / mdd : 0;
        if (mdd == 0 && ann_ret > 0) res.calmar = 100.0; // Arbitrary high calmar
        res.annualized = ann_ret * 100.0;
        
        // normalize combined score
        res.combined_score = 0.5 * res.sharpe + 0.5 * min(res.calmar, 10.0);
    } else {
        res.sharpe = -100;
        res.calmar = -100;
        res.combined_score = -100;
        res.mdd = 0;
        res.annualized = 0;
    }
    
    return res;
}

ScoreResult eval_params(const vector<ETFData>& etfs, const StrategyParams& params) {
    ScoreResult sr;
    sr.params = params;
    sr.train_res = run_portfolio_backtest(etfs, params, false);
    sr.test_res = run_portfolio_backtest(etfs, params, true);
    return sr;
}

int main() {
    string data_dir = "/ceph/dang_articles/yoj/market_data/";
    vector<ETFData> etfs;
    
    // Load Data
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        string path = entry.path().string();
        if (path.find(".csv") == string::npos) continue;
        
        string code = entry.path().stem().string();
        if (code.find("sh000") == 0 || code.find("sz399") == 0) continue; 
        
        ifstream file(path);
        string line;
        getline(file, line); 
        
        ETFData etf;
        etf.code = code;
        etf.name = g_etf_names.count(code) ? g_etf_names[code] : code;
        
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
    
    cout << "Loaded " << etfs.size() << " tradeable ETFs.\n";
    
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
            if (etf.global_close[i] > 0) {
                last_close = etf.global_close[i];
            } else {
                etf.global_close[i] = last_close; 
                // Don't forward fill open and high, they should just be 0 so we don't trade on empty days
            }
        }
    }

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
        return results;
    };
    
    vector<StrategyParams> phase_a_grid;
    vector<int> m_fasts_a = {6, 8, 10, 12, 14};
    vector<int> m_slows_a = {15, 17, 20, 24, 28};
    vector<int> m_sigs_a = {3, 5};
    vector<int> t_modes_a = {0, 1, 2, 3};
    vector<double> trailings = {0.0, 15.0, 20.0};
    vector<double> tps = {0.0, 20.0, 30.0, 50.0};
    vector<int> max_pos_opts = {3, 5, 10, 15};
    
    for (int mf : m_fasts_a) {
        for (int ms : m_slows_a) {
            if (mf >= ms) continue; // logical optimization
            for (int msg : m_sigs_a) {
                for (int tm : t_modes_a) {
                    for (double tr : trailings) {
                        for (double tp : tps) {
                            for (int maxp : max_pos_opts) {
                                StrategyParams p;
                                p.m_fast = mf; p.m_slow = ms; p.m_signal = msg; p.trend_mode = tm;
                                p.w_fast = 8; p.w_slow = 30; p.w_signal = 3;
                                p.buy_mode = 0; p.sell_mode = 2;
                                p.trailing_stop = tr; p.take_profit = tp;
                                p.max_positions = maxp;
                                phase_a_grid.push_back(p);
                            }
                        }
                    }
                }
            }
        }
    }
    
    cout << "Running Phase A grid of size " << phase_a_grid.size() << "...\n";
    auto results_a = run_grid(phase_a_grid);
    
    // Sort Phase A by combined score
    sort(results_a.begin(), results_a.end(), [](const ScoreResult& a, const ScoreResult& b){
        return a.train_res.combined_score > b.train_res.combined_score;
    });
    
    struct MConfig { int mf, ms, msg, tm; };
    vector<MConfig> top_m_configs;
    for(const auto& r : results_a) {
        bool exists = false;
        for(const auto& mc : top_m_configs) {
            if (mc.mf == r.params.m_fast && mc.ms == r.params.m_slow && 
                mc.msg == r.params.m_signal && mc.tm == r.params.trend_mode) {
                exists = true;
                break;
            }
        }
        if(!exists) {
            top_m_configs.push_back({r.params.m_fast, r.params.m_slow, r.params.m_signal, r.params.trend_mode});
            if(top_m_configs.size() == 5) break;
        }
    }
    
    vector<StrategyParams> phase_b_grid;
    vector<int> w_fasts_b = {6, 8, 10, 12};
    vector<int> w_slows_b = {25, 30, 35, 40};
    vector<int> w_sigs_b = {3, 5};
    vector<int> buy_modes_b = {0, 1};
    vector<int> sell_modes_b = {0, 1, 2};
    
    for (const auto& mc : top_m_configs) {
        for (int wf : w_fasts_b) {
            for (int ws : w_slows_b) {
                if (wf >= ws) continue; // logical optimization
                for (int wsig : w_sigs_b) {
                    for (int bm : buy_modes_b) {
                        for (int sm : sell_modes_b) {
                            for (double tr : trailings) {
                                for (double tp : tps) {
                                    for (int maxp : max_pos_opts) {
                                        StrategyParams p;
                                        p.m_fast = mc.mf; p.m_slow = mc.ms; p.m_signal = mc.msg; p.trend_mode = mc.tm;
                                        p.w_fast = wf; p.w_slow = ws; p.w_signal = wsig;
                                        p.buy_mode = bm; p.sell_mode = sm;
                                        p.trailing_stop = tr; p.take_profit = tp;
                                        p.max_positions = maxp;
                                        phase_b_grid.push_back(p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    cout << "Running Phase B grid of size " << phase_b_grid.size() << "...\n";
    auto results = run_grid(phase_b_grid);
    
    // Function to calculate win rate
    auto calc_win_rate = [](const vector<Trade>& trades) {
        if(trades.empty()) return 0.0;
        int wins = 0;
        for(const auto& t : trades) if(t.pnl_pct > 0) wins++;
        return (double)wins / trades.size() * 100.0;
    };
    
    // Sort and print helper
    auto print_top_3 = [&](const string& name, auto compare_func) {
        cout << "\n=== Top 3 by " << name << " ===\n";
        sort(results.begin(), results.end(), compare_func);
        for(int i = 0; i < min(3, (int)results.size()); ++i) {
            const auto& sr = results[i];
            cout << "Rank " << i+1 << " params: M(" << sr.params.m_fast << "," << sr.params.m_slow << "," << sr.params.m_signal << ") md=" << sr.params.trend_mode
                 << " W(" << sr.params.w_fast << "," << sr.params.w_slow << "," << sr.params.w_signal << ") b" << sr.params.buy_mode << " s" << sr.params.sell_mode
                 << " tr=" << sr.params.trailing_stop << " tp=" << sr.params.take_profit << " MaxPos=" << sr.params.max_positions << "\n";
            
            double tr_win_rate = calc_win_rate(sr.train_res.trade_list);
            double te_win_rate = calc_win_rate(sr.test_res.trade_list);
            
            cout << "Train: Sharpe=" << fixed << setprecision(2) << sr.train_res.sharpe << "  Return=" << sr.train_res.annualized << "%  MDD=" << sr.train_res.mdd*100.0 
                 << "%  Calmar=" << sr.train_res.calmar << "  Trades=" << sr.train_res.trades << "  WinRate=" << tr_win_rate << "%\n";
            cout << "Test:  Sharpe=" << fixed << setprecision(2) << sr.test_res.sharpe << "  Return=" << sr.test_res.annualized << "%  MDD=" << sr.test_res.mdd*100.0 
                 << "%  Calmar=" << sr.test_res.calmar << "  Trades=" << sr.test_res.trades << "  WinRate=" << te_win_rate << "%\n";
        }
        return results.front();
    };
    
    auto best_sharpe = print_top_3("Sharpe", [](const ScoreResult& a, const ScoreResult& b){ return a.train_res.sharpe > b.train_res.sharpe; });
    auto best_calmar = print_top_3("Calmar", [](const ScoreResult& a, const ScoreResult& b){ return a.train_res.calmar > b.train_res.calmar; });
    auto best_combined = print_top_3("Combined", [](const ScoreResult& a, const ScoreResult& b){ return a.train_res.combined_score > b.train_res.combined_score; });
    
    cout << "\nComparison (Test period):\n";
    cout << "  V8 Top20 Avg Ann: 33.84%\n";
    cout << "  Old V10 Sharpe-opt: Return 20.96%  Sharpe 0.81  MDD 17.04%\n";
    cout << "  New V10 Sharpe-opt:   Return " << fixed << setprecision(2) << best_sharpe.test_res.annualized << "%  Sharpe " << best_sharpe.test_res.sharpe << "  MDD " << best_sharpe.test_res.mdd*100.0 << "%\n";
    cout << "  New V10 Calmar-opt:   Return " << fixed << setprecision(2) << best_calmar.test_res.annualized << "%  Sharpe " << best_calmar.test_res.sharpe << "  MDD " << best_calmar.test_res.mdd*100.0 << "%\n";
    cout << "  New V10 Combined-opt: Return " << fixed << setprecision(2) << best_combined.test_res.annualized << "%  Sharpe " << best_combined.test_res.sharpe << "  MDD " << best_combined.test_res.mdd*100.0 << "%\n";
    ofstream csv("etf_macd_v10_results.csv");
    csv << "Code,Name,TestReturn,TestTrades,HoldDays,Contribution\n";
    for (const auto& t : best_sharpe.test_res.trade_list) {
        csv << t.etf_code << "," << t.etf_name << "," << t.pnl_pct << ",1,0,0\n";
    }
    
    return 0;
}
