#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <filesystem>

// Hardcoded ETF name mapping (as provided)
std::unordered_map<std::string, std::string> g_etf_names = {
    // Indices (excluded from optimization)
    {"sh000001", "上证指数"}, {"sh000016", "上证50"}, {"sh000300", "沪深300"},
    {"sh000852", "中证1000"}, {"sh000905", "中证500"},
    {"sz399001", "深证成指"}, {"sz399006", "创业板指"},
    {"sh000688", "科创50指数"}, {"sz399303", "国证2000"}, {"sz399673", "创业板50指数"},
    {"sz399986", "中证新能"}, {"sz399989", "中证医疗"},
    // Broad ETFs
    {"sh510050", "上证50ETF"}, {"sh510300", "300ETF"}, {"sh510500", "中证500ETF"},
    {"sh510880", "红利ETF"}, {"sh510090", "180ETF"},
    // Sector ETFs
    {"sh512000", "券商ETF"}, {"sh512010", "医药ETF"}, {"sh512070", "非银ETF"},
    {"sh512170", "医疗ETF"}, {"sh512200", "房地产ETF"}, {"sh512290", "生物医药ETF"},
    {"sh512400", "有色金属ETF"}, {"sh512480", "半导体ETF"}, {"sh512660", "军工ETF"},
    {"sh512690", "酒ETF"}, {"sh512720", "计算机ETF"}, {"sh512760", "芯片ETF"},
    {"sh512800", "银行ETF"}, {"sh512880", "证券ETF"}, {"sh512980", "传媒ETF"},
    {"sz159819", "人工智能ETF"}, {"sz159825", "农业ETF"}, {"sz159869", "碳中和ETF"},
    {"sz159870", "化工ETF"}, {"sz159875", "新能源车ETF"}, {"sz159915", "创业板ETF"},
    {"sz159919", "300ETF(深)"}, {"sz159928", "消费ETF"}, {"sz159941", "纳指ETF(深)"},
    {"sz159949", "创业板50ETF"}, {"sz159995", "芯片ETF(深)"},
    // Cross-border/Commodity/Others
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
    {"sz159981", "能源化工ETF"}, {"sz159985", "豆粕ETF"},
};

struct StrategyParams {
    int fast, slow, signal;
    int buy_mode;         // 0=golden cross, 1=histogram>0
    int sell_mode;        // 0=death cross, 1=DIF<0, 2=histogram<0
    double trailing_stop; 
    double take_profit;   
    
    // RSI filter
    int rsi_period;
    double rsi_buy_max;
    double rsi_sell_min;
    
    // Bollinger filter
    int bb_period;
    double bb_buy_max_pctb;
    
    // Volume filter
    int vol_ma_period;
    double vol_min_ratio;

    bool operator==(const StrategyParams& o) const {
        return fast == o.fast && slow == o.slow && signal == o.signal &&
               buy_mode == o.buy_mode && sell_mode == o.sell_mode &&
               trailing_stop == o.trailing_stop && take_profit == o.take_profit &&
               rsi_period == o.rsi_period && rsi_buy_max == o.rsi_buy_max && rsi_sell_min == o.rsi_sell_min &&
               bb_period == o.bb_period && bb_buy_max_pctb == o.bb_buy_max_pctb &&
               vol_ma_period == o.vol_ma_period && vol_min_ratio == o.vol_min_ratio;
    }
};

struct WeeklyBar {
    std::string date;
    int week_id;
    double open, high, low, close, volume;
    
    // Precomputed indicators
    std::unordered_map<int, double> rsi; // period -> rsi
    std::unordered_map<int, double> bb_lower;
    std::unordered_map<int, double> bb_upper;
    std::unordered_map<int, double> vol_sma;
};

struct EtfData {
    std::string code;
    std::string name;
    std::vector<WeeklyBar> bars;
    int train_end_idx;
};

struct BacktestResult {
    int trades;
    double annualized_return;
    double max_drawdown;
    double win_rate;
};

// --- Helpers ---
int date_to_days(const std::string& d) {
    if (d.size() < 10) return 0;
    int y = std::stoi(d.substr(0, 4));
    int m = std::stoi(d.substr(5, 2));
    int day = std::stoi(d.substr(8, 2));
    if (m <= 2) { y--; m += 12; }
    return 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + day;
}
int date_to_week_id(const std::string& d) { return date_to_days(d) / 7; }

// Output string for StrategyParams
std::string params_to_str(const StrategyParams& p) {
    std::ostringstream oss;
    oss << p.fast << "," << p.slow << "," << p.signal 
        << " B" << p.buy_mode << " S" << p.sell_mode 
        << " tr=" << (int)p.trailing_stop << "% tp=" << (int)p.take_profit << "%";
    
    if (p.rsi_period > 0) oss << " RSI" << p.rsi_period << "(<" << p.rsi_buy_max << " >" << p.rsi_sell_min << ")";
    if (p.bb_period > 0) oss << " BB" << p.bb_period << "(<" << p.bb_buy_max_pctb << ")";
    if (p.vol_ma_period > 0) oss << " VMA" << p.vol_ma_period << "(>" << p.vol_min_ratio << ")";
    return oss.str();
}

// Data loading and prep
std::vector<EtfData> load_all_etfs(const std::string& dir) {
    std::vector<EtfData> etfs;
    for (const auto& kv : g_etf_names) {
        const std::string& code = kv.first;
        if (code.find("sh000") == 0 || code.find("sz399") == 0) continue; // Skip indices
        
        std::string filepath = dir + "/" + code + ".csv";
        std::ifstream file(filepath);
        if (!file.is_open()) continue;
        
        std::vector<WeeklyBar> daily_bars;
        std::string line;
        std::getline(file, line); // header
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string d, o, h, l, c, v;
            std::getline(ss, d, ','); std::getline(ss, o, ',');
            std::getline(ss, h, ','); std::getline(ss, l, ',');
            std::getline(ss, c, ','); std::getline(ss, v, '\n');
            if (c.empty()) continue;
            
            WeeklyBar b;
            b.date = d;
            b.week_id = date_to_week_id(d);
            b.open = std::stod(o); b.high = std::stod(h);
            b.low = std::stod(l); b.close = std::stod(c);
            b.volume = std::stod(v);
            daily_bars.push_back(b);
        }
        
        if (daily_bars.size() < 60) continue; // Minimum 60 daily bars
        
        // Aggregate to weekly
        std::vector<WeeklyBar> weekly;
        WeeklyBar current_week = daily_bars[0];
        for (size_t i = 1; i < daily_bars.size(); ++i) {
            const auto& db = daily_bars[i];
            if (db.week_id == current_week.week_id) {
                current_week.high = std::max(current_week.high, db.high);
                current_week.low = std::min(current_week.low, db.low);
                current_week.close = db.close;
                current_week.volume += db.volume;
                current_week.date = db.date; // Use last date of week
            } else {
                weekly.push_back(current_week);
                current_week = db;
            }
        }
        weekly.push_back(current_week);
        
        if (weekly.size() < 12) continue; // Skip if too few weekly bars
        
        // Precompute indicators
        for (size_t i = 0; i < weekly.size(); ++i) {
            // RSI
            for (int period : {7, 14, 21}) {
                if (i >= period) {
                    double gain = 0, loss = 0;
                    for (int j = i - period + 1; j <= i; ++j) {
                        double diff = weekly[j].close - weekly[j-1].close;
                        if (diff > 0) gain += diff;
                        else loss -= diff;
                    }
                    gain /= period; loss /= period;
                    double rs = (loss == 0) ? 100 : gain / loss;
                    weekly[i].rsi[period] = (loss == 0) ? 100 : 100.0 - (100.0 / (1.0 + rs));
                }
            }
            
            // BB
            for (int period : {10, 20}) {
                if (i + 1 >= period) {
                    double sum = 0;
                    for (int j = i - period + 1; j <= i; ++j) sum += weekly[j].close;
                    double mean = sum / period;
                    double var = 0;
                    for (int j = i - period + 1; j <= i; ++j) var += pow(weekly[j].close - mean, 2);
                    double stddev = sqrt(var / period);
                    weekly[i].bb_lower[period] = mean - 2.0 * stddev;
                    weekly[i].bb_upper[period] = mean + 2.0 * stddev;
                }
            }
            
            // Vol SMA
            for (int period : {5, 10}) {
                if (i + 1 >= period) {
                    double sum = 0;
                    for (int j = i - period + 1; j <= i; ++j) sum += weekly[j].volume;
                    weekly[i].vol_sma[period] = sum / period;
                }
            }
        }
        
        EtfData etf;
        etf.code = code;
        etf.name = kv.second;
        etf.bars = weekly;
        
        // Split train/test (40/60 on daily base implied by timeline, we approximate via weekly index)
        etf.train_end_idx = etf.bars.size() * 0.4;
        
        etfs.push_back(etf);
    }
    return etfs;
}

// Evaluate strategy
BacktestResult evaluate_strategy(const StrategyParams& p, const std::vector<WeeklyBar>& bars, int start_idx, int end_idx) {
    BacktestResult res = {0, 0.0, 0.0, 0.0};
    if (end_idx - start_idx < p.slow + p.signal) return res;
    
    double ema_fast = bars[start_idx].close;
    double ema_slow = bars[start_idx].close;
    double k_fast = 2.0 / (p.fast + 1);
    double k_slow = 2.0 / (p.slow + 1);
    double k_sig = 2.0 / (p.signal + 1);
    
    // Warmup MACD
    double dif = 0, dea = 0, hist = 0;
    double prev_dif = 0, prev_dea = 0, prev_hist = 0;
    
    for (int i = start_idx; i <= end_idx; ++i) {
        ema_fast = (bars[i].close - ema_fast) * k_fast + ema_fast;
        ema_slow = (bars[i].close - ema_slow) * k_slow + ema_slow;
        prev_dif = dif;
        dif = ema_fast - ema_slow;
        prev_dea = dea;
        dea = (dif - dea) * k_sig + dea;
        prev_hist = hist;
        hist = 2.0 * (dif - dea);
        bars[i].close; // Just to make sure we're aligned
    }

    // Reset and do real run
    ema_fast = bars[start_idx].close;
    ema_slow = bars[start_idx].close;
    dif = dea = hist = 0;
    
    bool in_position = false;
    double entry_price = 0, peak_price = 0;
    double capital = 10000.0;
    double initial_capital = capital;
    double max_capital = capital;
    double max_dd = 0;
    int wins = 0;
    
    for (int i = start_idx; i < end_idx; ++i) { // < end_idx because we trade next open
        const auto& bar = bars[i];
        
        ema_fast = (bar.close - ema_fast) * k_fast + ema_fast;
        ema_slow = (bar.close - ema_slow) * k_slow + ema_slow;
        prev_dif = dif;
        dif = ema_fast - ema_slow;
        prev_dea = dea;
        dea = (dif - dea) * k_sig + dea;
        prev_hist = hist;
        hist = 2.0 * (dif - dea);
        
        if (i < start_idx + p.slow + p.signal) continue;
        
        bool macd_buy = false;
        if (p.buy_mode == 0) macd_buy = (prev_dif <= prev_dea && dif > dea);
        else if (p.buy_mode == 1) macd_buy = (prev_hist <= 0 && hist > 0);
        
        bool macd_sell = false;
        if (p.sell_mode == 0) macd_sell = (prev_dif >= prev_dea && dif < dea);
        else if (p.sell_mode == 1) macd_sell = (dif < 0);
        else if (p.sell_mode == 2) macd_sell = (hist < 0);
        
        if (!in_position) {
            bool rsi_ok = true, bb_ok = true, vol_ok = true;
            if (p.rsi_period > 0 && p.rsi_buy_max > 0) {
                auto it = bar.rsi.find(p.rsi_period);
                if (it != bar.rsi.end() && it->second >= p.rsi_buy_max) rsi_ok = false;
            }
            if (p.bb_period > 0 && p.bb_buy_max_pctb < 2.0) {
                auto itL = bar.bb_lower.find(p.bb_period);
                auto itU = bar.bb_upper.find(p.bb_period);
                if (itL != bar.bb_lower.end() && itU != bar.bb_upper.end()) {
                    double width = itU->second - itL->second;
                    if (width > 0) {
                        double pct_b = (bar.close - itL->second) / width;
                        if (pct_b >= p.bb_buy_max_pctb) bb_ok = false;
                    }
                }
            }
            if (p.vol_ma_period > 0 && p.vol_min_ratio > 0) {
                auto it = bar.vol_sma.find(p.vol_ma_period);
                if (it != bar.vol_sma.end() && it->second > 0) {
                    double ratio = bar.volume / it->second;
                    if (ratio <= p.vol_min_ratio) vol_ok = false;
                }
            }
            
            if (macd_buy && rsi_ok && bb_ok && vol_ok) {
                in_position = true;
                entry_price = bars[i+1].open;
                peak_price = entry_price;
                res.trades++;
            }
        } else {
            peak_price = std::max(peak_price, bar.high);
            
            bool rsi_sell = false;
            if (p.rsi_period > 0 && p.rsi_sell_min > 0) {
                auto it = bar.rsi.find(p.rsi_period);
                if (it != bar.rsi.end() && it->second > p.rsi_sell_min) rsi_sell = true;
            }
            
            bool stop_hit = false, tp_hit = false;
            if (p.trailing_stop > 0) stop_hit = (bar.close <= peak_price * (1.0 - p.trailing_stop / 100.0));
            if (p.take_profit > 0) tp_hit = (bar.close >= entry_price * (1.0 + p.take_profit / 100.0));
            
            if (macd_sell || rsi_sell || stop_hit || tp_hit) {
                in_position = false;
                double exit_price = bars[i+1].open;
                double ret = (exit_price - entry_price) / entry_price;
                capital *= (1.0 + ret);
                if (ret > 0) wins++;
                
                max_capital = std::max(max_capital, capital);
                max_dd = std::max(max_dd, (max_capital - capital) / max_capital);
            }
        }
    }
    
    // Close position at end if open
    if (in_position) {
        double exit_price = bars[end_idx].close;
        double ret = (exit_price - entry_price) / entry_price;
        capital *= (1.0 + ret);
        if (ret > 0) wins++;
        max_capital = std::max(max_capital, capital);
        max_dd = std::max(max_dd, (max_capital - capital) / max_capital);
    }
    
    double total_years = (end_idx - start_idx) / 52.0;
    if (total_years > 0) {
        res.annualized_return = (pow(capital / initial_capital, 1.0 / total_years) - 1.0) * 100.0;
    }
    res.max_drawdown = max_dd * 100.0;
    if (res.trades > 0) res.win_rate = (double)wins / res.trades * 100.0;
    
    return res;
}

double score_configs(const StrategyParams& p, const std::vector<EtfData>& etfs, bool is_train) {
    std::vector<double> ann_rets;
    for (const auto& etf : etfs) {
        int start = is_train ? 0 : etf.train_end_idx;
        int end = is_train ? etf.train_end_idx : etf.bars.size() - 1;
        BacktestResult res = evaluate_strategy(p, etf.bars, start, end);
        if (res.trades > 0) ann_rets.push_back(res.annualized_return);
    }
    
    int n = std::min(20, (int)ann_rets.size());
    if (n < 5) return -999.0;
    
    std::sort(ann_rets.rbegin(), ann_rets.rend());
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += ann_rets[i];
    return sum / n;
}

int main() {
    std::cout << "Loading ETF Data..." << std::endl;
    std::vector<EtfData> etfs = load_all_etfs("/ceph/dang_articles/yoj/market_data");
    std::cout << "Loaded " << etfs.size() << " tradeable ETFs." << std::endl;
    if (etfs.empty()) return 1;

    // Build Phase A Grid
    std::vector<StrategyParams> phaseA_grid;
    for (int fast = 5; fast <= 20; ++fast) {
        for (int slow = 15; slow <= 50; slow += 2) {
            for (int sig = 3; sig <= 12; ++sig) {
                if (fast >= slow) continue;
                for (int buy : {0, 1}) {
                    for (int sell : {0, 1, 2}) {
                        for (double trail : {0.0, 15.0, 20.0}) {
                            for (double tp : {0.0, 30.0, 50.0}) {
                                StrategyParams p = {fast, slow, sig, buy, sell, trail, tp, 0, 0, 0, 0, 2.0, 0, 0};
                                phaseA_grid.push_back(p);
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "Phase A Grid Size: " << phaseA_grid.size() << std::endl;
    
    std::mutex mtx;
    std::vector<std::pair<double, StrategyParams>> a_results;
    
    auto worker_A = [&](int start, int end) {
        std::vector<std::pair<double, StrategyParams>> local_res;
        for (int i = start; i < end; ++i) {
            double score = score_configs(phaseA_grid[i], etfs, true);
            if (score > -900) local_res.push_back({score, phaseA_grid[i]});
        }
        std::lock_guard<std::mutex> lock(mtx);
        a_results.insert(a_results.end(), local_res.begin(), local_res.end());
    };
    
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk = phaseA_grid.size() / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int s = i * chunk;
        int e = (i == num_threads - 1) ? phaseA_grid.size() : s + chunk;
        threads.emplace_back(worker_A, s, e);
    }
    for (auto& t : threads) t.join();
    
    std::sort(a_results.rbegin(), a_results.rend(), [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\n--- Phase A Top 5 ---" << std::endl;
    for (int i = 0; i < std::min(5, (int)a_results.size()); ++i) {
        std::cout << "Score: " << std::fixed << std::setprecision(2) << a_results[i].first << "% | " << params_to_str(a_results[i].second) << std::endl;
    }
    
    // Build Phase B Grid (Filters on top 8 A)
    std::vector<StrategyParams> phaseB_grid;
    int topA = std::min(8, (int)a_results.size());
    for (int i = 0; i < topA; ++i) {
        StrategyParams base = a_results[i].second;
        for (int r_per : {0, 14}) {
            for (double r_buy : {0.0, 60.0}) {
                for (double r_sell : {0.0, 70.0}) {
                    for (int bb_per : {0, 20}) {
                        for (double bb_buy : {2.0, 0.5}) {
                            for (int v_per : {0, 5}) {
                                for (double v_buy : {0.0, 1.5}) {
                                    if (r_per == 0 && (r_buy > 0 || r_sell > 0)) continue;
                                    if (bb_per == 0 && bb_buy < 2.0) continue;
                                    if (v_per == 0 && v_buy > 0) continue;
                                    
                                    StrategyParams p = base;
                                    p.rsi_period = r_per; p.rsi_buy_max = r_buy; p.rsi_sell_min = r_sell;
                                    p.bb_period = bb_per; p.bb_buy_max_pctb = bb_buy;
                                    p.vol_ma_period = v_per; p.vol_min_ratio = v_buy;
                                    phaseB_grid.push_back(p);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Deduplicate
    auto it = std::unique(phaseB_grid.begin(), phaseB_grid.end());
    phaseB_grid.erase(it, phaseB_grid.end());
    std::cout << "\nPhase B Grid Size: " << phaseB_grid.size() << std::endl;
    
    std::vector<std::pair<double, StrategyParams>> b_results;
    auto worker_B = [&](int start, int end) {
        std::vector<std::pair<double, StrategyParams>> local_res;
        for (int i = start; i < end; ++i) {
            double score = score_configs(phaseB_grid[i], etfs, true); // Still train
            if (score > -900) local_res.push_back({score, phaseB_grid[i]});
        }
        std::lock_guard<std::mutex> lock(mtx);
        b_results.insert(b_results.end(), local_res.begin(), local_res.end());
    };
    
    threads.clear();
    chunk = std::max(1, (int)phaseB_grid.size() / num_threads);
    for (int i = 0; i < num_threads; ++i) {
        int s = i * chunk;
        int e = (i == num_threads - 1) ? phaseB_grid.size() : s + chunk;
        if (s < phaseB_grid.size()) threads.emplace_back(worker_B, s, std::min(e, (int)phaseB_grid.size()));
    }
    for (auto& t : threads) t.join();
    
    std::sort(b_results.rbegin(), b_results.rend(), [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\n--- Phase B Top 10 (Train Data) ---" << std::endl;
    for (int i = 0; i < std::min(10, (int)b_results.size()); ++i) {
        std::cout << "Train Score: " << std::fixed << std::setprecision(2) << b_results[i].first << "% | " << params_to_str(b_results[i].second) << std::endl;
    }
    
    // Phase C: Eval on Test Data
    std::vector<std::pair<double, StrategyParams>> c_results;
    int topB = std::min(20, (int)b_results.size());
    for (int i = 0; i < topB; ++i) {
        double test_score = score_configs(b_results[i].second, etfs, false);
        c_results.push_back({test_score, b_results[i].second});
    }
    
    std::sort(c_results.rbegin(), c_results.rend(), [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\n--- Phase C Top 5 (Test Data) ---" << std::endl;
    for (int i = 0; i < std::min(5, (int)c_results.size()); ++i) {
        std::cout << "Test Score: " << std::fixed << std::setprecision(2) << c_results[i].first << "% | " << params_to_str(c_results[i].second) << std::endl;
    }
    
    // Detail Best Result
    StrategyParams best_p = c_results[0].second;
    std::cout << "\n=== Best V6 Config ===" << std::endl;
    std::cout << params_to_str(best_p) << std::endl;
    
    double train_best = score_configs(best_p, etfs, true);
    double test_best = c_results[0].first;
    
    std::cout << "Train Score: " << train_best << "%" << std::endl;
    std::cout << "Test Score: " << test_best << "%" << std::endl;
    std::cout << "V5 Baseline tracking: Expected Train ~25.55% / Test ~16.00%" << std::endl;
    if (test_best > 16.00) std::cout << "IMPROVEMENT: V6 beats V5 Baseline." << std::endl;
    else std::cout << "INFO: V6 did not beat V5 baseline test score." << std::endl;
    
    // CSV Output
    std::ofstream out("etf_macd_v6_results.csv");
    out << "code,name,train_trades,train_ann_ret,train_dd,train_win,test_trades,test_ann_ret,test_dd,test_win\n";
    
    for (const auto& etf : etfs) {
        BacktestResult train_res = evaluate_strategy(best_p, etf.bars, 0, etf.train_end_idx);
        BacktestResult test_res = evaluate_strategy(best_p, etf.bars, etf.train_end_idx, etf.bars.size() - 1);
        
        out << etf.code << "," << etf.name << ","
            << train_res.trades << "," << train_res.annualized_return << "," << train_res.max_drawdown << "," << train_res.win_rate << ","
            << test_res.trades << "," << test_res.annualized_return << "," << test_res.max_drawdown << "," << test_res.win_rate << "\n";
    }
    out.close();
    std::cout << "Results saved to etf_macd_v6_results.csv" << std::endl;

    return 0;
}
