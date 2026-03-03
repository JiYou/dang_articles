/*
 * MACD Strategy Optimizer (Grid Search)
 *
 * Compiles with: g++ -O3 -std=c++17 -pthread -o macd_optimizer macd_optimizer.cpp
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// Data structures
// ============================================================

struct DailyBar {
    std::string date;
    double open;
    double close;
    double high;
    double low;
    double volume;
};

struct MonthlyBar {
    std::string year_month;
    std::string first_date;
    std::string last_date;
    double open;
    double close;
    double high;
    double low;
    double volume;
    int first_daily_idx;
    int last_daily_idx;
    // MACD values
    double dif;
    double dea;
    double macd_hist;
};

struct StockData {
    std::string code;
    std::string name;
    std::vector<DailyBar> daily_bars;
    std::vector<MonthlyBar> train_monthly;
    std::vector<MonthlyBar> test_monthly;
    int split_idx; // 40% split
};

struct TradeRecord {
    std::string buy_date;
    std::string sell_date;
    double buy_price;
    double sell_price;
    double return_pct;
    int holding_days;
};

struct StrategyParams {
    int fast;
    int slow;
    int signal;
    double buy_dif_threshold;
    double sell_dif_threshold;
    int market_ema_period;
    double max_loss_pct;
    int max_hold_months;
};

struct EvalResult {
    int trades;
    double cumulative_return_pct;
    double annualized_return_pct;
    double win_rate;
    double buy_hold_return_pct;
    std::string date_start;
    std::string date_end;
};

// ============================================================
// Global Data
// ============================================================

std::unordered_map<std::string, std::string> g_stock_names;
std::vector<StockData> g_stocks;
std::vector<DailyBar> g_market_bars;
std::unordered_map<int, std::unordered_map<std::string, double>> g_market_ema;

// ============================================================
// Helpers
// ============================================================

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        result.push_back(cell);
    }
    return result;
}

void load_stock_names(const std::string& data_dir) {
    std::vector<std::string> paths = {
        data_dir + "/all_stock.csv",
        data_dir + "/../all_stock.csv",
        "../all_stock.csv"
    };

    std::ifstream file;
    for (const auto& p : paths) {
        file.open(p);
        if (file.is_open()) {
            std::cout << "Loaded all_stock.csv from: " << p << "\n";
            break;
        }
    }

    if (!file.is_open()) {
        std::cerr << "Warning: Could not find all_stock.csv\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.size() >= 3 && line[0] == '\xEF' && line[1] == '\xBB' && line[2] == '\xBF') {
            line = line.substr(3); // Remove BOM
        }
        if (line.empty()) continue;
        auto tokens = split_csv(line);
        if (tokens.size() >= 2) {
            std::string code = tokens[0];
            std::string name = tokens[1];
            if (name.back() == '\r') name.pop_back();
            if (g_stock_names.find(code) == g_stock_names.end()) {
                g_stock_names[code] = name;
            }
        }
    }
}

void load_market_data(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open market data: " << filepath << "\n";
        return;
    }

    std::string line;
    std::getline(file, line); // Skip header
    while (std::getline(file, line)) {
        auto tokens = split_csv(line);
        if (tokens.size() < 6) continue;
        DailyBar bar;
        bar.date = tokens[0];
        try {
            bar.open = std::stod(tokens[1]);
            bar.high = std::stod(tokens[2]);
            bar.low = std::stod(tokens[3]);
            bar.close = std::stod(tokens[4]);
            bar.volume = std::stod(tokens[5]);
            g_market_bars.push_back(bar);
        } catch (...) {}
    }

    std::sort(g_market_bars.begin(), g_market_bars.end(), [](const DailyBar& a, const DailyBar& b) {
        return a.date < b.date;
    });

    std::cout << "Loaded market data: " << g_market_bars.size() << " days.\n";

    // Precompute Market EMAs
    std::vector<int> periods = {20, 60, 120};
    for (int p : periods) {
        double mult = 2.0 / (p + 1);
        double current_ema = 0;
        for (size_t i = 0; i < g_market_bars.size(); ++i) {
            if (i == 0) {
                current_ema = g_market_bars[i].close;
            } else {
                current_ema = (g_market_bars[i].close - current_ema) * mult + current_ema;
            }
            g_market_ema[p][g_market_bars[i].date] = current_ema;
        }
    }
}

std::vector<MonthlyBar> aggregate_monthly(const std::vector<DailyBar>& daily, int start_idx, int end_idx) {
    std::vector<MonthlyBar> monthly;
    if (start_idx >= end_idx) return monthly;

    MonthlyBar current;
    current.year_month = daily[start_idx].date.substr(0, 7);
    current.first_date = daily[start_idx].date;
    current.last_date = daily[start_idx].date;
    current.open = daily[start_idx].open;
    current.close = daily[start_idx].close;
    current.high = daily[start_idx].high;
    current.low = daily[start_idx].low;
    current.volume = daily[start_idx].volume;
    current.first_daily_idx = start_idx;
    current.last_daily_idx = start_idx;

    for (int i = start_idx + 1; i < end_idx; ++i) {
        std::string ym = daily[i].date.substr(0, 7);
        if (ym == current.year_month) {
            current.last_date = daily[i].date;
            current.close = daily[i].close;
            current.high = std::max(current.high, daily[i].high);
            current.low = std::min(current.low, daily[i].low);
            current.volume += daily[i].volume;
            current.last_daily_idx = i;
        } else {
            monthly.push_back(current);
            current.year_month = ym;
            current.first_date = daily[i].date;
            current.last_date = daily[i].date;
            current.open = daily[i].open;
            current.close = daily[i].close;
            current.high = daily[i].high;
            current.low = daily[i].low;
            current.volume = daily[i].volume;
            current.first_daily_idx = i;
            current.last_daily_idx = i;
        }
    }
    monthly.push_back(current);
    return monthly;
}

void compute_macd(std::vector<MonthlyBar>& bars, int fast, int slow, int signal) {
    if (bars.empty()) return;
    double mult_fast = 2.0 / (fast + 1);
    double mult_slow = 2.0 / (slow + 1);
    double mult_sig = 2.0 / (signal + 1);

    double ema_fast = bars[0].close;
    double ema_slow = bars[0].close;
    double dea = 0;

    bars[0].dif = 0;
    bars[0].dea = 0;
    bars[0].macd_hist = 0;

    for (size_t i = 1; i < bars.size(); ++i) {
        ema_fast = (bars[i].close - ema_fast) * mult_fast + ema_fast;
        ema_slow = (bars[i].close - ema_slow) * mult_slow + ema_slow;
        bars[i].dif = ema_fast - ema_slow;
        
        if (i == 1) dea = bars[i].dif;
        else dea = (bars[i].dif - dea) * mult_sig + dea;
        
        bars[i].dea = dea;
        bars[i].macd_hist = 2.0 * (bars[i].dif - bars[i].dea);
    }
}

// Compute annualized return correctly handling negative totals
double compute_annualized(double total_return_ratio, double years) {
    if (years <= 0.082) return 0; // < 30 days
    if (total_return_ratio <= 0) return -100.0; // Lost everything
    double ann = std::pow(total_return_ratio, 1.0 / years) - 1.0;
    return ann * 100.0;
}

EvalResult run_strategy(const StockData& stock, const std::vector<MonthlyBar>& monthly, const StrategyParams& p, int global_start_idx, int global_end_idx) {
    EvalResult res = {0, 0.0, 0.0, 0.0, 0.0, "", ""};
    if (monthly.size() < 2 || global_end_idx <= global_start_idx) return res;

    res.date_start = stock.daily_bars[global_start_idx].date;
    res.date_end = stock.daily_bars[global_end_idx - 1].date;
    
    double first_open = stock.daily_bars[global_start_idx].open;
    double last_close = stock.daily_bars[global_end_idx - 1].close;
    res.buy_hold_return_pct = (last_close - first_open) / first_open * 100.0;

    bool holding = false;
    double buy_price = 0.0;
    std::string buy_date;
    int buy_month_idx = -1;
    int buy_daily_idx = -1;

    double capital = 1.0;
    int winning_trades = 0;

    for (size_t i = 1; i < monthly.size() - 1; ++i) {
        const auto& prev = monthly[i-1];
        const auto& curr = monthly[i];

        if (!holding) {
            // Check buy signal
            bool crossed_above = (prev.dif <= prev.dea) && (curr.dif > curr.dea);
            bool dif_cond = (curr.dif / curr.close) < p.buy_dif_threshold;
            
            bool market_cond = true;
            if (p.market_ema_period > 0) {
                auto it = g_market_ema[p.market_ema_period].find(curr.last_date);
                if (it != g_market_ema[p.market_ema_period].end()) {
                    // find market close
                    double market_close = 0;
                    for (int mi = g_market_bars.size()-1; mi >= 0; --mi) {
                        if (g_market_bars[mi].date <= curr.last_date) {
                            market_close = g_market_bars[mi].close;
                            break;
                        }
                    }
                    if (market_close <= it->second) {
                        market_cond = false;
                    }
                }
            }

            if (crossed_above && dif_cond && market_cond) {
                // Execute next month open
                int exec_idx = monthly[i+1].first_daily_idx;
                if (exec_idx < global_end_idx) {
                    buy_price = stock.daily_bars[exec_idx].open;
                    buy_date = stock.daily_bars[exec_idx].date;
                    buy_month_idx = i + 1;
                    buy_daily_idx = exec_idx;
                    holding = true;
                }
            }
        } else {
            // Check sell signal
            bool crossed_below = (prev.dif >= prev.dea) && (curr.dif < curr.dea);
            bool dif_cond = (curr.dif / curr.close) > p.sell_dif_threshold;

            bool force_sell_max_hold = false;
            if (p.max_hold_months > 0 && (i - buy_month_idx) >= p.max_hold_months) {
                force_sell_max_hold = true;
            }

            // Check intra-month stop loss
            bool stop_loss_hit = false;
            int stop_loss_exec_idx = -1;
            if (p.max_loss_pct < 0) {
                for (int d = curr.first_daily_idx; d <= curr.last_daily_idx; ++d) {
                    if (d <= buy_daily_idx) continue;
                    double unrealized_pct = (stock.daily_bars[d].close - buy_price) / buy_price * 100.0;
                    if (unrealized_pct < p.max_loss_pct) {
                        stop_loss_hit = true;
                        stop_loss_exec_idx = d + 1;
                        break;
                    }
                }
            }

            if (stop_loss_hit && stop_loss_exec_idx < global_end_idx) {
                double sell_price = stock.daily_bars[stop_loss_exec_idx].open;
                double ret = (sell_price - buy_price) / buy_price;
                capital *= (1.0 + ret);
                if (ret > 0) winning_trades++;
                res.trades++;
                holding = false;
            } else if ((crossed_below && dif_cond) || force_sell_max_hold) {
                int exec_idx = monthly[i+1].first_daily_idx;
                if (exec_idx < global_end_idx) {
                    double sell_price = stock.daily_bars[exec_idx].open;
                    double ret = (sell_price - buy_price) / buy_price;
                    capital *= (1.0 + ret);
                    if (ret > 0) winning_trades++;
                    res.trades++;
                    holding = false;
                }
            }
        }
    }

    // Force close at end
    if (holding) {
        double sell_price = stock.daily_bars[global_end_idx - 1].close;
        double ret = (sell_price - buy_price) / buy_price;
        capital *= (1.0 + ret);
        if (ret > 0) winning_trades++;
        res.trades++;
    }

    res.cumulative_return_pct = (capital - 1.0) * 100.0;
    if (res.trades > 0) {
        res.win_rate = (double)winning_trades / res.trades * 100.0;
    }

    double total_years = 0;
    if (res.trades > 0 && res.date_start.size() >= 10 && res.date_end.size() >= 10) {
        // Date diff logic
        try {
            int y1 = std::stoi(res.date_start.substr(0,4));
            int m1 = std::stoi(res.date_start.substr(5,2));
            int d1 = std::stoi(res.date_start.substr(8,2));
            int y2 = std::stoi(res.date_end.substr(0,4));
            int m2 = std::stoi(res.date_end.substr(5,2));
            int d2 = std::stoi(res.date_end.substr(8,2));
            int days = (y2-y1)*365 + (m2-m1)*30 + (d2-d1);
            total_years = days / 365.25;
            res.annualized_return_pct = compute_annualized(capital, total_years);
        } catch (...) {
            res.annualized_return_pct = 0;
        }
    }

    return res;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    int num_threads = std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = std::stoi(argv[i+1]);
        }
    }
    if (num_threads <= 0) num_threads = 8;

    std::string data_dir = ".";
    load_stock_names(data_dir);
    load_market_data("/ceph/dang_articles/yoj/market_data/sh000001.csv");

    // Load stocks
    std::cout << "Loading stock data...\n";
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".csv" && entry.path().filename().string().length() == 10) {
            std::string code = entry.path().stem().string();
            std::ifstream file(entry.path());
            if (!file.is_open()) continue;

            StockData sd;
            sd.code = code;
            sd.name = g_stock_names.count(code) ? g_stock_names[code] : code;

            std::string line;
            std::getline(file, line); // header
            while (std::getline(file, line)) {
                auto t = split_csv(line);
                if (t.size() < 6) continue;
                DailyBar b;
                b.date = t[0];
                try {
                    b.open = std::stod(t[2]);
                    b.close = std::stod(t[3]);
                    b.high = std::stod(t[4]);
                    b.low = std::stod(t[5]);
                    b.volume = std::stod(t[6]);
                    sd.daily_bars.push_back(b);
                } catch (...) {}
            }
            if (sd.daily_bars.size() < 100) continue; // Skip too short

            std::sort(sd.daily_bars.begin(), sd.daily_bars.end(), [](const DailyBar& a, const DailyBar& b) {
                return a.date < b.date;
            });

            sd.split_idx = sd.daily_bars.size() * 0.4;
            sd.train_monthly = aggregate_monthly(sd.daily_bars, 0, sd.split_idx);
            sd.test_monthly = aggregate_monthly(sd.daily_bars, sd.split_idx, sd.daily_bars.size());

            g_stocks.push_back(sd);
        }
    }
    std::cout << "Loaded " << g_stocks.size() << " stocks.\n";

    // Generate combos
    std::vector<StrategyParams> combos;
    int fasts[] = {8, 10, 12, 14, 16};
    int slows[] = {20, 24, 26, 30, 34};
    int signals[] = {7, 9, 11};
    double buy_difs[] = {-0.03, -0.01, 0.0};
    double sell_difs[] = {0.0, 0.01, 0.03};
    int m_emas[] = {0, 20, 60, 120};
    double max_losses[] = {0, -10, -15, -20, -30};
    int max_holds[] = {0, 6, 9, 12, 18};

    for (int f : fasts) {
        for (int s : slows) {
            if (f >= s) continue;
            for (int sig : signals) {
                for (double bd : buy_difs) {
                    for (double sd : sell_difs) {
                        for (int me : m_emas) {
                            for (double ml : max_losses) {
                                for (int mh : max_holds) {
                                    combos.push_back({f, s, sig, bd, sd, me, ml, mh});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "Total parameter combinations: " << combos.size() << "\n";
    std::cout << "Starting optimization on " << num_threads << " threads...\n";

    struct ComboResult {
        int combo_idx;
        double score;
    };
    std::vector<ComboResult> results(combos.size());
    std::atomic<int> combo_counter{0};
    std::atomic<int> best_idx{-1};
    std::atomic<double> best_score{-9999.0};
    std::mutex mtx;

    auto worker = [&]() {
        while (true) {
            int idx = combo_counter.fetch_add(1);
            if (idx >= combos.size()) break;

            const auto& p = combos[idx];
            std::vector<double> ann_returns;
            ann_returns.reserve(g_stocks.size());

            for (auto& stock : g_stocks) {
                // To support multithreading safely without deep copying monthly bars,
                // we copy the monthly array locally, compute MACD
                std::vector<MonthlyBar> local_monthly = stock.train_monthly;
                compute_macd(local_monthly, p.fast, p.slow, p.signal);
                
                EvalResult res = run_strategy(stock, local_monthly, p, 0, stock.split_idx);
                if (res.trades > 0) {
                    ann_returns.push_back(res.annualized_return_pct);
                }
            }

            double score = -999.0;
            if (ann_returns.size() >= 20) {
                std::sort(ann_returns.rbegin(), ann_returns.rend());
                double sum = 0;
                for (int i = 0; i < 20; ++i) sum += ann_returns[i];
                score = sum / 20.0;
            }

            results[idx] = {idx, score};

            if (score > best_score.load()) {
                std::lock_guard<std::mutex> lock(mtx);
                if (score > best_score.load()) {
                    best_score.store(score);
                    best_idx.store(idx);
                    std::cout << "\rNew Best: " << score << "% [f:" << p.fast << " s:" << p.slow 
                              << " sig:" << p.signal << " bd:" << p.buy_dif_threshold << " sd:" << p.sell_dif_threshold 
                              << " me:" << p.market_ema_period << " ml:" << p.max_loss_pct << " mh:" << p.max_hold_months << "]   " << std::flush;
                }
            }

            if (idx % 1000 == 0) {
                std::cout << "\rProgress: " << idx << " / " << combos.size() << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "\nOptimization completed.\n";

    if (best_idx.load() == -1) {
        std::cerr << "No valid combinations found (not enough trades).\n";
        return 1;
    }

    StrategyParams best_p = combos[best_idx.load()];
    std::cout << "\n=========================================\n";
    std::cout << "BEST PARAMETERS FOUND (Train Top 20 Score: " << best_score.load() << "%)\n";
    std::cout << "Fast: " << best_p.fast << ", Slow: " << best_p.slow << ", Signal: " << best_p.signal << "\n";
    std::cout << "Buy DIF Threshold: " << best_p.buy_dif_threshold << "\n";
    std::cout << "Sell DIF Threshold: " << best_p.sell_dif_threshold << "\n";
    std::cout << "Market EMA Period: " << best_p.market_ema_period << "\n";
    std::cout << "Max Loss Pct: " << best_p.max_loss_pct << "\n";
    std::cout << "Max Hold Months: " << best_p.max_hold_months << "\n";
    std::cout << "=========================================\n\n";

    // Run best on test set
    std::cout << "Evaluating best parameters on test period...\n";
    
    struct TestOutput {
        std::string code;
        std::string name;
        double train_ann;
        double test_ann;
        double total_ret;
        int trades;
        double win_rate;
        double buy_hold_ret;
        std::string date_start;
        std::string date_end;
    };
    std::vector<TestOutput> test_results;

    for (auto& stock : g_stocks) {
        // Train
        std::vector<MonthlyBar> local_train = stock.train_monthly;
        compute_macd(local_train, best_p.fast, best_p.slow, best_p.signal);
        EvalResult train_res = run_strategy(stock, local_train, best_p, 0, stock.split_idx);
        
        // Test
        std::vector<MonthlyBar> local_test = stock.test_monthly;
        compute_macd(local_test, best_p.fast, best_p.slow, best_p.signal);
        EvalResult test_res = run_strategy(stock, local_test, best_p, stock.split_idx, stock.daily_bars.size());

        TestOutput out;
        out.code = stock.code;
        out.name = stock.name;
        out.train_ann = train_res.annualized_return_pct;
        out.test_ann = test_res.annualized_return_pct;
        out.total_ret = test_res.cumulative_return_pct;
        out.trades = test_res.trades;
        out.win_rate = test_res.win_rate;
        out.buy_hold_ret = test_res.buy_hold_return_pct;
        out.date_start = test_res.date_start;
        out.date_end = test_res.date_end;
        test_results.push_back(out);
    }

    std::sort(test_results.begin(), test_results.end(), [](const TestOutput& a, const TestOutput& b) {
        return a.test_ann > b.test_ann;
    });

    std::ofstream out_full("macd_optimized_results.csv");
    out_full << "stock_code,stock_name,train_annualized,test_annualized,total_return_pct,trades,win_rate,buy_hold_return_pct,date_start,date_end\n";
    
    std::ofstream out_top20("macd_optimized_top20.csv");
    out_top20 << "stock_code,stock_name,train_annualized,test_annualized,total_return_pct,trades,win_rate,buy_hold_return_pct,date_start,date_end\n";

    double test_top20_sum = 0;
    int test_top20_count = 0;

    for (size_t i = 0; i < test_results.size(); ++i) {
        const auto& r = test_results[i];
        out_full << r.code << "," << r.name << "," << r.train_ann << "," << r.test_ann << ","
                 << r.total_ret << "," << r.trades << "," << r.win_rate << ","
                 << r.buy_hold_ret << "," << r.date_start << "," << r.date_end << "\n";
                 
        if (i < 20 && r.trades > 0) {
            out_top20 << r.code << "," << r.name << "," << r.train_ann << "," << r.test_ann << ","
                      << r.total_ret << "," << r.trades << "," << r.win_rate << ","
                      << r.buy_hold_ret << "," << r.date_start << "," << r.date_end << "\n";
            test_top20_sum += r.test_ann;
            test_top20_count++;
        }
    }

    std::cout << "Train Top 20 Avg Annualized: " << best_score.load() << "%\n";
    if (test_top20_count > 0) {
        std::cout << "Test Top 20 Avg Annualized: " << (test_top20_sum / test_top20_count) << "%\n";
    } else {
        std::cout << "Test Top 20 Avg Annualized: 0% (Not enough trades)\n";
    }

    std::cout << "Results saved to macd_optimized_results.csv and macd_optimized_top20.csv\n";
    return 0;
}
