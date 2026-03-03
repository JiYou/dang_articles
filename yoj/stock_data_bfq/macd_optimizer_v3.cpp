/*
 * MACD Strategy Optimizer V3 — Iteration 2
 * 
 * New features over V2:
 * 1. Finer MACD grid around sweet spot (fast 5-12, slow 16-30, signal 5-11)
 * 2. Underwater golden cross filter: only buy when DIF < 0
 * 3. Volume surge filter: buy only when monthly volume > N-month avg
 * 4. DIF momentum filter: require DIF[i] > DIF[i-1] at cross
 * 5. Sell mode variants: death cross vs DIF sign flip
 * 
 * Compiles: g++ -O3 -std=c++17 -pthread -o macd_optimizer_v3 macd_optimizer_v3.cpp
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
    double open, close, high, low, volume;
};

struct MonthlyBarFlat {
    double close;
    double volume;  // monthly total volume
    int first_daily_idx;
    int last_daily_idx;
};

struct StockData {
    std::string code;
    std::string name;
    std::vector<DailyBar> daily_bars;
    std::vector<MonthlyBarFlat> train_monthly;
    std::vector<MonthlyBarFlat> test_monthly;
    int split_idx;
};

struct StrategyParams {
    int fast;
    int slow;
    int signal;
    bool underwater_only;     // only buy when DIF < 0
    bool dif_momentum;        // require DIF rising at cross
    int vol_surge_period;     // 0=off, N=buy when vol > N-month avg
    int sell_mode;            // 0=death cross, 1=DIF turns negative (from positive)
};

struct EvalResult {
    int trades = 0;
    double cumulative_return_pct = 0;
    double annualized_return_pct = 0;
    double win_rate = 0;
    double buy_hold_return_pct = 0;
    std::string date_start, date_end;
};

// ============================================================
// Global Data
// ============================================================

std::unordered_map<std::string, std::string> g_stock_names;
std::vector<StockData> g_stocks;

// ============================================================
// Helpers
// ============================================================

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) result.push_back(cell);
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
        if (line.size() >= 3 && line[0] == '\xEF' && line[1] == '\xBB' && line[2] == '\xBF')
            line = line.substr(3);
        if (line.empty()) continue;
        auto tokens = split_csv(line);
        if (tokens.size() >= 2) {
            std::string code = tokens[0];
            std::string name = tokens[1];
            if (!name.empty() && name.back() == '\r') name.pop_back();
            if (g_stock_names.find(code) == g_stock_names.end())
                g_stock_names[code] = name;
        }
    }
}

std::vector<MonthlyBarFlat> aggregate_monthly(const std::vector<DailyBar>& daily, int start, int end) {
    std::vector<MonthlyBarFlat> monthly;
    if (start >= end) return monthly;

    std::string cur_ym = daily[start].date.substr(0, 7);
    double cur_close = daily[start].close;
    double cur_vol = daily[start].volume;
    int cur_first = start, cur_last = start;

    for (int i = start + 1; i < end; ++i) {
        std::string ym = daily[i].date.substr(0, 7);
        if (ym == cur_ym) {
            cur_close = daily[i].close;
            cur_vol += daily[i].volume;
            cur_last = i;
        } else {
            monthly.push_back({cur_close, cur_vol, cur_first, cur_last});
            cur_ym = ym;
            cur_close = daily[i].close;
            cur_vol = daily[i].volume;
            cur_first = i;
            cur_last = i;
        }
    }
    monthly.push_back({cur_close, cur_vol, cur_first, cur_last});
    return monthly;
}

void compute_macd_inline(const std::vector<MonthlyBarFlat>& bars, int fast, int slow, int signal,
                         std::vector<double>& dif, std::vector<double>& dea) {
    size_t n = bars.size();
    dif.resize(n);
    dea.resize(n);
    if (n == 0) return;

    double mf = 2.0 / (fast + 1), ms = 2.0 / (slow + 1), msig = 2.0 / (signal + 1);
    double ef = bars[0].close, es = bars[0].close, d = 0;
    dif[0] = 0; dea[0] = 0;

    for (size_t i = 1; i < n; ++i) {
        ef = (bars[i].close - ef) * mf + ef;
        es = (bars[i].close - es) * ms + es;
        dif[i] = ef - es;
        if (i == 1) d = dif[i];
        else d = (dif[i] - d) * msig + d;
        dea[i] = d;
    }
}

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    return (std::pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

EvalResult run_strategy_v3(const StockData& stock, const std::vector<MonthlyBarFlat>& monthly,
                           const std::vector<double>& dif, const std::vector<double>& dea,
                           const StrategyParams& p, int global_start, int global_end) {
    EvalResult res;
    if (monthly.size() < 3 || global_end <= global_start) return res;

    res.date_start = stock.daily_bars[global_start].date;
    res.date_end = stock.daily_bars[global_end - 1].date;
    double first_open = stock.daily_bars[global_start].open;
    double last_close = stock.daily_bars[global_end - 1].close;
    res.buy_hold_return_pct = (last_close - first_open) / first_open * 100.0;

    bool holding = false;
    double buy_price = 0, capital = 1.0;
    int winning = 0;

    for (size_t i = 1; i < monthly.size() - 1; ++i) {
        double prev_dif = dif[i-1], prev_dea = dea[i-1];
        double curr_dif = dif[i], curr_dea = dea[i];

        if (!holding) {
            // Buy signal: golden cross (DIF crosses above DEA)
            bool golden_cross = (prev_dif <= prev_dea) && (curr_dif > curr_dea);
            if (!golden_cross) continue;

            // Filter: underwater only (DIF < 0 at cross)
            if (p.underwater_only && curr_dif >= 0) continue;

            // Filter: DIF momentum (DIF rising)
            if (p.dif_momentum && i >= 2 && curr_dif <= dif[i-1]) continue;

            // Filter: volume surge
            if (p.vol_surge_period > 0 && (int)i >= p.vol_surge_period) {
                double vol_avg = 0;
                for (int v = (int)i - p.vol_surge_period; v < (int)i; ++v)
                    vol_avg += monthly[v].volume;
                vol_avg /= p.vol_surge_period;
                if (monthly[i].volume < vol_avg * 1.2) continue; // need 20% above average
            }

            // Execute buy on next month open
            int exec_idx = monthly[i+1].first_daily_idx;
            if (exec_idx < global_end) {
                buy_price = stock.daily_bars[exec_idx].open;
                holding = true;
            }
        } else {
            // Sell signal
            bool sell_signal = false;
            
            if (p.sell_mode == 0) {
                // Mode 0: death cross (DIF crosses below DEA)
                sell_signal = (prev_dif >= prev_dea) && (curr_dif < curr_dea);
            } else if (p.sell_mode == 1) {
                // Mode 1: DIF turns negative from positive (stricter — holds through minor pullbacks)
                sell_signal = (prev_dif >= 0) && (curr_dif < 0);
            }

            if (sell_signal) {
                int exec_idx = monthly[i+1].first_daily_idx;
                if (exec_idx < global_end) {
                    double sell_price = stock.daily_bars[exec_idx].open;
                    double ret = (sell_price - buy_price) / buy_price;
                    capital *= (1.0 + ret);
                    if (ret > 0) winning++;
                    res.trades++;
                    holding = false;
                }
            }
        }
    }

    // Force close
    if (holding) {
        double sell_price = stock.daily_bars[global_end - 1].close;
        double ret = (sell_price - buy_price) / buy_price;
        capital *= (1.0 + ret);
        if (ret > 0) winning++;
        res.trades++;
    }

    res.cumulative_return_pct = (capital - 1.0) * 100.0;
    if (res.trades > 0) res.win_rate = (double)winning / res.trades * 100.0;

    if (res.trades > 0 && res.date_start.size() >= 10 && res.date_end.size() >= 10) {
        try {
            int y1 = std::stoi(res.date_start.substr(0,4));
            int m1 = std::stoi(res.date_start.substr(5,2));
            int d1 = std::stoi(res.date_start.substr(8,2));
            int y2 = std::stoi(res.date_end.substr(0,4));
            int m2 = std::stoi(res.date_end.substr(5,2));
            int d2 = std::stoi(res.date_end.substr(8,2));
            int days = (y2-y1)*365 + (m2-m1)*30 + (d2-d1);
            res.annualized_return_pct = compute_annualized(capital, days / 365.25);
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
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = std::stoi(argv[i+1]);
    }
    if (num_threads <= 0) num_threads = 8;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::string data_dir = ".";
    load_stock_names(data_dir);

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
                if (t.size() < 7) continue;
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
            if (sd.daily_bars.size() < 100) continue;

            std::sort(sd.daily_bars.begin(), sd.daily_bars.end(),
                      [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });

            sd.split_idx = sd.daily_bars.size() * 0.4;
            sd.train_monthly = aggregate_monthly(sd.daily_bars, 0, sd.split_idx);
            sd.test_monthly = aggregate_monthly(sd.daily_bars, sd.split_idx, sd.daily_bars.size());
            g_stocks.push_back(std::move(sd));
        }
    }
    std::cout << "Loaded " << g_stocks.size() << " stocks.\n";

    // ============================================================
    // Phase A: Fine MACD grid (no filters)
    // ============================================================
    std::cout << "\n=== Phase A: Fine-grained MACD parameter search ===\n";
    auto t_phA = std::chrono::high_resolution_clock::now();

    std::vector<StrategyParams> phase_a;
    // Finer grid: fast 5-14, slow 16-34, signal 5-11
    for (int f = 5; f <= 14; ++f) {
        for (int s = 16; s <= 34; s += 2) {
            if (f >= s) continue;
            for (int sig = 5; sig <= 11; sig += 1) {
                phase_a.push_back({f, s, sig, false, false, 0, 0});
            }
        }
    }

    std::cout << "Phase A combos: " << phase_a.size() << "\n";

    struct ComboResult {
        StrategyParams params;
        double score;
    };
    std::vector<ComboResult> phase_a_results(phase_a.size());
    std::atomic<int> counter_a{0};

    auto worker_a = [&]() {
        thread_local std::vector<double> tl_dif, tl_dea;
        while (true) {
            int idx = counter_a.fetch_add(1);
            if (idx >= (int)phase_a.size()) break;

            const auto& p = phase_a[idx];
            std::vector<double> ann;
            ann.reserve(g_stocks.size());

            for (const auto& stock : g_stocks) {
                compute_macd_inline(stock.train_monthly, p.fast, p.slow, p.signal, tl_dif, tl_dea);
                EvalResult res = run_strategy_v3(stock, stock.train_monthly, tl_dif, tl_dea, p, 0, stock.split_idx);
                if (res.trades > 0) ann.push_back(res.annualized_return_pct);
            }

            double score = -999;
            if (ann.size() >= 20) {
                std::sort(ann.rbegin(), ann.rend());
                double sum = 0;
                for (int i = 0; i < 20; ++i) sum += ann[i];
                score = sum / 20.0;
            }
            phase_a_results[idx] = {p, score};
        }
    };

    { std::vector<std::thread> threads;
      for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_a);
      for (auto& t : threads) t.join(); }

    std::sort(phase_a_results.begin(), phase_a_results.end(),
              [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

    int top_n = std::min<int>(10, phase_a_results.size());
    std::cout << "\nTop " << top_n << " MACD parameter sets:\n";
    for (int i = 0; i < top_n; ++i) {
        auto& r = phase_a_results[i];
        printf("  %2d. [f:%d s:%d sig:%d] -> Score: %.2f%%\n",
               i+1, r.params.fast, r.params.slow, r.params.signal, r.score);
    }

    auto t_phA_end = std::chrono::high_resolution_clock::now();
    printf("Phase A: %.2fs\n\n", std::chrono::duration<double>(t_phA_end - t_phA).count());

    // ============================================================
    // Phase B: Filter grid for top MACD sets
    // ============================================================
    std::cout << "=== Phase B: Filter optimization for top MACD sets ===\n";
    auto t_phB = std::chrono::high_resolution_clock::now();

    int top_macd = std::min<int>(5, phase_a_results.size());
    std::vector<StrategyParams> phase_b;

    for (int t = 0; t < top_macd; ++t) {
        int f = phase_a_results[t].params.fast;
        int s = phase_a_results[t].params.slow;
        int sig = phase_a_results[t].params.signal;

        // Grid over new filter params
        bool underwater_vals[] = {false, true};
        bool momentum_vals[] = {false, true};
        int vol_surge_vals[] = {0, 3, 6};
        int sell_mode_vals[] = {0, 1};

        for (bool uw : underwater_vals) {
            for (bool mom : momentum_vals) {
                for (int vs : vol_surge_vals) {
                    for (int sm : sell_mode_vals) {
                        phase_b.push_back({f, s, sig, uw, mom, vs, sm});
                    }
                }
            }
        }
    }

    std::cout << "Phase B combos: " << phase_b.size() << "\n";

    std::vector<ComboResult> phase_b_results(phase_b.size());
    std::atomic<int> counter_b{0};
    std::atomic<double> best_score{-9999.0};
    std::atomic<int> best_idx{-1};
    std::mutex mtx;

    auto worker_b = [&]() {
        thread_local std::vector<double> tl_dif, tl_dea;
        while (true) {
            int idx = counter_b.fetch_add(1);
            if (idx >= (int)phase_b.size()) break;

            const auto& p = phase_b[idx];
            std::vector<double> ann;
            ann.reserve(g_stocks.size());

            for (const auto& stock : g_stocks) {
                compute_macd_inline(stock.train_monthly, p.fast, p.slow, p.signal, tl_dif, tl_dea);
                EvalResult res = run_strategy_v3(stock, stock.train_monthly, tl_dif, tl_dea, p, 0, stock.split_idx);
                if (res.trades > 0) ann.push_back(res.annualized_return_pct);
            }

            double score = -999;
            if (ann.size() >= 20) {
                std::sort(ann.rbegin(), ann.rend());
                double sum = 0;
                for (int i = 0; i < 20; ++i) sum += ann[i];
                score = sum / 20.0;
            }

            phase_b_results[idx] = {p, score};

            if (score > best_score.load()) {
                std::lock_guard<std::mutex> lock(mtx);
                if (score > best_score.load()) {
                    best_score.store(score);
                    best_idx.store(idx);
                }
            }
        }
    };

    { std::vector<std::thread> threads;
      for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_b);
      for (auto& t : threads) t.join(); }

    // Sort and print top results
    std::sort(phase_b_results.begin(), phase_b_results.end(),
              [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

    std::cout << "\nTop 15 configurations (MACD + filters):\n";
    printf("  %-4s %-4s %-4s %-6s  %-10s %-10s %-10s %-10s\n",
           "Fast", "Slow", "Sig", "Score", "Underwater", "Momentum", "VolSurge", "SellMode");
    printf("  %-4s %-4s %-4s %-5s  %-10s %-10s %-10s %-10s\n",
           "----", "----", "---", "-----", "----------", "--------", "--------", "--------");
    for (int i = 0; i < std::min<int>(15, phase_b_results.size()); ++i) {
        auto& r = phase_b_results[i];
        printf("  %-4d %-4d %-4d %6.2f  %-10s %-10s %-10d %-10s\n",
               r.params.fast, r.params.slow, r.params.signal, r.score,
               r.params.underwater_only ? "YES" : "no",
               r.params.dif_momentum ? "YES" : "no",
               r.params.vol_surge_period,
               r.params.sell_mode == 0 ? "DeathX" : "DIF<0");
    }

    auto t_phB_end = std::chrono::high_resolution_clock::now();
    printf("\nPhase B: %.2fs\n\n", std::chrono::duration<double>(t_phB_end - t_phB).count());

    // ============================================================
    // Phase C: Evaluate best params on test set
    // ============================================================
    if (phase_b_results.empty() || phase_b_results[0].score < -900) {
        std::cerr << "No valid results found.\n";
        return 1;
    }

    StrategyParams best_p = phase_b_results[0].params;
    double train_score = phase_b_results[0].score;

    std::cout << "=========================================\n";
    printf("BEST PARAMS (Train Top 20: %.2f%%)\n", train_score);
    printf("  Fast: %d, Slow: %d, Signal: %d\n", best_p.fast, best_p.slow, best_p.signal);
    printf("  Underwater: %s, Momentum: %s\n", best_p.underwater_only ? "YES" : "NO", best_p.dif_momentum ? "YES" : "NO");
    printf("  Volume Surge Period: %d, Sell Mode: %s\n", best_p.vol_surge_period, best_p.sell_mode == 0 ? "Death Cross" : "DIF<0");
    std::cout << "=========================================\n\n";

    std::cout << "Evaluating on test period...\n";

    struct TestOutput {
        std::string code, name;
        double train_ann, test_ann, total_ret;
        int trades;
        double win_rate, buy_hold_ret;
        std::string date_start, date_end;
    };
    std::vector<TestOutput> test_results;

    std::vector<double> tl_dif, tl_dea;
    for (const auto& stock : g_stocks) {
        // Train
        compute_macd_inline(stock.train_monthly, best_p.fast, best_p.slow, best_p.signal, tl_dif, tl_dea);
        EvalResult train_res = run_strategy_v3(stock, stock.train_monthly, tl_dif, tl_dea, best_p, 0, stock.split_idx);

        // Test
        compute_macd_inline(stock.test_monthly, best_p.fast, best_p.slow, best_p.signal, tl_dif, tl_dea);
        EvalResult test_res = run_strategy_v3(stock, stock.test_monthly, tl_dif, tl_dea, best_p, stock.split_idx, stock.daily_bars.size());

        test_results.push_back({stock.code, stock.name, train_res.annualized_return_pct,
                                test_res.annualized_return_pct, test_res.cumulative_return_pct,
                                test_res.trades, test_res.win_rate, test_res.buy_hold_return_pct,
                                test_res.date_start, test_res.date_end});
    }

    std::sort(test_results.begin(), test_results.end(),
              [](const TestOutput& a, const TestOutput& b) { return a.test_ann > b.test_ann; });

    // Save CSVs
    std::ofstream out_full("macd_v3_results.csv");
    out_full << "stock_code,stock_name,train_annualized,test_annualized,total_return_pct,trades,win_rate,buy_hold_return_pct,date_start,date_end\n";

    std::ofstream out_top20("macd_v3_top20.csv");
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

    // Also evaluate V2 baseline for comparison
    std::cout << "\n--- Comparison ---\n";
    StrategyParams baseline = {8, 24, 7, false, false, 0, 0};
    std::vector<double> baseline_ann;
    for (const auto& stock : g_stocks) {
        compute_macd_inline(stock.train_monthly, 8, 24, 7, tl_dif, tl_dea);
        EvalResult res = run_strategy_v3(stock, stock.train_monthly, tl_dif, tl_dea, baseline, 0, stock.split_idx);
        if (res.trades > 0) baseline_ann.push_back(res.annualized_return_pct);
    }
    double baseline_score = -999;
    if (baseline_ann.size() >= 20) {
        std::sort(baseline_ann.rbegin(), baseline_ann.rend());
        double sum = 0;
        for (int i = 0; i < 20; ++i) sum += baseline_ann[i];
        baseline_score = sum / 20.0;
    }
    printf("V2 Baseline (8,24,7 no filters): Train %.2f%%\n", baseline_score);
    printf("V3 Best:                         Train %.2f%%\n", train_score);
    printf("Improvement:                     %+.2f%%\n", train_score - baseline_score);
    
    printf("\nTrain Top 20 Avg Annualized: %.2f%%\n", train_score);
    if (test_top20_count > 0) {
        printf("Test Top 20 Avg Annualized:  %.2f%%\n", test_top20_sum / test_top20_count);
    }
    printf("Results saved to macd_v3_results.csv and macd_v3_top20.csv\n");

    auto t_end = std::chrono::high_resolution_clock::now();
    printf("Total Runtime: %.2fs\n", std::chrono::duration<double>(t_end - t_start).count());

    return 0;
}
