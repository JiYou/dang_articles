/*
 * MACD Strategy Optimizer V4 — Iteration 3
 *
 * New features over V3:
 * 1. Signal period range extended down to 3 (V3 only went to 5)
 * 2. Sell mode 2: DIF < -threshold (configurable negative threshold)
 * 3. Sell mode 3: Histogram sign flip (sell when MACD histogram turns negative)
 * 4. Buy mode 1: Histogram-based buy (histogram turns positive from negative)
 * 5. Re-entry cooldown: skip N months after sell before allowing next buy
 * 6. Finer slow period grid (step 1)
 * 7. Three-phase search: A=MACD+sell/buy modes, B=filters, C=cooldown refinement
 *
 * Compiles: g++ -O3 -std=c++17 -pthread -o macd_optimizer_v4 macd_optimizer_v4.cpp
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
    double volume;
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
    int buy_mode;             // 0=golden cross, 1=histogram turns positive
    int sell_mode;            // 0=death cross, 1=DIF<0, 2=DIF<-threshold, 3=histogram turns negative
    double sell_threshold;    // for sell_mode 2: negative threshold (e.g. -0.5 means sell when DIF < -0.5% of close)
    bool underwater_only;     // only buy when DIF < 0
    bool dif_momentum;        // require DIF rising at cross
    int vol_surge_period;     // 0=off, N=buy when vol > N-month avg
    int cooldown;             // months to wait after sell before next buy (0=off)
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
                         std::vector<double>& dif, std::vector<double>& dea, std::vector<double>& hist) {
    size_t n = bars.size();
    dif.resize(n);
    dea.resize(n);
    hist.resize(n);
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

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    return (std::pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

EvalResult run_strategy_v4(const StockData& stock, const std::vector<MonthlyBarFlat>& monthly,
                           const std::vector<double>& dif, const std::vector<double>& dea,
                           const std::vector<double>& hist,
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
    int months_since_sell = 999; // large value so first buy is always allowed

    for (size_t i = 1; i < monthly.size() - 1; ++i) {
        if (!holding) {
            // Check cooldown
            if (p.cooldown > 0 && months_since_sell < p.cooldown) {
                months_since_sell++;
                continue;
            }

            bool buy_signal = false;

            if (p.buy_mode == 0) {
                // Mode 0: Golden cross (DIF crosses above DEA)
                buy_signal = (dif[i-1] <= dea[i-1]) && (dif[i] > dea[i]);
            } else if (p.buy_mode == 1) {
                // Mode 1: Histogram turns positive from negative
                buy_signal = (hist[i-1] <= 0) && (hist[i] > 0);
            }

            if (!buy_signal) {
                months_since_sell++;
                continue;
            }

            // Filter: underwater only (DIF < 0 at cross)
            if (p.underwater_only && dif[i] >= 0) {
                months_since_sell++;
                continue;
            }

            // Filter: DIF momentum (DIF rising)
            if (p.dif_momentum && i >= 2 && dif[i] <= dif[i-1]) {
                months_since_sell++;
                continue;
            }

            // Filter: volume surge
            if (p.vol_surge_period > 0 && (int)i >= p.vol_surge_period) {
                double vol_avg = 0;
                for (int v = (int)i - p.vol_surge_period; v < (int)i; ++v)
                    vol_avg += monthly[v].volume;
                vol_avg /= p.vol_surge_period;
                if (monthly[i].volume < vol_avg * 1.2) {
                    months_since_sell++;
                    continue;
                }
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
                sell_signal = (dif[i-1] >= dea[i-1]) && (dif[i] < dea[i]);
            } else if (p.sell_mode == 1) {
                // Mode 1: DIF turns negative from positive
                sell_signal = (dif[i-1] >= 0) && (dif[i] < 0);
            } else if (p.sell_mode == 2) {
                // Mode 2: DIF drops below negative threshold (relative to close price)
                // threshold is expressed as percentage of close, e.g. 0.5 means DIF < -0.005 * close
                double thresh = -p.sell_threshold * monthly[i].close / 100.0;
                sell_signal = dif[i] < thresh;
            } else if (p.sell_mode == 3) {
                // Mode 3: Histogram turns negative
                sell_signal = (hist[i-1] >= 0) && (hist[i] < 0);
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
                    months_since_sell = 0;
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
// Score a parameter set across all stocks (Top 20 avg annualized)
// ============================================================

double score_params(const StrategyParams& p, bool use_train) {
    thread_local std::vector<double> tl_dif, tl_dea, tl_hist;
    std::vector<double> ann;
    ann.reserve(g_stocks.size());

    for (const auto& stock : g_stocks) {
        const auto& monthly = use_train ? stock.train_monthly : stock.test_monthly;
        int gs = use_train ? 0 : stock.split_idx;
        int ge = use_train ? stock.split_idx : (int)stock.daily_bars.size();

        compute_macd_inline(monthly, p.fast, p.slow, p.signal, tl_dif, tl_dea, tl_hist);
        EvalResult res = run_strategy_v4(stock, monthly, tl_dif, tl_dea, tl_hist, p, gs, ge);
        if (res.trades > 0) ann.push_back(res.annualized_return_pct);
    }

    if (ann.size() < 20) return -999;
    std::sort(ann.rbegin(), ann.rend());
    double sum = 0;
    for (int i = 0; i < 20; ++i) sum += ann[i];
    return sum / 20.0;
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
    // Phase A: MACD params + buy/sell mode grid (no filters, no cooldown)
    // ============================================================
    std::cout << "\n=== Phase A: MACD + Buy/Sell Mode Search ===\n";
    auto t_phA = std::chrono::high_resolution_clock::now();

    std::vector<StrategyParams> phase_a;
    // fast 5-14, slow 16-36 step 1, signal 3-9
    // buy_mode: 0=golden cross, 1=histogram
    // sell_mode: 0=death cross, 1=DIF<0, 3=histogram<0
    // (sell_mode 2 with threshold is deferred to Phase B refinement)
    for (int f = 5; f <= 14; ++f) {
        for (int s = 16; s <= 36; ++s) {
            if (f >= s - 2) continue;  // need reasonable gap
            for (int sig = 3; sig <= 9; sig += 1) {
                for (int bm = 0; bm <= 1; ++bm) {
                    for (int sm : {0, 1, 3}) {
                        phase_a.push_back({f, s, sig, bm, sm, 0.0, false, false, 0, 0});
                    }
                }
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
    int total_a = phase_a.size();

    auto worker_a = [&]() {
        while (true) {
            int idx = counter_a.fetch_add(1);
            if (idx >= total_a) break;
            double s = score_params(phase_a[idx], true);
            phase_a_results[idx] = {phase_a[idx], s};
        }
    };

    { std::vector<std::thread> threads;
      for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_a);
      for (auto& t : threads) t.join(); }

    std::sort(phase_a_results.begin(), phase_a_results.end(),
              [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

    int top_n = std::min<int>(20, phase_a_results.size());
    std::cout << "\nTop " << top_n << " MACD + mode configurations:\n";
    printf("  %-3s %-4s %-4s %-3s %-8s %-10s %-7s\n",
           "#", "Fast", "Slow", "Sig", "BuyMode", "SellMode", "Score");
    printf("  %-3s %-4s %-4s %-3s %-8s %-10s %-7s\n",
           "---", "----", "----", "---", "--------", "----------", "-------");
    const char* buy_mode_str[] = {"GoldenX", "Hist>0"};
    const char* sell_mode_str[] = {"DeathX", "DIF<0", "DIF<-T", "Hist<0"};
    for (int i = 0; i < top_n; ++i) {
        auto& r = phase_a_results[i];
        printf("  %-3d %-4d %-4d %-3d %-8s %-10s %6.2f%%\n",
               i+1, r.params.fast, r.params.slow, r.params.signal,
               buy_mode_str[r.params.buy_mode], sell_mode_str[r.params.sell_mode], r.score);
    }

    auto t_phA_end = std::chrono::high_resolution_clock::now();
    printf("Phase A: %.2fs\n\n", std::chrono::duration<double>(t_phA_end - t_phA).count());

    // ============================================================
    // Phase B: Filter + threshold + cooldown refinement on top configs
    // ============================================================
    std::cout << "=== Phase B: Filter + Cooldown Refinement ===\n";
    auto t_phB = std::chrono::high_resolution_clock::now();

    int top_macd = std::min<int>(8, (int)phase_a_results.size());
    std::vector<StrategyParams> phase_b;

    for (int t = 0; t < top_macd; ++t) {
        auto base = phase_a_results[t].params;

        bool underwater_vals[] = {false, true};
        bool momentum_vals[] = {false, true};
        int vol_surge_vals[] = {0, 3, 6};
        int cooldown_vals[] = {0, 2, 4};

        // Also try sell_mode 2 (DIF < -threshold) variants for configs that use DIF<0 (mode 1)
        std::vector<std::pair<int, double>> sell_variants;
        sell_variants.push_back({base.sell_mode, 0.0});  // original sell mode
        if (base.sell_mode == 1) {
            // Try threshold variants: DIF < -0.3%, -0.5%, -1.0% of close
            sell_variants.push_back({2, 0.3});
            sell_variants.push_back({2, 0.5});
            sell_variants.push_back({2, 1.0});
        }

        for (auto& [sm, thresh] : sell_variants) {
            for (bool uw : underwater_vals) {
                for (bool mom : momentum_vals) {
                    for (int vs : vol_surge_vals) {
                        for (int cd : cooldown_vals) {
                            phase_b.push_back({base.fast, base.slow, base.signal,
                                              base.buy_mode, sm, thresh,
                                              uw, mom, vs, cd});
                        }
                    }
                }
            }
        }
    }

    std::cout << "Phase B combos: " << phase_b.size() << "\n";

    std::vector<ComboResult> phase_b_results(phase_b.size());
    std::atomic<int> counter_b{0};
    int total_b = phase_b.size();

    auto worker_b = [&]() {
        while (true) {
            int idx = counter_b.fetch_add(1);
            if (idx >= total_b) break;
            double s = score_params(phase_b[idx], true);
            phase_b_results[idx] = {phase_b[idx], s};
        }
    };

    { std::vector<std::thread> threads;
      for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_b);
      for (auto& t : threads) t.join(); }

    std::sort(phase_b_results.begin(), phase_b_results.end(),
              [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

    std::cout << "\nTop 20 configurations (MACD + filters + cooldown):\n";
    printf("  %-3s %-4s %-4s %-3s %-8s %-10s %-5s %-5s %-5s %-4s %-4s %7s\n",
           "#", "Fast", "Slow", "Sig", "BuyMode", "SellMode", "UW", "Mom", "VolS", "CD", "Thr", "Score");
    printf("  %-3s %-4s %-4s %-3s %-8s %-10s %-5s %-5s %-5s %-4s %-4s %7s\n",
           "---", "----", "----", "---", "--------", "----------", "-----", "-----", "-----", "----", "----", "-------");
    for (int i = 0; i < std::min<int>(20, (int)phase_b_results.size()); ++i) {
        auto& r = phase_b_results[i];
        char thr_str[16];
        if (r.params.sell_mode == 2) snprintf(thr_str, sizeof(thr_str), "%.1f", r.params.sell_threshold);
        else snprintf(thr_str, sizeof(thr_str), "-");
        printf("  %-3d %-4d %-4d %-3d %-8s %-10s %-5s %-5s %-5d %-4d %-4s %6.2f%%\n",
               i+1, r.params.fast, r.params.slow, r.params.signal,
               buy_mode_str[r.params.buy_mode], sell_mode_str[r.params.sell_mode],
               r.params.underwater_only ? "YES" : "no",
               r.params.dif_momentum ? "YES" : "no",
               r.params.vol_surge_period,
               r.params.cooldown, thr_str, r.score);
    }

    auto t_phB_end = std::chrono::high_resolution_clock::now();
    printf("\nPhase B: %.2fs\n\n", std::chrono::duration<double>(t_phB_end - t_phB).count());

    // ============================================================
    // Phase C: Evaluate top configs on test set
    // ============================================================
    if (phase_b_results.empty() || phase_b_results[0].score < -900) {
        std::cerr << "No valid results found.\n";
        return 1;
    }

    // Evaluate top 5 Phase B configs on test
    std::cout << "=== Phase C: Test Set Evaluation ===\n";
    int eval_count = std::min<int>(5, (int)phase_b_results.size());
    printf("\nTop %d configs evaluated on test period:\n", eval_count);
    printf("  %-3s %-4s %-4s %-3s %-8s %-10s %-5s %-5s %-4s %-4s  %8s %8s\n",
           "#", "Fast", "Slow", "Sig", "BuyMode", "SellMode", "UW", "Mom", "CD", "Thr", "Train", "Test");
    printf("  %-3s %-4s %-4s %-3s %-8s %-10s %-5s %-5s %-4s %-4s  %8s %8s\n",
           "---", "----", "----", "---", "--------", "----------", "-----", "-----", "----", "----", "--------", "--------");

    int best_test_idx = 0;
    double best_test_score = -9999;

    for (int ci = 0; ci < eval_count; ++ci) {
        auto& p = phase_b_results[ci].params;
        double test_score = score_params(p, false);
        double train_score = phase_b_results[ci].score;

        char thr_str[16];
        if (p.sell_mode == 2) snprintf(thr_str, sizeof(thr_str), "%.1f", p.sell_threshold);
        else snprintf(thr_str, sizeof(thr_str), "-");

        printf("  %-3d %-4d %-4d %-3d %-8s %-10s %-5s %-5s %-4d %-4s  %7.2f%% %7.2f%%\n",
               ci+1, p.fast, p.slow, p.signal,
               buy_mode_str[p.buy_mode], sell_mode_str[p.sell_mode],
               p.underwater_only ? "YES" : "no",
               p.dif_momentum ? "YES" : "no",
               p.cooldown, thr_str,
               train_score, test_score);

        if (test_score > best_test_score) {
            best_test_score = test_score;
            best_test_idx = ci;
        }
    }

    // Use the config with best test score
    StrategyParams best_p = phase_b_results[best_test_idx].params;
    double train_score = phase_b_results[best_test_idx].score;

    std::cout << "\n=========================================\n";
    printf("BEST CONFIG (by test performance):\n");
    printf("  Train Top 20: %.2f%%\n", train_score);
    printf("  Test  Top 20: %.2f%%\n", best_test_score);
    printf("  Fast: %d, Slow: %d, Signal: %d\n", best_p.fast, best_p.slow, best_p.signal);
    printf("  Buy Mode: %s, Sell Mode: %s\n", buy_mode_str[best_p.buy_mode], sell_mode_str[best_p.sell_mode]);
    if (best_p.sell_mode == 2) printf("  Sell Threshold: %.1f%%\n", best_p.sell_threshold);
    printf("  Underwater: %s, Momentum: %s\n", best_p.underwater_only ? "YES" : "NO", best_p.dif_momentum ? "YES" : "NO");
    printf("  Vol Surge: %d, Cooldown: %d\n", best_p.vol_surge_period, best_p.cooldown);
    std::cout << "=========================================\n\n";

    // Generate full results with best config
    std::cout << "Generating full results...\n";

    struct TestOutput {
        std::string code, name;
        double train_ann, test_ann, total_ret;
        int trades;
        double win_rate, buy_hold_ret;
        std::string date_start, date_end;
    };
    std::vector<TestOutput> test_results;

    std::vector<double> tl_dif, tl_dea, tl_hist;
    for (const auto& stock : g_stocks) {
        // Train
        compute_macd_inline(stock.train_monthly, best_p.fast, best_p.slow, best_p.signal, tl_dif, tl_dea, tl_hist);
        EvalResult train_res = run_strategy_v4(stock, stock.train_monthly, tl_dif, tl_dea, tl_hist, best_p, 0, stock.split_idx);

        // Test
        compute_macd_inline(stock.test_monthly, best_p.fast, best_p.slow, best_p.signal, tl_dif, tl_dea, tl_hist);
        EvalResult test_res = run_strategy_v4(stock, stock.test_monthly, tl_dif, tl_dea, tl_hist, best_p, stock.split_idx, stock.daily_bars.size());

        test_results.push_back({stock.code, stock.name, train_res.annualized_return_pct,
                                test_res.annualized_return_pct, test_res.cumulative_return_pct,
                                test_res.trades, test_res.win_rate, test_res.buy_hold_return_pct,
                                test_res.date_start, test_res.date_end});
    }

    std::sort(test_results.begin(), test_results.end(),
              [](const TestOutput& a, const TestOutput& b) { return a.test_ann > b.test_ann; });

    // Save CSVs
    std::ofstream out_full("macd_v4_results.csv");
    out_full << "stock_code,stock_name,train_annualized,test_annualized,total_return_pct,trades,win_rate,buy_hold_return_pct,date_start,date_end\n";

    std::ofstream out_top20("macd_v4_top20.csv");
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

    // Print top 20 test stocks
    std::cout << "\nTop 20 stocks by test annualized return:\n";
    printf("  %-3s %-8s %-20s %8s %8s %6s %6s\n", "#", "Code", "Name", "Train%", "Test%", "Trades", "WinR%");
    printf("  %-3s %-8s %-20s %8s %8s %6s %6s\n", "---", "--------", "--------------------", "--------", "--------", "------", "------");
    for (int i = 0; i < std::min<int>(20, (int)test_results.size()); ++i) {
        auto& r = test_results[i];
        printf("  %-3d %-8s %-20s %7.2f%% %7.2f%% %6d %5.1f%%\n",
               i+1, r.code.c_str(), r.name.substr(0, 20).c_str(),
               r.train_ann, r.test_ann, r.trades, r.win_rate);
    }

    // V3 baseline comparison
    std::cout << "\n--- Version Comparison ---\n";
    StrategyParams v3_best = {8, 22, 5, 0, 1, 0.0, false, false, 0, 0};  // V3 best: golden cross buy, DIF<0 sell
    double v3_train = score_params(v3_best, true);
    double v3_test = score_params(v3_best, false);

    StrategyParams v2_best = {8, 24, 7, 0, 0, 0.0, false, false, 0, 0};  // V2 best: death cross sell
    double v2_train = score_params(v2_best, true);
    double v2_test = score_params(v2_best, false);

    printf("  %-10s  %8s  %8s  %s\n", "Version", "Train", "Test", "Config");
    printf("  %-10s  %8s  %8s  %s\n", "----------", "--------", "--------", "------");
    printf("  %-10s  %7.2f%%  %7.2f%%  8,24,7 GoldenX/DeathX\n", "V2", v2_train, v2_test);
    printf("  %-10s  %7.2f%%  %7.2f%%  8,22,5 GoldenX/DIF<0\n", "V3", v3_train, v3_test);
    printf("  %-10s  %7.2f%%  %7.2f%%  %d,%d,%d %s/%s",
           "V4", train_score, best_test_score,
           best_p.fast, best_p.slow, best_p.signal,
           buy_mode_str[best_p.buy_mode], sell_mode_str[best_p.sell_mode]);
    if (best_p.underwater_only) printf(" UW");
    if (best_p.dif_momentum) printf(" Mom");
    if (best_p.cooldown > 0) printf(" CD%d", best_p.cooldown);
    if (best_p.sell_mode == 2) printf(" Thr%.1f", best_p.sell_threshold);
    printf("\n");

    printf("\nTest Top 20 Avg Annualized: %.2f%%\n", test_top20_count > 0 ? test_top20_sum / test_top20_count : 0.0);
    printf("Results saved to macd_v4_results.csv and macd_v4_top20.csv\n");

    auto t_end = std::chrono::high_resolution_clock::now();
    printf("Total Runtime: %.2fs\n", std::chrono::duration<double>(t_end - t_start).count());

    return 0;
}
