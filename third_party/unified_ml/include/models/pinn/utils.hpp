// utils.h — Logging, convergence checking, and reporting utilities.

#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace pinn {

//   Training logger                               

struct TrainingRecord {
    int    epoch;
    double total_loss;
    double pde_loss;
    double bc_loss;
    double data_loss;
};

class TrainingLogger {
public:
    explicit TrainingLogger(int log_every = 100, bool verbose = true)
        : log_every_(log_every), verbose_(verbose) {}

    void log(int epoch, double total, double pde, double bc, double data) {
        history_.push_back({epoch, total, pde, bc, data});
        if (verbose_ && epoch % log_every_ == 0) {
            std::cout << "Epoch " << std::setw(6) << epoch
                      << "  total=" << std::scientific << std::setprecision(4) << total
                      << "  pde="  << pde
                      << "  bc="   << bc;
            if (data > 0.0)
                std::cout << "  data=" << data;
            std::cout << "\n";
        }
    }

    const std::vector<TrainingRecord>& history() const { return history_; }

    bool converged(double tol, int window = 50) const {
        if (static_cast<int>(history_.size()) < window) return false;
        double recent = history_.back().total_loss;
        return recent < tol;
    }

    void print_summary() const {
        if (history_.empty()) return;
        const auto& first = history_.front();
        const auto& last  = history_.back();
        std::cout << "\n=== Training Summary ===\n"
                  << "  Epochs:       " << last.epoch << "\n"
                  << "  Initial loss: " << std::scientific << first.total_loss << "\n"
                  << "  Final loss:   " << last.total_loss << "\n"
                  << "  Reduction:    " << std::fixed << std::setprecision(1)
                  << (first.total_loss / std::max(last.total_loss, 1e-30)) << "x\n";
    }

private:
    int                         log_every_;
    bool                        verbose_;
    std::vector<TrainingRecord> history_;
};

//   Finite-difference gradient check                      

/// Check |analytical - numerical| / max(1, |numerical|) < tol.
inline bool grad_check(double analytical, double numerical,
                        double tol = 1e-4,
                        const std::string& label = "")
{
    double rel = std::abs(analytical - numerical) / std::max(1.0, std::abs(numerical));
    bool ok = rel < tol;
    if (!label.empty()) {
        std::cout << (ok ? "  PASS" : "  FAIL")
                  << "  " << label
                  << "  analytical=" << std::scientific << analytical
                  << "  numerical=" << numerical
                  << "  rel_err=" << rel << "\n";
    }
    return ok;
}

//   Table printer                                

inline void print_table(const std::string& title,
                         const std::vector<std::string>& headers,
                         const std::vector<std::vector<double>>& rows,
                         int width = 14)
{
    std::cout << "\n" << title << "\n";
    for (const auto& h : headers)
        std::cout << std::setw(width) << h;
    std::cout << "\n" << std::string(headers.size() * width, '-') << "\n";
    for (const auto& row : rows) {
        for (double v : row)
            std::cout << std::setw(width) << std::scientific
                      << std::setprecision(5) << v;
        std::cout << "\n";
    }
}

}  // namespace pinn
