// Sparse symmetric linear systems: assembly from blocks, a fill-reducing
// ordering (minimum degree on the graph of variables), and an LDL^T
// factorization (elimination tree, symbolic column counts, up-looking
// numeric factorization, as in T. Davis, "Direct Methods for Sparse Linear
// Systems", 2006). Used by every least-squares problem in Rover.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace rover {

/// Minimum degree ordering of an undirected graph (adjacency lists):
/// eliminate the node of smallest current degree, join its neighbors into a
/// clique, repeat. Exact (no approximation), fine for graphs of variables.
inline std::vector<int> minimum_degree(const std::vector<std::vector<int>>& adj) {
    int n = static_cast<int>(adj.size());
    std::vector<std::set<int>> g(n);
    for (int i = 0; i < n; ++i)
        for (int j : adj[i])
            if (j != i) {
                g[i].insert(j);
                g[j].insert(i);
            }
    std::vector<char> done(n, 0);
    std::set<std::pair<int, int>> pq;  // (degree, node)
    for (int i = 0; i < n; ++i) pq.insert({static_cast<int>(g[i].size()), i});
    std::vector<int> order;
    order.reserve(n);
    while (!pq.empty()) {
        auto [d, v] = *pq.begin();
        pq.erase(pq.begin());
        done[v] = 1;
        order.push_back(v);
        std::vector<int> nb(g[v].begin(), g[v].end());
        for (int u : nb) {
            pq.erase({static_cast<int>(g[u].size()), u});
            g[u].erase(v);
        }
        for (size_t a = 0; a < nb.size(); ++a)
            for (size_t b = a + 1; b < nb.size(); ++b) {
                g[nb[a]].insert(nb[b]);
                g[nb[b]].insert(nb[a]);
            }
        for (int u : nb) pq.insert({static_cast<int>(g[u].size()), u});
        g[v].clear();
    }
    return order;
}

/// A symmetric matrix in compressed sparse column form, upper triangle
/// (row <= column), assembled from (row, col, value) triplets.
struct SparseSym {
    int n = 0;
    std::vector<int> colptr, rowidx;
    std::vector<double> val;

    static SparseSym from_triplets(int n, std::vector<std::tuple<int, int, double>>& t) {
        for (auto& [r, c, v] : t)
            if (r > c) std::swap(r, c);
        std::sort(t.begin(), t.end(), [](const auto& a, const auto& b) {
            return std::get<1>(a) != std::get<1>(b) ? std::get<1>(a) < std::get<1>(b) : std::get<0>(a) < std::get<0>(b);
        });
        SparseSym m;
        m.n = n;
        m.colptr.assign(n + 1, 0);
        for (size_t i = 0; i < t.size(); ++i) {
            auto [r, c, v] = t[i];
            if (!m.rowidx.empty() && i > 0 && std::get<0>(t[i - 1]) == r && std::get<1>(t[i - 1]) == c) {
                m.val.back() += v;
                continue;
            }
            m.rowidx.push_back(r);
            m.val.push_back(v);
            m.colptr[c + 1]++;
        }
        for (int c = 0; c < n; ++c) m.colptr[c + 1] += m.colptr[c];
        return m;
    }

    /// y = A x (A symmetric, stored as the upper triangle)
    std::vector<double> multiply(const std::vector<double>& x) const {
        std::vector<double> y(n, 0.0);
        for (int c = 0; c < n; ++c)
            for (int k = colptr[c]; k < colptr[c + 1]; ++k) {
                int r = rowidx[k];
                y[r] += val[k] * x[c];
                if (r != c) y[c] += val[k] * x[r];
            }
        return y;
    }
};

/// LDL^T factorization of a permuted symmetric matrix P A P^T.
class LDLT {
public:
    /// perm[k] = original index of the k-th unknown in the elimination order.
    bool factor(const SparseSym& A, const std::vector<int>& perm) {
        analyze(A, perm);
        return refactor(A);
    }

    /// The symbolic part: the permuted pattern and the elimination tree. A
    /// later refactor() needs a matrix with exactly the same pattern.
    void analyze(const SparseSym& A, const std::vector<int>& perm) {
        n_ = A.n;
        perm_ = perm;
        pinv_.assign(n_, 0);
        for (int k = 0; k < n_; ++k) pinv_[perm_[k]] = k;
        // permuted upper triangle C = P A P^T; remember where each entry of A goes
        std::vector<std::tuple<int, int, double>> t;
        t.reserve(A.val.size());
        for (int c = 0; c < n_; ++c)
            for (int k = A.colptr[c]; k < A.colptr[c + 1]; ++k) t.emplace_back(pinv_[A.rowidx[k]], pinv_[c], static_cast<double>(k));
        C_ = SparseSym::from_triplets(n_, t);
        map_.assign(A.val.size(), 0);
        for (size_t q = 0; q < C_.val.size(); ++q) map_[static_cast<size_t>(C_.val[q])] = static_cast<int>(q);
        symbolic();
    }

    /// The numeric factorization of a matrix with the analyzed pattern.
    bool refactor(const SparseSym& A) {
        for (size_t k = 0; k < A.val.size(); ++k) C_.val[map_[k]] = A.val[k];
        return numeric();
    }

    std::vector<double> solve(const std::vector<double>& b) const {
        std::vector<double> x(n_);
        for (int k = 0; k < n_; ++k) x[k] = b[perm_[k]];
        // L y = x (unit lower triangular, column storage)
        for (int j = 0; j < n_; ++j)
            for (int p = Lp_[j]; p < Lp_[j + 1]; ++p) x[Li_[p]] -= Lx_[p] * x[j];
        for (int j = 0; j < n_; ++j) x[j] /= D_[j];
        for (int j = n_ - 1; j >= 0; --j)
            for (int p = Lp_[j]; p < Lp_[j + 1]; ++p) x[j] -= Lx_[p] * x[Li_[p]];
        std::vector<double> out(n_);
        for (int k = 0; k < n_; ++k) out[perm_[k]] = x[k];
        return out;
    }

    size_t nonzeros() const { return Li_.size(); }
    double min_pivot() const {
        double m = 1e300;
        for (double d : D_) m = std::min(m, d);
        return m;
    }

private:
    int n_ = 0;
    std::vector<int> perm_, pinv_, parent_, Lp_, Li_, map_;
    std::vector<double> Lx_, D_;
    SparseSym C_;

    // elimination tree and the number of nonzeros of each column of L
    void symbolic() {
        parent_.assign(n_, -1);
        std::vector<int> flag(n_, -1), lnz(n_, 0);
        for (int k = 0; k < n_; ++k) {
            flag[k] = k;
            for (int p = C_.colptr[k]; p < C_.colptr[k + 1]; ++p) {
                int i = C_.rowidx[p];  // i <= k
                for (; i < k && flag[i] != k; i = parent_[i]) {
                    if (parent_[i] == -1) parent_[i] = k;
                    lnz[i]++;
                    flag[i] = k;
                }
            }
        }
        Lp_.assign(n_ + 1, 0);
        for (int k = 0; k < n_; ++k) Lp_[k + 1] = Lp_[k] + lnz[k];
        Li_.assign(Lp_[n_], 0);
        Lx_.assign(Lp_[n_], 0.0);
        D_.assign(n_, 0.0);
    }

    // row k of L by a sparse triangular solve along the elimination tree
    bool numeric() {
        std::vector<double> y(n_, 0.0);
        std::vector<int> pattern(n_), flag(n_, -1), lnz(n_, 0);
        for (int k = 0; k < n_; ++k) {
            y[k] = 0;
            int top = n_;
            flag[k] = k;
            for (int p = C_.colptr[k]; p < C_.colptr[k + 1]; ++p) {
                int i = C_.rowidx[p];
                y[i] += C_.val[p];
                int len = 0;
                for (; flag[i] != k; i = parent_[i]) {
                    pattern[len++] = i;
                    flag[i] = k;
                }
                while (len > 0) pattern[--top] = pattern[--len];
            }
            D_[k] = y[k];
            y[k] = 0;
            for (; top < n_; ++top) {
                int i = pattern[top];
                double yi = y[i];
                y[i] = 0;
                int p2 = Lp_[i] + lnz[i];
                for (int p = Lp_[i]; p < p2; ++p) y[Li_[p]] -= Lx_[p] * yi;
                double l = yi / D_[i];
                D_[k] -= l * yi;
                Li_[p2] = k;
                Lx_[p2] = l;
                lnz[i]++;
            }
            if (!(D_[k] > 0) || !std::isfinite(D_[k])) return false;  // not positive definite
        }
        return true;
    }
};

/// Expand an ordering of block variables into an ordering of scalar unknowns.
inline std::vector<int> expand_order(const std::vector<int>& block_order, const std::vector<int>& offset, const std::vector<int>& dim) {
    std::vector<int> perm;
    for (int b : block_order)
        for (int k = 0; k < dim[b]; ++k) perm.push_back(offset[b] + k);
    return perm;
}

}  // namespace rover
