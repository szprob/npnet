#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
const double kEPS = 1e-6;

// Longest commen string.
int LCS(const std::string &s, const std::string &t);

// The ratio between longest commen string and length of `s`.
double LCSRatio(const std::string &s, const std::string &t);

// Single thread form of function `GetCooccurance`.
// You should never call this function manually.
std::map<std::pair<int64_t, int64_t>, int64_t> GetCooccuranceSingle(
    std::unordered_map<std::string, int64_t> &word2int,
    std::vector<std::vector<std::string>> &corpus, const int window_size,
    const int64_t start, const int64_t count);

// Returns a cooccurance map of any pair of words from `corpus`
// as long as their interval is not greater than `window_size`.
// `word2int` helps map corpus string to index.`num_worlers` controls
// the number of threads.
std::map<std::pair<int64_t, int64_t>, int64_t> GetCooccurance(
    std::unordered_map<std::string, int64_t> &word2int,
    std::vector<std::vector<std::string>> &corpus, const int window_size,
    const int num_workers);

int LCS(const std::string &s, const std::string &t) {
  const int sl = s.length();
  const int tl = t.length();
  int dp[sl + 1][tl + 1];
  memset(dp, 0, sizeof(dp));
  for (int i = 1; i <= sl; ++i) {
    for (int j = 1; j <= tl; ++j) {
      if (s[i] == t[j]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else if (dp[i - 1][j] >= dp[i][j - 1]) {
        dp[i][j] = dp[i - 1][j];
      } else {
        dp[i][j] = dp[i][j - 1];
      }
    }
  }
  return dp[sl][tl];
}

double LCSRatio(const std::string &s, const std::string &t) {
  const int lcs_value = LCS(s, t);
  const int sl = s.length() + 1;
  const double ratio = 1.0 * lcs_value / sl;
  return ratio;
}

std::map<std::pair<int64_t, int64_t>, int64_t> GetCooccuranceSingle(
    std::unordered_map<std::string, int64_t> &word2int,
    std::vector<std::vector<std::string>> &corpus, const int window_size,
    const int64_t start, const int64_t count) {
  // map corpus by word2int
  std::vector<std::vector<int64_t>> corpus_int;
  corpus_int.reserve(count);
  for (int64_t i = 0; i < count; i++) {
    std::vector<int64_t> text_int;
    const int64_t text_len = corpus[start + i].size();
    text_int.reserve(text_len);
    text_int.push_back(word2int["<S>"]);
    for (auto &j : corpus[start + i]) {
      if (word2int.find(j) != word2int.end()) {
        text_int.push_back(word2int[j]);
      } else {
        text_int.push_back(0);
      }
    }
    text_int.push_back(word2int["<E>"]);
    corpus_int.push_back(text_int);
  }
  // count cooccurance by corpus
  std::map<std::pair<int64_t, int64_t>, int64_t> cooccurance;
  for (const auto &sentence : corpus_int) {
    const int sentence_len = sentence.size();
    for (int j = 0; j < sentence_len; j++) {
      const int64_t center = sentence[j];
      for (int k = j + 1; k < j + window_size && k < sentence_len; k++) {
        const int64_t v = sentence[k];
        const auto map_key = (v < center ? std::make_pair(v, center)
                                         : std::make_pair(center, v));
        cooccurance[map_key]++;
      }
    }
  }
  return cooccurance;
}

std::map<std::pair<int64_t, int64_t>, int64_t> GetCooccurance(
    std::unordered_map<std::string, int64_t> &word2int,
    std::vector<std::vector<std::string>> &corpus, const int window_size,
    const int num_workers) {
  const int64_t n = corpus.size();
  std::vector<std::future<std::map<std::pair<int64_t, int64_t>, int64_t>>> ans;
  int64_t batch_each = n / num_workers;
  for (int64_t i = 0; i < num_workers; i++) {
    if (i == num_workers - 1) {
      batch_each = n - batch_each * (num_workers - 1);
    }
    if (batch_each == 0) {
      continue;
    }
    ans.push_back(async(std::launch::async, GetCooccuranceSingle, ref(word2int),
                        ref(corpus), window_size, i * (n / num_workers),
                        batch_each));
  }
  std::map<std::pair<int64_t, int64_t>, int64_t> cooccurance;
  for (auto &ans_slice : ans) {
    std::map<std::pair<int64_t, int64_t>, int64_t> cooccurance_slice =
        ans_slice.get();
    for (const auto &cooc_it : cooccurance_slice) {
      cooccurance[cooc_it.first] = cooccurance[cooc_it.first] + cooc_it.second;
    }
  }
  return cooccurance;
}

PYBIND11_MODULE(counting_utils, m) {
  m.def("lcs", &LCS, "LCS", py::arg("s"), py::arg("t"));
  m.def("lcs_ratio", &LCSRatio, "LCSRatio", py::arg("s"), py::arg("t"));
  m.def("get_cooccurance", &GetCooccurance, "GetCooccurance",
        py::arg("word2int"), py::arg("corpus"), py::arg("window_size"),
        py::arg("num_workers"));
}
