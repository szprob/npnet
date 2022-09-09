#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Damerau–Levenshtein
int LevenshteinDistance(const std::string &s, const std::string &t);

// BKTree for spelling correction.
// Example:
//    std::vector<std::string> words = {"i","love","you"};
//    BKTree *t = new BKTree(words);
//    auto sim_words = t->Query("like");
class BKTree {
 private:
  std::string word_;
  std::unordered_map<int, BKTree *> children_;

  // Single form of function `BatchQuery`.
  std::vector<std::vector<std::string>> QueryWordsSlice(
      std::vector<std::string>::const_iterator first,
      std::vector<std::string>::const_iterator last, const int distance);

 public:
  // Create a void bk-tree.
  // By calling function `Add` can make it work.
  BKTree();

  // Create a bk-tree from wrods directly.
  BKTree(const std::vector<std::string> &words);

  ~BKTree();

  // Insert a word to BKTree.
  void Add(const std::string &word);

  // Query sim words of `word` from BKTree.
  // Larger distance means larger coverage.
  std::vector<std::string> Query(const std::string &word, const int distance);

  // Batch form of function `Query`.
  // You can control threads num by `num_workers`.
  std::vector<std::vector<std::string>> BatchQuery(
      const std::vector<std::string> &words, const int distance,
      const int num_workers);
};

int LevenshteinDistance(const std::string &s, const std::string &t) {
  int dp[s.length() + 1][t.length() + 1];
  const int sl = s.length();
  const int tl = t.length();

  for (int i = 0; i <= sl; i++) {
    dp[i][0] = i;
  }
  for (int j = 1; j <= tl; j++) {
    dp[0][j] = j;
  }
  for (int j = 1; j <= tl; j++) {
    for (int i = 1; i <= sl; i++) {
      const int cost = (s[i-1] == t[j-1] ? 0 : 1);
      dp[i][j] = std::min(
          {dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + cost});

      if ((i > 1) && (j > 1) && (s[i-1] == t[j - 2]) && (s[i - 2] == t[j - 1])) {
        dp[i][j] = std::min({dp[i][j], dp[i - 2][j - 2] + 1});
      }
    }
  }

  return dp[sl][tl];
}

BKTree::BKTree() {}
BKTree::BKTree(const std::vector<std::string> &words) {
  for (const auto &word : words) {
    Add(word);
  }
}

void BKTree::Add(const std::string &word) {
  if (word_.empty()) {
    word_ = word;
  } else {
    const int distance = LevenshteinDistance(word, word_);
    if (distance > 0) {
      auto iter = children_.find(distance);
      if (iter != children_.end()) {
        iter->second->Add(word);
      } else {
        BKTree *new_node = new BKTree();
        new_node->Add(word);
        children_[distance] = new_node;
      }
    }
  }
}

BKTree::~BKTree(void) {
  for (auto &node : children_) {
    delete node.second;
    node.second = NULL;
  }
}

std::vector<std::string> BKTree::Query(const std::string &word,
                                       const int distance) {
  int this_dis = LevenshteinDistance(word_, word);
  std::vector<std::string> sim_words;

  if (this_dis <= distance) {
    sim_words.push_back(word_);
  }
  for (int x = std::max(1, this_dis - distance); x <= this_dis + distance;
       x++) {
    auto iter = children_.find(x);
    if (iter != children_.end()) {
      auto sim_words_slice = iter->second->Query(word, distance);
      sim_words.insert(sim_words.end(), sim_words_slice.begin(),
                       sim_words_slice.end());
    }
  }
  return sim_words;
}

std::vector<std::vector<std::string>> BKTree::QueryWordsSlice(
    std::vector<std::string>::const_iterator first,
    std::vector<std::string>::const_iterator last, const int distance) {
  std::vector<std::vector<std::string>> res;
  res.reserve(std::distance(first, last));
  for (auto it = first; it != last; ++it) {
    res.push_back(Query(*it, distance));
  }
  return res;
}

std::vector<std::vector<std::string>> BKTree::BatchQuery(
    const std::vector<std::string> &words, const int distance,
    const int num_workers) {
  const int64_t n = words.size();

  std::vector<std::future<std::vector<std::vector<std::string>>>> ans;
  int64_t batch_each = n / num_workers;
  for (int64_t i = 0; i < num_workers; i++) {
    if (i == num_workers - 1) {
      batch_each = n - batch_each * (num_workers - 1);
    }
    if (batch_each == 0) {
      continue;
    }
    auto start = words.begin() + i * (n / num_workers);
    auto end = start + batch_each;
    ans.push_back(async(std::launch::async, &BKTree::QueryWordsSlice, this,
                        start, end, distance));
  }

  std::vector<std::vector<std::string>> sim_words;
  for (auto &ans_slice : ans) {
    auto sim_words_slice = ans_slice.get();
    sim_words.insert(sim_words.end(), sim_words_slice.begin(),
                     sim_words_slice.end());
  }
  return sim_words;
}

PYBIND11_MODULE(bktree, m) {
  py::class_<BKTree>(m, "BKTree")
      .def(py::init<const std::vector<std::string>>())
      .def("add", &BKTree::Add, py::arg("word"))
      .def("query", &BKTree::Query, py::arg("word"), py::arg("distance"))
      .def("batch_query", &BKTree::BatchQuery, py::arg("words"),
           py::arg("distance"), py::arg("num_workers"));
  m.def("edit_distance", &LevenshteinDistance, "LevenshteinDistance",
        py::arg("s"), py::arg("t"));
}
