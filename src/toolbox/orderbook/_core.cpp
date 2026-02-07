// src/toolbox/orderbook/_core.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Entry {
    py::dict key;                         // stored by reference (refcount held)
    std::vector<std::uint64_t> idx;       // sorted indices
    std::vector<double> val;              // corresponding values (same length)
    std::unique_ptr<Entry> next;

    Entry(py::dict k,
          std::vector<std::uint64_t> i,
          std::vector<double> v)
        : key(std::move(k)), idx(std::move(i)), val(std::move(v)), next(nullptr) {}
};

py::object get_graphblas_vector_class() {
    // graphblas.Vector
    static py::object VectorCls;
    if (!VectorCls || VectorCls.is_none()) {
        py::module_ graphblas = py::module_::import("graphblas");
        VectorCls = graphblas.attr("Vector");
    }
    return VectorCls;
}

std::int64_t vector_size(const py::object& v) {
    return v.attr("size").cast<std::int64_t>();
}

std::pair<std::vector<std::uint64_t>, std::vector<double>> vector_to_coo_f64(const py::object& v) {
    // Expect: idx, vals = v.to_coo()
    // idx: numpy array of integers (commonly uint64)
    // vals: numpy array of float64 (or castable to float64)
    py::tuple t = v.attr("to_coo")();
    if (t.size() != 2) {
        throw py::type_error("Vector.to_coo() must return (indices, values).");
    }

    py::array idx_arr = py::reinterpret_borrow<py::array>(t[0]);
    py::array val_arr = py::reinterpret_borrow<py::array>(t[1]);

    // Force contiguous copies in known dtypes
    py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> idx_u64(idx_arr);
    py::array_t<double, py::array::c_style | py::array::forcecast> val_f64(val_arr);

    if (idx_u64.ndim() != 1 || val_f64.ndim() != 1) {
        throw py::type_error("Vector.to_coo() must return 1D arrays.");
    }
    if (idx_u64.shape(0) != val_f64.shape(0)) {
        throw py::type_error("Vector.to_coo() returned arrays of different length.");
    }

    const auto n = static_cast<std::size_t>(idx_u64.shape(0));
    std::vector<std::uint64_t> idx(n);
    std::vector<double> val(n);

    auto idx_r = idx_u64.unchecked<1>();
    auto val_r = val_f64.unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        idx[i] = idx_r(i);
        val[i] = val_r(i);
    }

    // Sort by index (to be safe)
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return idx[a] < idx[b];
    });

    std::vector<std::uint64_t> idx2;
    std::vector<double> val2;
    idx2.reserve(n);
    val2.reserve(n);

    // Also drop non-positive entries (greedy only cares about positive supply)
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t i = order[k];
        const double x = val[i];
        if (x > 0.0) {
            idx2.push_back(idx[i]);
            val2.push_back(x);
        }
    }

    return {std::move(idx2), std::move(val2)};
}

py::object vector_from_coo_f64(std::int64_t size,
                               const std::vector<std::uint64_t>& idx,
                               const std::vector<double>& val) {
    if (idx.size() != val.size()) {
        throw std::logic_error("Internal error: idx/val size mismatch.");
    }

    py::object VectorCls = get_graphblas_vector_class();

    // Build numpy arrays to pass to Vector.from_coo
    py::array_t<std::uint64_t> idx_arr(static_cast<py::ssize_t>(idx.size()));
    py::array_t<double> val_arr(static_cast<py::ssize_t>(val.size()));

    {
        auto idx_w = idx_arr.mutable_unchecked<1>();
        auto val_w = val_arr.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < idx_arr.shape(0); ++i) {
            idx_w(i) = idx[static_cast<std::size_t>(i)];
            val_w(i) = val[static_cast<std::size_t>(i)];
        }
    }

    // Vector.from_coo(indices, values, size=..., dtype="FP64")
    // We explicitly request float64 / FP64.
    return VectorCls.attr("from_coo")(idx_arr, val_arr,
                                     py::arg("size") = size,
                                     py::arg("dtype") = "FP64");
}

} // namespace

class Orderbook {
public:
    Orderbook() = default;

    std::int64_t size() const { return size_; }
    std::int64_t length() const { return length_.value_or(0); }

    void append(py::dict key, py::object vec) {
        const std::int64_t n = vector_size(vec);
        if (!length_.has_value()) {
            length_ = n;
        } else if (n != *length_) {
            throw py::type_error("Length mismatch.");
        }

        // Extract sparse COO once; do NOT mutate vec.
        auto [idx, val] = vector_to_coo_f64(vec);

        auto node = std::make_unique<Entry>(std::move(key), std::move(idx), std::move(val));
        Entry* node_raw = node.get();

        if (!head_) {
            head_ = std::move(node);
            tail_ = node_raw;
        } else {
            tail_->next = std::move(node);
            tail_ = node_raw;
        }
        ++size_;
    }

    void remove(std::int64_t index) {
        if (index < 0 || index >= size_) {
            throw py::index_error("Index out of range");
        }

        Entry* prev = nullptr;
        Entry* curr = head_.get();
        for (std::int64_t i = 0; i < index; ++i) {
            prev = curr;
            curr = curr->next.get();
        }

        if (!prev) {
            head_ = std::move(head_->next);
            if (!head_) {
                tail_ = nullptr;
            }
        } else {
            std::unique_ptr<Entry> doomed = std::move(prev->next);
            prev->next = std::move(doomed->next);
            if (!prev->next) {
                tail_ = prev;
            }
        }

        --size_;
        if (size_ == 0) {
            length_.reset();
        }
    }

    py::tuple get_at_index(std::int64_t index) const {
        if (index < 0 || index >= size_) {
            throw py::index_error("Index out of range");
        }

        Entry* curr = head_.get();
        for (std::int64_t i = 0; i < index; ++i) {
            curr = curr->next.get();
        }

        const std::int64_t n = length_.value_or(0);
        py::object vec = vector_from_coo_f64(n, curr->idx, curr->val);
        return py::make_tuple(curr->key, vec);
    }

    py::list allocate_greedy(py::object values) {
        if (!length_.has_value()) {
            return py::list();
        }
        const std::int64_t n = vector_size(values);
        if (n != *length_) {
            throw py::type_error("Length mismatch.");
        }

        // Extract values sparse COO once and move into a mutable sparse map.
        auto [v_idx, v_val] = vector_to_coo_f64(values);
        std::unordered_map<std::uint64_t, double> values_map;
        values_map.reserve(v_idx.size() * 2 + 8);
        for (std::size_t i = 0; i < v_idx.size(); ++i) {
            values_map.emplace(v_idx[i], v_val[i]);
        }

        py::list result;

        Entry* prev = nullptr;
        Entry* curr = head_.get();

        while (curr != nullptr) {
            // Compute allocations for this order entry.
            std::vector<std::uint64_t> a_idx;
            std::vector<double> a_val;
            a_idx.reserve(curr->idx.size());
            a_val.reserve(curr->idx.size());

            // Update order in place: decrement remaining.
            for (std::size_t j = 0; j < curr->idx.size(); ++j) {
                const std::uint64_t k = curr->idx[j];
                double& order_x = curr->val[j];

                if (order_x <= 0.0) {
                    continue;
                }

                auto it = values_map.find(k);
                if (it == values_map.end()) {
                    continue;
                }

                double& avail = it->second;
                if (avail <= 0.0) {
                    continue;
                }

                const double alloc = (order_x < avail) ? order_x : avail;
                if (alloc <= 0.0) {
                    continue;
                }

                a_idx.push_back(k);
                a_val.push_back(alloc);

                order_x -= alloc;
                avail -= alloc;
                if (avail <= 0.0) {
                    values_map.erase(it);
                }
            }

            if (!a_idx.empty()) {
                py::object alloc_vec = vector_from_coo_f64(n, a_idx, a_val);
                result.append(py::make_tuple(curr->key, alloc_vec));
            }

            // Compress order (drop <=0 entries).
            std::vector<std::uint64_t> new_idx;
            std::vector<double> new_val;
            new_idx.reserve(curr->idx.size());
            new_val.reserve(curr->val.size());

            for (std::size_t j = 0; j < curr->idx.size(); ++j) {
                if (curr->val[j] > 0.0) {
                    new_idx.push_back(curr->idx[j]);
                    new_val.push_back(curr->val[j]);
                }
            }

            curr->idx = std::move(new_idx);
            curr->val = std::move(new_val);

            Entry* next = curr->next.get();

            const bool fulfilled = curr->idx.empty();
            if (fulfilled) {
                remove_next(prev);
                // prev unchanged
            } else {
                prev = curr;
            }

            curr = next;
        }

        // Build updated values vector from values_map and assign back to the passed Vector.
        std::vector<std::uint64_t> out_idx;
        std::vector<double> out_val;
        out_idx.reserve(values_map.size());
        out_val.reserve(values_map.size());

        for (const auto& kv : values_map) {
            if (kv.second > 0.0) {
                out_idx.push_back(kv.first);
                out_val.push_back(kv.second);
            }
        }

        // Sort by index for stable output
        std::vector<std::size_t> ord(out_idx.size());
        for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
            return out_idx[a] < out_idx[b];
        });

        std::vector<std::uint64_t> out_idx2;
        std::vector<double> out_val2;
        out_idx2.reserve(out_idx.size());
        out_val2.reserve(out_val.size());
        for (std::size_t i = 0; i < ord.size(); ++i) {
            out_idx2.push_back(out_idx[ord[i]]);
            out_val2.push_back(out_val[ord[i]]);
        }

        py::object new_values = vector_from_coo_f64(n, out_idx2, out_val2);

        // Mutate `values` in-place using the canonical python-graphblas idiom: values << new_values
        if (!py::hasattr(values, "__lshift__")) {
            throw py::type_error("values Vector does not support in-place assignment (missing __lshift__).");
        }
        values.attr("__lshift__")(new_values);

        if (size_ == 0) {
            length_.reset();
        }

        return result;
    }

    py::tuple __getstate__() const {
        py::list entries;

        Entry* curr = head_.get();
        while (curr != nullptr) {
            if (curr->idx.size() != curr->val.size()) {
                PyErr_SetString(PyExc_RuntimeError, "Internal error: idx/val size mismatch in Orderbook entry.");
                throw py::error_already_set();
            }

            const std::size_t n = curr->idx.size();

            py::array_t<std::uint64_t> idx_arr(static_cast<py::ssize_t>(n));
            py::array_t<double> val_arr(static_cast<py::ssize_t>(n));

            auto idx_w = idx_arr.mutable_unchecked<1>();
            auto val_w = val_arr.mutable_unchecked<1>();

            for (std::size_t i = 0; i < n; ++i) {
                idx_w(static_cast<py::ssize_t>(i)) = curr->idx[i];
                val_w(static_cast<py::ssize_t>(i)) = curr->val[i];
            }

            entries.append(py::make_tuple(curr->key, idx_arr, val_arr));
            curr = curr->next.get();
        }

        const std::int64_t dim = length_.value_or(0);
        return py::make_tuple(dim, entries);
    }

    void __setstate__(py::tuple state) {
        if (state.size() != 2) {
            throw py::value_error("Invalid state: expected (length, entries).");
        }

        clear();

        const std::int64_t n = state[0].cast<std::int64_t>();
        py::iterable entries = state[1].cast<py::iterable>();

        if (n > 0) {
            length_ = n;
        }

        for (py::handle item : entries) {
            py::tuple t = py::reinterpret_borrow<py::tuple>(item);
            if (t.size() != 3) {
                throw py::value_error("Invalid entry state: expected (key, idx_array, val_array).");
            }

            py::dict key = t[0].cast<py::dict>();

            py::array idx_arr = py::reinterpret_borrow<py::array>(t[1]);
            py::array val_arr = py::reinterpret_borrow<py::array>(t[2]);

            py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> idx_u64(idx_arr);
            py::array_t<double, py::array::c_style | py::array::forcecast> val_f64(val_arr);

            if (idx_u64.ndim() != 1 || val_f64.ndim() != 1 || idx_u64.shape(0) != val_f64.shape(0)) {
                throw py::value_error("Invalid entry arrays in state.");
            }

            std::vector<std::uint64_t> idx(static_cast<std::size_t>(idx_u64.shape(0)));
            std::vector<double> val(static_cast<std::size_t>(val_f64.shape(0)));

            auto idx_r = idx_u64.unchecked<1>();
            auto val_r = val_f64.unchecked<1>();

            for (py::ssize_t i = 0; i < idx_u64.shape(0); ++i) {
                idx[static_cast<std::size_t>(i)] = idx_r(i);
                val[static_cast<std::size_t>(i)] = val_r(i);
            }

            // Ensure sorted and positive-only (defensive)
            std::vector<std::size_t> order(idx.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return idx[a] < idx[b];
            });

            std::vector<std::uint64_t> idx2;
            std::vector<double> val2;
            idx2.reserve(idx.size());
            val2.reserve(val.size());
            for (std::size_t k = 0; k < order.size(); ++k) {
                const std::size_t i = order[k];
                if (val[i] > 0.0) {
                    idx2.push_back(idx[i]);
                    val2.push_back(val[i]);
                }
            }

            // Insert without re-extracting a Vector
            auto node = std::make_unique<Entry>(std::move(key), std::move(idx2), std::move(val2));
            Entry* node_raw = node.get();
            if (!head_) {
                head_ = std::move(node);
                tail_ = node_raw;
            } else {
                tail_->next = std::move(node);
                tail_ = node_raw;
            }
            ++size_;
        }

        if (size_ == 0) {
            length_.reset();
        }
    }

private:
    std::unique_ptr<Entry> head_;
    Entry* tail_ = nullptr;
    std::int64_t size_ = 0;
    std::optional<std::int64_t> length_;

    void clear() {
        head_.reset();
        tail_ = nullptr;
        size_ = 0;
        length_.reset();
    }

    void remove_next(Entry* prev) {
        if (!head_) return;

        if (prev == nullptr) {
            head_ = std::move(head_->next);
            if (!head_) tail_ = nullptr;
        } else {
            if (!prev->next) return;
            std::unique_ptr<Entry> doomed = std::move(prev->next);
            prev->next = std::move(doomed->next);
            if (!prev->next) tail_ = prev;
        }

        --size_;
        if (size_ == 0) {
            length_.reset();
        }
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "toolbox.orderbook C++ core (pybind11), sparse COO internal storage";

    py::class_<Orderbook>(m, "Orderbook")
        .def(py::init<>())
        .def("append", &Orderbook::append, py::arg("key"), py::arg("arr"),
             "Append an order. key is a dict (stored by reference). arr is a graphblas.Vector (not mutated).")
        .def("remove", &Orderbook::remove, py::arg("index"))
        .def("get_at_index", &Orderbook::get_at_index, py::arg("index"))
        .def("allocate_greedy", &Orderbook::allocate_greedy, py::arg("values"),
             "Greedy allocation against values (mutates values in-place via <<).")
        .def_property_readonly("size", &Orderbook::size)
        .def_property_readonly("length", &Orderbook::length)
        .def("__getstate__", &Orderbook::__getstate__)
        .def("__setstate__", &Orderbook::__setstate__);
}
