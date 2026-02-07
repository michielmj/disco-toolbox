// src/toolbox/orderbook/_core.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

// Safely extract raw bytes (binary-safe; supports embedded NUL bytes)
static std::string bytes_to_string(const py::bytes& b) {
    PyObject* obj = b.ptr();
    char* buf = nullptr;
    Py_ssize_t n = 0;
    if (PyBytes_AsStringAndSize(obj, &buf, &n) != 0) {
        throw py::error_already_set();
    }
    return std::string(buf, static_cast<std::size_t>(n));
}

static py::bytes string_to_bytes(const std::string& s) {
    return py::bytes(s.data(), static_cast<py::ssize_t>(s.size()));
}

static void ensure_1d(const py::array& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw py::type_error(std::string(name) + " must be a 1D array");
    }
}

} // namespace

class Orderbook {
public:
    Orderbook() = default;

    ~Orderbook() { clear(); }

    std::int64_t size() const { return size_; }
    std::int64_t length() const { return length_; }

    void append(py::bytes key, py::array arr) {
        // forcecast to float64, 1D, contiguous
        py::array_t<double, py::array::c_style | py::array::forcecast> a(arr);
        ensure_1d(a, "arr");

        const std::int64_t n = static_cast<std::int64_t>(a.shape(0));
        if (length_ == 0) {
            length_ = n;
        } else if (n != length_) {
            throw py::type_error("Length mismatch.");
        }

        const std::string k = bytes_to_string(key);

        Entry* node = new Entry(k, static_cast<std::size_t>(length_));
        const double* src = a.data();
        std::memcpy(node->data.data(), src, static_cast<std::size_t>(length_) * sizeof(double));

        // insert at end
        if (tail_ == nullptr) {
            head_ = node;
            tail_ = node;
        } else {
            tail_->next = node;
            tail_ = node;
        }
        ++size_;
    }

    void remove(std::int64_t index) {
        if (index < 0 || index >= size_) {
            throw py::index_error("Index out of range");
        }

        Entry* curr = head_;
        Entry* prev = nullptr;

        for (std::int64_t i = 0; i < index; ++i) {
            prev = curr;
            curr = curr->next;
        }

        if (prev == nullptr) {
            head_ = curr->next;
        } else {
            prev->next = curr->next;
        }

        if (curr->next == nullptr) {
            tail_ = prev;
        }

        delete curr;
        --size_;
        if (size_ == 0) {
            length_ = 0;
        }
    }

    py::tuple get_at_index(std::int64_t index) const {
        if (index < 0 || index >= size_) {
            throw py::index_error("Index out of range");
        }

        Entry* curr = head_;
        for (std::int64_t i = 0; i < index; ++i) {
            curr = curr->next;
        }

        py::bytes key = string_to_bytes(curr->key);

        // create numpy array copy
        py::array_t<double> out(static_cast<py::ssize_t>(length_));
        auto out_mut = out.mutable_unchecked<1>();
        for (std::int64_t i = 0; i < length_; ++i) {
            out_mut(static_cast<py::ssize_t>(i)) = curr->data[static_cast<std::size_t>(i)];
        }

        return py::make_tuple(key, out);
    }

    py::list allocate_greedy(py::array values) {
        if (length_ == 0) {
            return py::list();
        }

        // Must be writable because we mutate in-place.
        py::array_t<double, py::array::c_style | py::array::forcecast> v(values);
        ensure_1d(v, "values");

        if (!v.writeable()) {
            throw py::type_error("values must be a writable numpy array");
        }

        const std::int64_t n = static_cast<std::int64_t>(v.shape(0));
        if (n != length_) {
            throw py::type_error("Length mismatch.");
        }

        double* values_ptr = v.mutable_data();

        py::list result;

        Entry* curr = head_;
        Entry* prev = nullptr;

        while (curr != nullptr) {
            // We'll create allocation array lazily only if something > 0 is allocated
            bool make_new = true;
            py::array_t<double> alloc_arr;
            double remaining = 0.0;

            // Greedy allocation, in-place update of order and values
            for (std::int64_t i = 0; i < length_; ++i) {
                double allocation = curr->data[static_cast<std::size_t>(i)];
                const double avail = values_ptr[i];

                if (allocation > avail) {
                    allocation = avail;
                }

                if (allocation > 0.0) {
                    if (make_new) {
                        alloc_arr = py::array_t<double>(static_cast<py::ssize_t>(length_));
                        // zero it
                        std::memset(alloc_arr.mutable_data(), 0, static_cast<std::size_t>(length_) * sizeof(double));
                        make_new = false;
                    }
                    alloc_arr.mutable_data()[i] = allocation;

                    values_ptr[i] -= allocation;
                    curr->data[static_cast<std::size_t>(i)] -= allocation;
                }

                remaining += curr->data[static_cast<std::size_t>(i)];
            }

            if (!make_new) {
                result.append(py::make_tuple(string_to_bytes(curr->key), alloc_arr));
            }

            Entry* next = curr->next;
            if (remaining <= 0.0) {
                remove_next(prev);
                // prev unchanged
            } else {
                prev = curr;
            }
            curr = next;
        }

        if (size_ == 0) {
            length_ = 0;
        }

        return result;
    }

private:
    struct Entry {
        std::string key;          // binary bytes stored as std::string (size-aware)
        std::vector<double> data; // dense, fixed length
        Entry* next = nullptr;

        Entry(std::string k, std::size_t length)
            : key(std::move(k)), data(length, 0.0), next(nullptr) {}
    };

    Entry* head_ = nullptr;
    Entry* tail_ = nullptr;
    std::int64_t length_ = 0;
    std::int64_t size_ = 0;

    void clear() {
        Entry* curr = head_;
        while (curr != nullptr) {
            Entry* tmp = curr;
            curr = curr->next;
            delete tmp;
        }
        head_ = nullptr;
        tail_ = nullptr;
        length_ = 0;
        size_ = 0;
    }

    void remove_next(Entry* prev) {
        if (head_ == nullptr) return;

        Entry* curr = nullptr;

        // remove head
        if (prev == nullptr) {
            curr = head_;
            head_ = curr->next;
        } else {
            curr = prev->next;
        }

        // if prev is tail, do nothing
        if (curr == nullptr) return;

        // relink previous
        if (prev != nullptr) {
            prev->next = curr->next;
        }

        // set tail if last removed
        if (curr->next == nullptr) {
            tail_ = prev;
        }

        delete curr;
        --size_;
        if (size_ == 0) {
            length_ = 0;
        }
    }

public:
    // Pickle support (robust)
    py::list get_state() const {
        py::list state;
        Entry* curr = head_;
        while (curr != nullptr) {
            py::bytes key = string_to_bytes(curr->key);

            py::array_t<double> data(static_cast<py::ssize_t>(length_));
            auto data_mut = data.mutable_unchecked<1>();
            for (std::int64_t i = 0; i < length_; ++i) {
                data_mut(static_cast<py::ssize_t>(i)) = curr->data[static_cast<std::size_t>(i)];
            }

            state.append(py::make_tuple(key, data));
            curr = curr->next;
        }
        return state;
    }

    static std::unique_ptr<Orderbook> from_state(py::list state) {
        auto ob = std::make_unique<Orderbook>();
        for (py::handle item : state) {
            py::tuple t = py::reinterpret_borrow<py::tuple>(item);
            if (t.size() != 2) throw py::value_error("Invalid state");
            ob->append(py::reinterpret_borrow<py::bytes>(t[0]),
                       py::reinterpret_borrow<py::array>(t[1]));
        }
        return ob;
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "toolbox.orderbook C++ core (pybind11), binary keys + dense numpy float64 storage";

    py::class_<Orderbook>(m, "Orderbook")
        .def(py::init<>())
        .def("append", &Orderbook::append, py::arg("key"), py::arg("arr"),
             "Append an order. key: bytes. arr: 1D array-like castable to float64.")
        .def("remove", &Orderbook::remove, py::arg("index"))
        .def("get_at_index", &Orderbook::get_at_index, py::arg("index"))
        .def("allocate_greedy", &Orderbook::allocate_greedy, py::arg("values"),
             "Greedy allocation against values (mutates values in-place). values must be writable float64 1D.")
        .def_property_readonly("size", &Orderbook::size)
        .def_property_readonly("length", &Orderbook::length)
        .def(py::pickle(
            [](const Orderbook& ob) { return ob.get_state(); },
            [](py::list state) { return Orderbook::from_state(state); }
        ));
}
