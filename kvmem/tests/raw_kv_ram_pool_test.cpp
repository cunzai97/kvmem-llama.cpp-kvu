#include "kvmem/raw_kv_store.hpp"
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

static void test_shared_branches() {
    using namespace kvmem;
    RawKvStoreConfig cfg;
    cfg.n_layer = 2; cfg.n_embd_k = 1; cfg.n_embd_v = 1;
    cfg.block_tokens = 128; cfg.k_gpu_row_bytes = 1; cfg.v_gpu_row_bytes = 1;
    constexpr size_t page_bytes = 2 * (128 + 128 + sizeof(float));
    cfg.ram_pool = std::make_shared<RawKvRamPool>(100 * page_bytes);
    RawKvStore a(cfg), b(cfg);
    auto fill = [&](RawKvStore & store, uint32_t bid, uint8_t value) {
        std::vector<uint8_t> rows(128, value);
        std::vector<float> means(128, float(value));
        for (uint32_t il = 0; il < 2; ++il) {
            store.write_layer_k_gpu(bid * 128, 128, il, rows.data());
            store.write_layer_v_gpu(bid * 128, 128, il, rows.data());
            store.write_layer_mean_k(bid * 128, 128, il, means.data());
        }
    };
    auto read = [&](RawKvStore & store, uint32_t bid, uint8_t value) {
        uint8_t rows[128] = {};
        for (uint32_t il = 0; il < 2; ++il) {
            assert(store.copy_k_gpu(bid, il, rows, 128) && rows[0] == value && rows[127] == value);
            assert(store.copy_v_gpu(bid, il, rows, 128) && rows[0] == value && rows[127] == value);
            float mean = 0;
            store.mean_k(bid, il, &mean);
            assert(mean == float(value));
        }
    };
    for (uint32_t i = 0; i < 10; ++i) fill(a, i, uint8_t(i + 1));
    assert(cfg.ram_pool->used_bytes() == 10 * page_bytes);
    b.set_ram_pinned(true);
    a.clone_prefix_shared_to(b, 5 * 128);
    assert(cfg.ram_pool->used_bytes() == 10 * page_bytes); // not 15 pages
    for (uint32_t i = 0; i < 5; ++i) read(b, i, uint8_t(i + 1));
    bool rejected = false;
    try { a.clone_prefix_shared_to(b, 128); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected); // destination must be empty
    rejected = false;
    try { a.clone_prefix_shared_to(b, 127); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    RawKvStoreConfig incomplete_cfg = cfg;
    incomplete_cfg.ram_pool = std::make_shared<RawKvRamPool>(100 * page_bytes);
    RawKvStore incomplete(incomplete_cfg), incomplete_dst(incomplete_cfg);
    std::vector<uint8_t> packed(128, 9);
    incomplete.write_layer_k_gpu(0, 128, 0, packed.data());
    rejected = false;
    try { incomplete.clone_prefix_shared_to(incomplete_dst, 128); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected && !incomplete_dst.has_block(0)); // no partial publication
    // A single-row overwrite must detach the entire physical page, not A's copy.
    uint8_t changed = 77;
    b.write_layer_k_gpu(2 * 128, 1, 0, &changed);
    assert(cfg.ram_pool->used_bytes() == 11 * page_bytes);
    uint8_t row[128] = {};
    assert(a.copy_k_gpu(2, 0, row, 128) && row[0] == 3);
    assert(b.copy_k_gpu(2, 0, row, 128) && row[0] == 77);
    fill(b, 5, 66); // divergent suffix, A retains its own block 5
    read(a, 5, 6);
    read(b, 5, 66);
    assert(cfg.ram_pool->used_bytes() == 12 * page_bytes);
    b.truncate_to(3 * 128 + 16); // COW tail's mean and token counts
    assert(a.n_tokens(3) == 128 && b.n_tokens(3) == 16);
    read(a, 3, 4);
    assert(cfg.ram_pool->used_bytes() == 12 * page_bytes);
    b.invalidate_packed_from(128);
    assert(a.has_k_gpu(1, 0, 128) && a.has_v_gpu(1, 1, 128));
    assert(!b.has_k_gpu(1, 0, 128) && !b.has_v_gpu(1, 1, 128));
    assert(cfg.ram_pool->used_bytes() == 13 * page_bytes);
    b.clear();
    assert(cfg.ram_pool->used_bytes() == 10 * page_bytes);
    read(a, 2, 3);
    RawKvStore partial(cfg);
    a.clone_prefix_shared_to(partial, 128 + 1);
    assert(cfg.ram_pool->used_bytes() == 11 * page_bytes);
    assert(partial.n_tokens(0) == 128 && partial.n_tokens(1) == 1);
    uint8_t first[1] = {};
    assert(partial.copy_k_gpu(1, 0, first, 1) && first[0] == 2);
    assert(!partial.has_k_gpu(1, 0, 128));
    partial.write_layer_k_gpu(128, 1, 0, &changed);
    read(a, 1, 2);
    partial.clear();
    assert(cfg.ram_pool->used_bytes() == 10 * page_bytes);

    // A cold source may lose *only* its unreferenced suffix while the pinned
    // child keeps all five shared physical pages intact under pressure.
    cfg.ram_pool = std::make_shared<RawKvRamPool>(5 * page_bytes);
    RawKvStore old(cfg), child(cfg);
    for (uint32_t i = 0; i < 10; ++i) fill(old, i, uint8_t(i + 1));
    old.set_fork_pinned(true);
    old.set_ram_pinned(false); // source detaches before the branch copies it
    assert(!old.ram_evicted() && cfg.ram_pool->used_bytes() == 10 * page_bytes);
    old.clone_prefix_shared_to(child, 5 * 128);
    const uintptr_t shared0 = old.physical_page_id(0);
    const uintptr_t discarded5 = old.physical_page_id(5);
    old.set_fork_pinned(false);
    assert(old.ram_evicted());
    assert(!child.ram_evicted());
    assert(cfg.ram_pool->used_bytes() == 5 * page_bytes);
    assert(old.physical_page_id(0) == shared0 && child.physical_page_id(0) == shared0);
    assert(old.physical_page_id(5) == discarded5 && !old.has_k_gpu(5, 0, 128));
    // Only blocks 5..9 are gone; the shared prefix pages remain physically
    // identical. A checkpoint at row 640 permits suffix-only regeneration.
    bool refused = false;
    old.set_ram_pinned(true);
    try { old.recover_evicted_suffix(4 * 128); }
    catch (const std::runtime_error &) { refused = true; }
    assert(refused && old.ram_evicted() && old.physical_page_id(0) == shared0);
    old.recover_evicted_suffix(5 * 128);
    assert(!old.ram_evicted() && old.physical_page_id(0) == shared0);
    for (uint32_t i = 0; i < 5; ++i) read(old, i, uint8_t(i + 1));
    assert(old.physical_page_id(1) == child.physical_page_id(1));
    assert(!old.has_block(5) && cfg.ram_pool->used_bytes() == 5 * page_bytes);
    fill(old, 5, 77); // recompute one missing page, COW leaves prefix untouched
    read(old, 5, 77);
    assert(old.physical_page_id(0) == shared0 && cfg.ram_pool->used_bytes() == 6 * page_bytes);
    for (uint32_t i = 0; i < 5; ++i) read(child, i, uint8_t(i + 1));
    old.clear(); // dropping the old reference cannot free the child's pages
    assert(cfg.ram_pool->used_bytes() == 5 * page_bytes);
    read(child, 0, 1);
    child.clear();
    assert(cfg.ram_pool->used_bytes() == 0);

    // A partial-write/allocation failure must never be mistaken for an
    // ordinary, recoverable whole-page eviction, even if the tail is empty.
    cfg.ram_pool = std::make_shared<RawKvRamPool>(page_bytes);
    RawKvStore poisoned(cfg);
    fill(poisoned, 0, 9);
    fill(poisoned, 1, 10);
    poisoned.set_ram_pinned(false);
    assert(poisoned.ram_recoverable_eviction());
    poisoned.set_ram_pinned(true);
    poisoned.mark_ram_unusable();
    assert(!poisoned.ram_recoverable_eviction());
    bool poison_rejected = false;
    try { poisoned.recover_evicted_suffix(128); }
    catch (const std::runtime_error &) { poison_rejected = true; }
    assert(poison_rejected);
    poisoned.clear();

    cfg.ram_pool = std::make_shared<RawKvRamPool>(0);
    RawKvStore x(cfg), y(cfg);
    fill(x, 0, 42);
    x.clone_prefix_shared_to(y, 128);
    assert(cfg.ram_pool->used_bytes() == page_bytes);
    x.set_ram_pinned(false);
    assert(!x.ram_evicted() && !y.ram_evicted()); // y still pins the page
    y.set_ram_pinned(false);
    assert(x.ram_evicted() && y.ram_evicted()); // both aliases invalidated together
    assert(!x.has_k_gpu(0, 0, 128) && !y.has_v_gpu(0, 0, 128));
    assert(cfg.ram_pool->used_bytes() == 0);
}

int main() {
    auto pool = std::make_shared<kvmem::RawKvRamPool>(64);
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 1; cfg.n_embd_k = 2; cfg.n_embd_v = 2;
    cfg.block_tokens = 4; cfg.k_gpu_row_bytes = 4; cfg.v_gpu_row_bytes = 4;
    cfg.ram_pool = pool;
    auto active = std::make_unique<kvmem::RawKvStore>(cfg);
    auto cold = std::make_unique<kvmem::RawKvStore>(cfg);
    uint8_t a[16] = {1}, b[16] = {2}, out[16] = {};
    active->write_layer_k_gpu(0, 4, 0, a);
    active->write_layer_v_gpu(0, 4, 0, a);
    cold->set_ram_pinned(false);
    cold->write_layer_k_gpu(0, 4, 0, b);
    cold->write_layer_v_gpu(0, 4, 0, b);
    assert(pool->used_bytes() == 64);
    // Next block tips the shared cap, dropping the old block's K AND V.
    cold->write_layer_k_gpu(4, 4, 0, b);
    assert(pool->used_bytes() <= pool->limit_bytes());
    assert(active->copy_k_gpu(0, 0, out, 4) && out[0] == 1);
    assert(active->copy_v_gpu(0, 0, out, 4) && out[0] == 1);
    assert(!cold->has_block(0) && !cold->has_k_gpu(0, 0, 4) && !cold->has_v_gpu(0, 0, 4));
    assert(cold->ram_evicted());
    bool rejected = false;
    // Switching to a partially evicted store is permitted: attach must turn
    // it into a full cache miss and reset it before any inference uses it.
    cold->set_ram_pinned(true);
    try { cold->write_layer_v_gpu(4, 4, 0, b); rejected = false; }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    cold->clear();
    assert(!cold->ram_evicted());
    cold->set_ram_pinned(true);
    cold->write_layer_k_gpu(0, 4, 0, b);
    cold->write_layer_v_gpu(0, 4, 0, b);
    assert(pool->used_bytes() == 64);
    cold.reset();
    assert(pool->used_bytes() == 32);
    // An active request may exceed the soft cap, but remains intact. Once it
    // is detached the pool must trim it, never silently evict live KV.
    active->write_layer_k_gpu(4, 4, 0, a);
    active->write_layer_v_gpu(4, 4, 0, a);
    active->write_layer_k_gpu(8, 4, 0, a);
    assert(!active->ram_evicted() && pool->used_bytes() > pool->limit_bytes());
    active->set_ram_pinned(false);
    assert(pool->used_bytes() <= pool->limit_bytes());
    assert(active->ram_evicted());
    active->clear();
    assert(!active->ram_evicted());
    // Explicit RAM-only mode rejects NVMe even if the directory is set.
    cfg.nvme_bytes = 1024;
    rejected = false;
    try { kvmem::RawKvStore invalid(cfg); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    test_shared_branches();
    return 0;
}
