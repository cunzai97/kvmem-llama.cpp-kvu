#include "llama-kvmem-diag.h"
#include "kvmem-vision.h"
#include "server-common.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

kvmem_prompt::kvmem_prompt(const std::vector<llama_token> & input)
    : tokens(input), native_(std::make_shared<server_tokens>(input, true)) {}

kvmem_prompt::kvmem_prompt(std::shared_ptr<server_tokens> native) : native_(std::move(native)) {
    tokens.reserve(native_->size());
    for (size_t i = 0; i < native_->size(); ++i) tokens.push_back((*native_)[i]);
    for (const auto & range : media_ranges()) {
        position_offsets_[range.second] = native_->pos_next(range.second) - (llama_pos) range.second;
    }
}

bool kvmem_prompt::has_media() const {
    return std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end();
}

size_t kvmem_prompt::common_prefix(const kvmem_prompt & other) const {
    return native_->get_common_prefix(*other.native_);
}

llama_pos kvmem_prompt::model_pos(size_t row) const {
    const auto it = std::prev(position_offsets_.upper_bound(row));
    return (llama_pos) row + it->second;
}

const mtmd_input_chunk * kvmem_prompt::chunk(size_t row) const {
    return native_->find_chunk(row).get();
}

size_t kvmem_prompt::media_end(size_t row) const {
    return row + mtmd_input_chunk_get_n_tokens(chunk(row));
}

std::vector<std::pair<uint32_t, uint32_t>> kvmem_prompt::media_ranges() const {
    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (size_t row = 0; row < tokens.size();) {
        if (tokens[row] != LLAMA_TOKEN_NULL) { ++row; continue; }
        const auto end = media_end(row);
        result.emplace_back(row, end);
        row = end;
    }
    return result;
}

std::shared_ptr<kvmem_prompt> kvmem_prompt::with_generated(const std::vector<llama_token> & gen) const {
    auto native = std::make_shared<server_tokens>(native_->clone());
    native->insert(gen);
    return std::make_shared<kvmem_prompt>(std::move(native));
}

common_chat_msg_spans kvmem_prompt::message_spans(const common_chat_msg_delimiters & delimiters) const {
    return native_->find_message_spans(delimiters);
}

std::vector<std::pair<uint32_t, std::string>> kvmem_prompt::media_identity() const {
    std::vector<std::pair<uint32_t, std::string>> ids;
    for (const auto & range : media_ranges()) ids.emplace_back(range.first, mtmd_input_chunk_get_id(chunk(range.first)));
    return ids;
}

std::shared_ptr<kvmem_prompt> kvmem_prompt::prefix(size_t rows) const {
    auto native = std::make_shared<server_tokens>(native_->clone());
    native->keep_first(rows);
    return std::make_shared<kvmem_prompt>(std::move(native));
}

std::shared_ptr<kvmem_prompt> kvmem_prompt::cache_index() const {
    auto native = std::make_shared<server_tokens>();
    native->has_mtmd = true;
    for (size_t row = 0; row < tokens.size();) {
        if (tokens[row] == LLAMA_TOKEN_NULL) {
            native->push_back_placeholder(chunk(row));
            row = media_end(row);
        } else {
            native->push_back(tokens[row++]);
        }
    }
    return std::make_shared<kvmem_prompt>(std::move(native));
}

size_t kvmem_prompt::index_bytes() const {
    // The private native token vector can retain growth capacity; bound it by
    // twice its length. Media entries here are placeholders in disk-cache mode.
    size_t bytes = sizeof(*this) + sizeof(server_tokens) + tokens.capacity()*sizeof(llama_token) +
        tokens.size()*sizeof(llama_token)*2 + position_offsets_.size()*64;
    for (const auto & range : media_ranges()) {
        size_t size = 0;
        if (mtmd_input_chunk_save(chunk(range.first), nullptr, 0, &size))
            throw std::runtime_error("cannot measure session media index");
        bytes += size + 256;
    }
    return bytes;
}

std::string kvmem_parse_media_messages(const std::string & body, bool allow_images,
                                      std::vector<std::vector<uint8_t>> & files) {
    auto parsed = common_json::parse(body);
    server_chat_params params;
    params.allow_image = allow_images;
    params.allow_audio = false;
    params.allow_video = false;
    oaicompat_chat_process_media(parsed, params, files);
    return parsed.dump();
}

kvmem_vision::kvmem_vision(llama_model * model, const std::string & path, bool gpu,
                           ggml_backend_dev_t device, int min_tokens, int max_tokens, int n_threads) {
    auto params = mtmd_context_params_default();
    params.media_marker = get_media_marker();
    params.use_gpu = gpu;
    params.device = gpu ? device : nullptr;
    params.image_min_tokens = min_tokens;
    params.image_max_tokens = max_tokens;
    if (n_threads > 0) params.n_threads = n_threads;
    // Native lazy warmup uses the real image instead of a fixed 2116-token dummy.
    params.warmup = false;
    params.print_timings = true;
    ctx_ = mtmd_init_from_file(path.c_str(), model, params);
    if (!ctx_) throw std::runtime_error("failed to load mmproj: " + path);
    n_embd_ = llama_model_n_embd_inp(model);
    kvmem_diag("KVMEM_TRACE vision_load device=%s embedding_width=%d min_tokens=%d max_tokens=%d threads=%d\n",
            gpu ? (device ? ggml_backend_dev_name(device) : "GPU-auto") : "CPU",
            n_embd_, min_tokens, max_tokens, params.n_threads);
}

kvmem_vision::~kvmem_vision() { mtmd_free(ctx_); }

std::shared_ptr<kvmem_prompt> kvmem_vision::tokenize(const std::string & prompt,
                                                 const std::vector<std::vector<uint8_t>> & files) {
    auto native = std::make_shared<server_tokens>(process_mtmd_prompt(ctx_, prompt, files, mtmd_helper_init_opt_default()));
    return std::make_shared<kvmem_prompt>(std::move(native));
}

int kvmem_vision::decode(llama_context * ctx, llama_context * ctx_dft, const kvmem_prompt & prompt, size_t row, int n_batch,
                       const std::function<int(llama_batch)> & dispatch) {
    const auto * chunk = prompt.chunk(row);
    const std::string id = mtmd_input_chunk_get_id(chunk);
    auto it = cache_.find(id);
    if (it == cache_.end()) {
        const size_t count = mtmd_input_chunk_get_n_tokens(chunk) * (size_t) n_embd_;
        constexpr size_t limit = 128ull*1024*1024;
        if (count > limit/sizeof(float)) throw std::runtime_error("image embeddings exceed 128 MiB; reduce --image-max-tokens");
        while (cache_bytes_ + count*sizeof(float) > limit && !cache_.empty()) {
            auto victim = std::min_element(cache_.begin(), cache_.end(), [](const auto & a, const auto & b) {
                return a.second.used < b.second.used;
            });
            cache_bytes_ -= victim->second.embd.size()*sizeof(float);
            cache_.erase(victim);
        }
        const auto start = std::chrono::steady_clock::now();
        using park_fn = bool (*)(int, size_t, bool);
        park_fn park = nullptr;
        const char * park_env = std::getenv("GGML_CUDA_VISION_PARK_MIB");
        size_t park_bytes = 0;
        if (park_env && std::string(park_env) != "0") {
            char * end = nullptr;
            const unsigned long long mib = std::strtoull(park_env, &end, 10);
            if (!*park_env || *end || mib < 64 || mib > 8192) {
                throw std::runtime_error("GGML_CUDA_VISION_PARK_MIB must be 64..8192 or 0");
            }
            const char * on_demand = std::getenv("MTMD_VISION_GPU_ON_DEMAND");
            if (!on_demand || std::string(on_demand) != "1") {
                throw std::runtime_error("CUDA vision weight parking requires MTMD_VISION_GPU_ON_DEMAND=1");
            }
            auto reg = ggml_backend_reg_by_name("CUDA");
            if (reg) {
                park = reinterpret_cast<park_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vision_park"));
            }
            if (!park) throw std::runtime_error("CUDA backend lacks VMM vision weight parking support");
            park_bytes = size_t(mib) * 1024 * 1024;
            llama_synchronize(ctx);
            if (ctx_dft) llama_synchronize(ctx_dft);
            kvmem_diag("KVMEM_TRACE vision_weight_park begin mib=%llu target=%p draft=%p\n",
                    mib, (void *) ctx, (void *) ctx_dft);
            if (!park(0, park_bytes, true)) throw std::runtime_error("could not park CUDA language-model weights for vision encode");
        }
        struct restore_guard {
            park_fn park;
            bool active;
            ~restore_guard() {
                if (active) {
                    if (!park(0, 0, false)) {
                        GGML_ABORT("cannot restore parked CUDA weights after vision encode; refusing unsafe decode");
                    }
                    kvmem_diag("KVMEM_TRACE vision_weight_park restored\n");
                }
            }
        } restore{park, park_bytes != 0};
        if (mtmd_encode_chunk(ctx_, chunk) != 0) throw std::runtime_error("vision encoder failed");
        ++encode_calls;
        encode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const float * embd = mtmd_get_output_embd(ctx_);
        if (!embd) throw std::runtime_error("vision encoder returned no embeddings");
        entry cached;
        cached.embd.assign(embd, embd + count);
        cache_bytes_ += count*sizeof(float);
        it = cache_.emplace(id, std::move(cached)).first;
    }
    it->second.used = ++clock_;
    struct callback_data {
        size_t row;
        const std::function<int(llama_batch)> * dispatch;
    } data {row, &dispatch};
    auto decode = [](llama_context *, llama_batch batch, void * opaque) -> int32_t {
        auto & data = *static_cast<callback_data *>(opaque);
        std::vector<llama_pos> logical(batch.n_tokens);
        for (int i = 0; i < batch.n_tokens; ++i) logical[i] = data.row + i;
        batch.logical_pos = logical.data();
        const int rc = (*data.dispatch)(batch);
        if (rc == 0) data.row += batch.n_tokens;
        return rc;
    };
    llama_pos next = prompt.model_pos(row);
    const int rc = mtmd_helper_decode_image_chunk_with_decoder(ctx_, ctx, chunk, it->second.embd.data(),
            next, 0, n_batch, &next, nullptr, &data, decode);
    if (rc == 0 && next != prompt.model_pos(prompt.media_end(row))) throw std::runtime_error("inconsistent image position cursor");
    return rc;
}
