"""
Dual-backend KV Cache — auto-selects based on GPU:
  SM75 (T4):         Contiguous cache + torch SDPA — zero-copy reads
  SM80+ (L4/A100+):  FlashInfer paged attention + radix prefix caching
"""

import math, time, threading, torch, torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set


_FLASHINFER_AVAILABLE = False
try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
except ImportError:
    flashinfer = None

_torch_ver = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
_SDPA_GQA = _torch_ver >= (2, 5)


def get_sm_version(device=None) -> int:
    if device is None:
        device = 0
    if isinstance(device, torch.device):
        device = device.index or 0
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def should_use_flashinfer(device=None) -> bool:
    sm = get_sm_version(device)
    if not _FLASHINFER_AVAILABLE:
        if sm >= 80:
            print(f"[KVCache] ⚠  SM{sm} supports FlashInfer but it's not installed.")
            print(f"[KVCache] pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/")
            print(f"[KVCache] Falling back to contiguous SDPA (slower decode).")
        return False
    return sm >= 80


def _sdpa(q, k, v, attn_mask=None, is_causal=False):
    """q:[B,Hq,Sq,D]  k,v:[B,Hkv,Skv,D]"""
    need_gqa = q.shape[1] != k.shape[1]
    if need_gqa:
        if _SDPA_GQA:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                enable_gqa=True)
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal)


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int          # per TP rank
    num_qo_heads: int          # per TP rank
    head_dim: int
    page_size: int = 16        # FlashInfer path
    max_pages: int = 0         # 0 = auto (FlashInfer)
    max_batch_size: int = 8    # contiguous path
    max_seq_len: int = 4096    # contiguous path
    memory_fraction: float = 0.30
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = None
    enable_prefix_caching: bool = True
    backend: str = "auto"      # "auto" | "flashinfer" | "contiguous"

    def __post_init__(self):
        if self.backend == "auto":
            self.backend = ("flashinfer"
                            if should_use_flashinfer(self.device)
                            else "contiguous")
        if self.backend == "flashinfer" and self.max_pages <= 0:
            self.max_pages = self._auto_pages()
        if self.backend == "contiguous":
            self._auto_seq_len()

    def _elem_bytes(self) -> int:
        return {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(self.dtype, 2)

    def _auto_pages(self) -> int:
        if self.device is None:
            return 256
        try:
            free, total = torch.cuda.mem_get_info(self.device)
            headroom = max(2 << 30, int(total * 0.15))
            budget = int(max(0, free - headroom) * self.memory_fraction)
            per_page = (2 * self.num_layers * self.page_size
                        * self.num_kv_heads * self.head_dim * self._elem_bytes())
            n = max(64, min(budget // max(per_page, 1), 65536))
            print(f"[KVCache] FlashInfer: {n} pages, "
                  f"budget={budget / (1 << 20):.0f}MB")
            return n
        except Exception:
            return 256

    def _auto_seq_len(self):
        if self.device is None:
            return
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
            budget = int(free * self.memory_fraction * 0.5)
            per_tok = (2 * self.num_layers * self.num_kv_heads
                       * self.head_dim * self._elem_bytes())
            cap = budget // max(per_tok, 1) // max(self.max_batch_size, 1)
            self.max_seq_len = min(self.max_seq_len, max(512, cap))
            print(f"[KVCache] Contiguous: batch={self.max_batch_size}, "
                  f"max_seq={self.max_seq_len}")
        except Exception:
            pass

    @classmethod
    def from_hf_config(cls, hf_config, parallel_config=None,
                       device=None, **kw) -> "KVCacheConfig":
        nl = getattr(hf_config, "num_hidden_layers",
             getattr(hf_config, "num_layers", 32))
        nkv = getattr(hf_config, "num_key_value_heads",
              getattr(hf_config, "multi_query_group_num",
              getattr(hf_config, "num_attention_heads", 32)))
        nq = getattr(hf_config, "num_attention_heads", 32)
        hd = getattr(hf_config, "head_dim",
             getattr(hf_config, "kv_channels",
             hf_config.hidden_size // nq))
        if parallel_config is not None:
            tp = parallel_config.tp_size
            if tp > 1:
                nkv //= tp;  nq //= tp
            if (parallel_config.pp_size > 1
                    and parallel_config.pp_layer_splits):
                try:
                    from .parallel import ParallelContext
                    ctx = ParallelContext.get()
                    if ctx and ctx.pp_rank in parallel_config.pp_layer_splits:
                        nl = len(parallel_config.pp_layer_splits[ctx.pp_rank])
                except Exception:
                    pass
        return cls(num_layers=nl, num_kv_heads=nkv,
                   num_qo_heads=nq, head_dim=hd, device=device, **kw)



class CacheContext(ABC):
    seq_ids: List[str]
    new_token_counts: List[int]
    input_offsets: List[int]
    batch_size: int
    seq_lens: List[int]

    @abstractmethod
    def get_position_ids(self, max_input_len: int) -> torch.Tensor: ...
    @abstractmethod
    def attend(self, layer_idx: int,
               q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
               n_heads: int, n_kv_heads: int) -> torch.Tensor: ...
    @abstractmethod
    def get_start_pos(self, batch_idx: int = 0) -> int: ...


class KVCacheManager(ABC):
    cfg: KVCacheConfig
    @abstractmethod
    def allocate_sequence(self, seq_id: str,
                          token_ids: Optional[List[int]] = None,
                          num_tokens: int = 0) -> int: ...
    @abstractmethod
    def extend_sequence(self, seq_id: str, num_new: int = 1): ...
    @abstractmethod
    def begin_forward(self, seq_ids: List[str],
                      new_token_counts: List[int],
                      input_offsets: Optional[List[int]] = None
                      ) -> CacheContext: ...
    @abstractmethod
    def end_forward(self, ctx: CacheContext): ...
    @abstractmethod
    def release_sequence(self, seq_id: str): ...
    @abstractmethod
    def append_token_ids(self, seq_id: str, new_ids: List[int]): ...
    @abstractmethod
    def reset(self): ...
    @property
    @abstractmethod
    def stats(self) -> Dict: ...


def _build_causal_mask(batch_size, input_width, start_positions,
                       new_token_counts, input_offsets, seq_lens,
                       max_kv_len, device, dtype):
    """Returns [B,1,Q,KV] float mask or None if not needed."""
    B = batch_size
    # Fast paths
    if B == 1 and start_positions[0] == 0 and input_offsets[0] == 0:
        return None                         # single prefill from 0
    if (all(n == 1 for n in new_token_counts)
            and len(set(seq_lens)) == 1):
        return None                         # uniform decode
    if (all(s == 0 for s in start_positions)
            and all(o == 0 for o in input_offsets)
            and len(set(new_token_counts)) == 1):
        return None                         # uniform prefill from 0

    neg = -65504.0 if dtype == torch.float16 else float("-inf")
    mask = torch.full((B, 1, input_width, max_kv_len),
                      neg, dtype=dtype, device=device)
    for b in range(B):
        off = input_offsets[b]
        ntok = new_token_counts[b]
        base = start_positions[b]
        kv = seq_lens[b]
        if ntok == 0:
            continue
        qp = torch.arange(ntok, device=device) + base
        kp = torch.arange(max_kv_len, device=device)
        ok = (kp[None, :] <= qp[:, None]) & (kp[None, :] < kv)
        mask[b, 0, off:off + ntok] = torch.where(
            ok, torch.tensor(0., dtype=dtype, device=device),
            torch.tensor(neg, dtype=dtype, device=device))
    return mask


class _ContiguousPool:
    def __init__(self, cfg: KVCacheConfig):
        shape = (cfg.num_layers, cfg.max_batch_size, cfg.max_seq_len,
                 cfg.num_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self.v = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        mb = self.k.nelement() * cfg._elem_bytes() * 2 / (1 << 20)
        print(f"[ContiguousPool] {cfg.max_batch_size}×{cfg.max_seq_len} "
              f"= {mb:.0f} MB  {cfg.device}")


class _ContigSeqState:
    __slots__ = ("seq_id", "slot", "num_tokens", "token_ids")
    def __init__(self, seq_id, slot):
        self.seq_id = seq_id
        self.slot = slot
        self.num_tokens = 0
        self.token_ids: List[int] = []


class ContiguousCacheContext(CacheContext):
    """Per-forward context for contiguous backend."""

    def __init__(self, cache: "ContiguousKVCache",
                 seq_ids, new_token_counts, input_offsets):
        self.cache = cache
        self.seq_ids = seq_ids
        self.new_token_counts = new_token_counts
        self.input_offsets = input_offsets or [0] * len(seq_ids)
        self.batch_size = B = len(seq_ids)
        cfg = cache.cfg;  dev = cfg.device

        self.slots = [cache.states[s].slot for s in seq_ids]
        self.start_positions = [cache.states[s].num_tokens for s in seq_ids]
        self.seq_lens = [self.start_positions[i] + new_token_counts[i]
                         for i in range(B)]
        self.max_kv_len = max(self.seq_lens) if self.seq_lens else 0
        self._is_decode = all(n == 1 for n in new_token_counts)

        # input width
        self._iw = (max(o + n for o, n in
                        zip(self.input_offsets, new_token_counts))
                    if B else 1)

        # position IDs [B, iw]
        self._pos = torch.zeros(B, self._iw, dtype=torch.long, device=dev)
        for b in range(B):
            o = self.input_offsets[b]; n = new_token_counts[b]
            if n > 0:
                self._pos[b, o:o + n] = torch.arange(
                    self.start_positions[b],
                    self.start_positions[b] + n, device=dev)

        # vectorised decode indices
        if self._is_decode and B:
            self._d_batch = torch.arange(B, device=dev)
            self._d_pos = torch.tensor(self.start_positions,
                                       dtype=torch.long, device=dev)

        # mask (None when not needed)
        self._mask = _build_causal_mask(
            B, self._iw, self.start_positions, new_token_counts,
            self.input_offsets, self.seq_lens, self.max_kv_len,
            dev, cfg.dtype)

    def get_position_ids(self, S: int) -> torch.Tensor:
        return self._pos[:, :S] if S <= self._iw else torch.cat(
            [self._pos, self._pos.new_zeros(
                self.batch_size, S - self._iw)], 1)

    def attend(self, layer_idx, q, k_new, v_new, n_heads, n_kv_heads):
        B, S, _, D = q.shape
        pool = self.cache.pool

        if self._is_decode:
            pool.k[layer_idx, self._d_batch, self._d_pos] = k_new[:, 0]
            pool.v[layer_idx, self._d_batch, self._d_pos] = v_new[:, 0]
        else:
            for b in range(B):
                o = self.input_offsets[b];  n = self.new_token_counts[b]
                s = self.start_positions[b]
                if n > 0:
                    pool.k[layer_idx, b, s:s + n] = k_new[b, o:o + n]
                    pool.v[layer_idx, b, s:s + n] = v_new[b, o:o + n]

        kv = self.max_kv_len
        k_t = pool.k[layer_idx, :B, :kv].transpose(1, 2).contiguous()
        v_t = pool.v[layer_idx, :B, :kv].transpose(1, 2).contiguous()
        q_t = q.transpose(1, 2)

        mask = self._mask
        if mask is not None:
            mask = mask[:, :, :S, :kv].to(dtype=q.dtype)
        out = _sdpa(q_t, k_t, v_t, attn_mask=mask,
                    is_causal=(mask is None and S > 1))
        return out.transpose(1, 2).contiguous().reshape(B, S, -1)

    def get_start_pos(self, idx=0):
        return self.start_positions[idx]


class ContiguousKVCache(KVCacheManager):
    """Simple pre-allocated KV cache — no paging, no prefix caching."""

    def __init__(self, cfg: KVCacheConfig):
        self.cfg = cfg
        self.pool = _ContiguousPool(cfg)
        self.states: Dict[str, _ContigSeqState] = {}
        self._slot = 0

    def allocate_sequence(self, seq_id, token_ids=None, num_tokens=0):
        s = self._slot;  self._slot += 1
        assert s < self.cfg.max_batch_size, "Exceeded max_batch_size"
        self.pool.k[:, s].zero_();  self.pool.v[:, s].zero_()
        st = _ContigSeqState(seq_id, s)
        if token_ids:
            st.token_ids = list(token_ids)
        self.states[seq_id] = st
        return 0                            # no prefix caching

    def extend_sequence(self, seq_id, num_new=1):
        st = self.states[seq_id]
        assert st.num_tokens + num_new <= self.cfg.max_seq_len

    def begin_forward(self, seq_ids, ntoks, offsets=None):
        for sid, n in zip(seq_ids, ntoks):
            self.extend_sequence(sid, n)
        return ContiguousCacheContext(self, seq_ids, ntoks, offsets)

    def end_forward(self, ctx):
        for b in range(ctx.batch_size):
            self.states[ctx.seq_ids[b]].num_tokens += ctx.new_token_counts[b]

    def release_sequence(self, seq_id):
        if seq_id in self.states:
            del self.states[seq_id]
            if not self.states:
                self._slot = 0

    def append_token_ids(self, seq_id, ids):
        if seq_id in self.states:
            self.states[seq_id].token_ids.extend(ids)

    def reset(self):
        for s in list(self.states):
            self.release_sequence(s)

    @property
    def stats(self):
        return {"backend": "contiguous",
                "active_seqs": len(self.states),
                "slots": f"{self._slot}/{self.cfg.max_batch_size}",
                "max_seq": self.cfg.max_seq_len}


class PagePool:
    """
    k,v : [num_layers, max_pages, page_size, num_kv_heads, head_dim]
    Page 0 = permanent null.
    """
    def __init__(self, cfg: KVCacheConfig):
        self.cfg = cfg
        shape = (cfg.num_layers, cfg.max_pages, cfg.page_size,
                 cfg.num_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self.v = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self.ref_count = [0] * cfg.max_pages
        self.tokens_used = [0] * cfg.max_pages
        self.ref_count[0] = 1
        self.free_set: Set[int] = set(range(1, cfg.max_pages))
        self._lock = threading.Lock()
        mb = self.k.nelement() * cfg._elem_bytes() * 2 / (1 << 20)
        print(f"[PagePool] {cfg.max_pages}×{cfg.page_size} = "
              f"{cfg.max_pages * cfg.page_size} cap, {mb:.0f} MB")

    def alloc(self, n=1) -> Optional[List[int]]:
        with self._lock:
            if len(self.free_set) < n:
                return None
            out = []
            for _ in range(n):
                pid = self.free_set.pop()
                self.ref_count[pid] = 1
                self.tokens_used[pid] = 0
                self.k[:, pid].zero_();  self.v[:, pid].zero_()
                out.append(pid)
            return out

    def release(self, pid):
        if pid <= 0: return
        with self._lock:
            self.ref_count[pid] -= 1
            if self.ref_count[pid] <= 0:
                self.ref_count[pid] = 0
                self.tokens_used[pid] = 0
                self.free_set.add(pid)

    def add_ref(self, pid):
        with self._lock: self.ref_count[pid] += 1

    def is_shared(self, pid): return self.ref_count[pid] > 1

    def cow(self, pid):
        new = self.alloc(1)
        if new is None: return None
        d = new[0]
        self.k[:, d].copy_(self.k[:, pid])
        self.v[:, d].copy_(self.v[:, pid])
        self.tokens_used[d] = self.tokens_used[pid]
        return d

    @property
    def num_free(self): return len(self.free_set)
    @property
    def utilization(self):
        return 1.0 - self.num_free / max(self.cfg.max_pages - 1, 1)


class _RNode:
    __slots__ = ("children", "page_id", "ref", "last_access", "depth")
    def __init__(self, d=0):
        self.children: Dict[int, "_RNode"] = {}
        self.page_id = -1;  self.ref = 0
        self.last_access = 0.0;  self.depth = d


class RadixTree:
    def __init__(self, ps, pool):
        self.root = _RNode()
        self.ps = ps;  self.pool = pool
        self._lock = threading.Lock()

    def match(self, tokens):
        with self._lock:
            node = self.root;  pages = [];  now = time.monotonic()
            for i, tok in enumerate(tokens):
                ch = node.children.get(tok)
                if ch is None: break
                node = ch;  node.last_access = now
                d = i + 1
                if d % self.ps == 0 and node.page_id >= 0:
                    if self.pool.ref_count[node.page_id] > 0:
                        pages.append(node.page_id)
                    else:
                        node.page_id = -1;  break
            return len(pages) * self.ps, pages

    def insert(self, tokens, page_ids):
        with self._lock:
            node = self.root;  pi = 0;  now = time.monotonic()
            for i, tok in enumerate(tokens):
                if tok not in node.children:
                    node.children[tok] = _RNode(i + 1)
                node = node.children[tok];  node.last_access = now
                d = i + 1
                if d % self.ps == 0 and pi < len(page_ids):
                    incoming = page_ids[pi];  pi += 1
                    if node.page_id < 0:
                        node.page_id = incoming
                        self.pool.add_ref(incoming)

    def add_refs(self, tokens, n=1):
        with self._lock:
            node = self.root
            for tok in tokens:
                ch = node.children.get(tok)
                if ch is None: break
                node = ch;  node.ref += n

    def dec_refs(self, tokens, n=1):
        with self._lock:
            node = self.root
            for tok in tokens:
                ch = node.children.get(tok)
                if ch is None: break
                node = ch;  node.ref = max(0, node.ref - n)

    def evict_lru(self, need):
        with self._lock:
            cands = [];  self._collect(self.root, cands)
            cands.sort(key=lambda x: x[0])
            freed = 0
            for _, nd, par, edge in cands:
                if freed >= need: break
                if nd.ref > 0: continue
                if nd.page_id >= 0:
                    self.pool.release(nd.page_id)
                    nd.page_id = -1;  freed += 1
                if not nd.children and par is not None:
                    par.children.pop(edge, None)
            return freed

    def _collect(self, node, out, par=None, edge=-1):
        if node.page_id >= 0 and node.ref == 0:
            out.append((node.last_access, node, par, edge))
        for t, ch in list(node.children.items()):
            self._collect(ch, out, node, t)

    @property
    def cached_pages(self):
        c = [0]
        def _cnt(n):
            if n.page_id >= 0: c[0] += 1
            for ch in n.children.values(): _cnt(ch)
        _cnt(self.root);  return c[0]


class _PagedSeqState:
    __slots__ = ("seq_id", "block_table", "num_tokens",
                 "prefix_len", "token_ids")
    def __init__(self, sid, prefix_pages=None, plen=0, tids=None):
        self.seq_id = sid
        self.block_table = list(prefix_pages or [])
        self.num_tokens = plen
        self.prefix_len = plen
        self.token_ids = list(tids or [])


class FlashInferCacheContext(CacheContext):

    def __init__(self, cache: "FlashInferPagedKVCache",
                 seq_ids, new_token_counts, input_offsets):
        self.cache = cache
        self.seq_ids = seq_ids
        self.new_token_counts = new_token_counts
        self.input_offsets = input_offsets or [0] * len(seq_ids)
        self.batch_size = B = len(seq_ids)

        cfg = cache.cfg;  dev = cfg.device;  pool = cache.pool;  ps = cfg.page_size

        self.start_positions = [cache.seqs[s].num_tokens for s in seq_ids]
        self.seq_lens = [self.start_positions[i] + new_token_counts[i]
                         for i in range(B)]
        self.max_kv_len = max(self.seq_lens) if self.seq_lens else 0
        self._is_decode = all(n == 1 for n in new_token_counts)

        self._iw = (max(o + n for o, n in
                        zip(self.input_offsets, new_token_counts))
                    if B else 1)

        self._pos = torch.zeros(B, self._iw, dtype=torch.long, device=dev)
        for b in range(B):
            o = self.input_offsets[b]; n = new_token_counts[b]
            if n > 0:
                self._pos[b, o:o + n] = torch.arange(
                    self.start_positions[b],
                    self.start_positions[b] + n, device=dev)

        self._write_info: List[List[Tuple[int, int]]] = []
        for b in range(B):
            st = cache.seqs[seq_ids[b]]
            ntok = new_token_counts[b]
            start = st.num_tokens;  info = []
            for t in range(ntok):
                ap = start + t
                pi = ap // ps;  sl = ap % ps
                if pi >= len(st.block_table):
                    info.append((0, 0)); continue
                pid = st.block_table[pi]
                if pool.is_shared(pid):
                    np_ = pool.cow(pid)
                    if np_ is None:
                        raise RuntimeError("KV OOM during CoW")
                    st.block_table[pi] = np_
                    pool.release(pid);  pid = np_
                info.append((pid, sl))
                pool.tokens_used[pid] = max(pool.tokens_used[pid], sl + 1)
            self._write_info.append(info)

        self._plan_flashinfer(dev, cfg)

    def _plan_flashinfer(self, dev, cfg):
        B = self.batch_size;  ps = cfg.page_size
        indptr = [0]; indices = []; lpl = []
        for b in range(B):
            st = self.cache.seqs[self.seq_ids[b]]
            kv = self.seq_lens[b]
            np_ = math.ceil(kv / ps) if kv > 0 else 0
            for p in range(np_):
                indices.append(
                    st.block_table[p] if p < len(st.block_table) else 0)
            indptr.append(indptr[-1] + np_)
            lpl.append((kv - (np_ - 1) * ps) if np_ > 0 else 0)

        self._fi_ind = torch.tensor(indptr, dtype=torch.int32, device=dev)
        self._fi_idx = torch.tensor(indices or [0],
                                    dtype=torch.int32, device=dev)
        self._fi_lpl = torch.tensor(lpl or [0],
                                    dtype=torch.int32, device=dev)

        if self._is_decode:
            self.cache._dec_wrap.plan(
                self._fi_ind, self._fi_idx, self._fi_lpl,
                cfg.num_qo_heads, cfg.num_kv_heads,
                cfg.head_dim, ps)
        else:
            qo = [0]
            for b in range(B):
                qo.append(qo[-1] + self.new_token_counts[b])
            self._fi_qo = torch.tensor(qo, dtype=torch.int32, device=dev)
            self.cache._pf_wrap.plan(
                self._fi_qo,
                self._fi_ind, self._fi_idx, self._fi_lpl,
                cfg.num_qo_heads, cfg.num_kv_heads,
                cfg.head_dim, ps,
                causal=True)


    def get_position_ids(self, S):
        return self._pos[:, :S] if S <= self._iw else torch.cat(
            [self._pos, self._pos.new_zeros(self.batch_size, S - self._iw)], 1)

    def attend(self, layer_idx, q, k_new, v_new, n_heads, n_kv_heads):
        B, S = q.shape[0], q.shape[1]

        # write
        self._write_kv(layer_idx, k_new, v_new)

        pool = self.cache.pool
        kv = (pool.k[layer_idx], pool.v[layer_idx])

        if self._is_decode:
            q_fi = q[:, 0]                       # [B, Hq, D]
            out = self.cache._dec_wrap.run(q_fi, kv)   # [B, Hq, D]
            return out.reshape(B, 1, -1)
        else:
            q_rag = self._pack_q(q)              # [T, Hq, D]
            out = self.cache._pf_wrap.run(q_rag, kv)
            return self._unpack(out, S)

    def get_start_pos(self, idx=0):
        return self.start_positions[idx]


    def _write_kv(self, li, k_new, v_new):
        pool = self.cache.pool
        for b in range(self.batch_size):
            o = self.input_offsets[b]; n = self.new_token_counts[b]
            if n == 0: continue
            info = self._write_info[b]; i = 0
            while i < n:
                pid, ss = info[i]; j = i + 1
                while j < n and info[j][0] == pid and info[j][1] == info[j-1][1]+1:
                    j += 1
                c = j - i
                pool.k[li, pid, ss:ss+c] = k_new[b, o+i:o+j]
                pool.v[li, pid, ss:ss+c] = v_new[b, o+i:o+j]
                i = j

    def _pack_q(self, q):
        parts = []
        for b in range(self.batch_size):
            o = self.input_offsets[b]; n = self.new_token_counts[b]
            if n: parts.append(q[b, o:o+n])
        return torch.cat(parts, 0) if parts else q.new_zeros(0, q.shape[2], q.shape[3])

    def _unpack(self, rag, S):
        B = self.batch_size;  HD = rag.shape[1] * rag.shape[2]
        out = rag.new_zeros(B, S, HD);  idx = 0
        for b in range(B):
            o = self.input_offsets[b]; n = self.new_token_counts[b]
            if n:
                out[b, o:o+n] = rag[idx:idx+n].reshape(n, -1)
                idx += n
        return out


class FlashInferPagedKVCache(KVCacheManager):

    def __init__(self, cfg: KVCacheConfig):
        self.cfg = cfg
        self.pool = PagePool(cfg)
        self.tree = (RadixTree(cfg.page_size, self.pool)
                     if cfg.enable_prefix_caching else None)
        self.seqs: Dict[str, _PagedSeqState] = {}

        ws = torch.empty(128 << 20, dtype=torch.uint8, device=cfg.device)
        self._dec_wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        self._pf_wrap = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")

    def allocate_sequence(self, seq_id, token_ids=None, num_tokens=0):
        tids = token_ids or []
        total = len(tids) if tids else num_tokens
        plen, ppages = 0, []
        if self.tree and len(tids) >= self.cfg.page_size:
            plen, ppages = self.tree.match(tids)
            for p in ppages: self.pool.add_ref(p)
            if ppages: self.tree.add_refs(tids[:plen])
        st = _PagedSeqState(seq_id, ppages, plen, tids)
        st.num_tokens = plen
        new = total - plen
        if new > 0:
            st.block_table.extend(
                self._alloc(math.ceil(new / self.cfg.page_size)))
        self.seqs[seq_id] = st
        return plen

    def extend_sequence(self, seq_id, num_new=1):
        st = self.seqs[seq_id]
        need = math.ceil((st.num_tokens + num_new) / self.cfg.page_size)
        if need > len(st.block_table):
            st.block_table.extend(self._alloc(need - len(st.block_table)))

    def begin_forward(self, seq_ids, ntoks, offsets=None):
        for s, n in zip(seq_ids, ntoks):
            self.extend_sequence(s, n)
        return FlashInferCacheContext(self, seq_ids, ntoks, offsets)

    def end_forward(self, ctx):
        for b in range(ctx.batch_size):
            self.seqs[ctx.seq_ids[b]].num_tokens += ctx.new_token_counts[b]

    def release_sequence(self, seq_id):
        if seq_id not in self.seqs: return
        st = self.seqs[seq_id]
        if self.tree and st.token_ids:
            nf = st.num_tokens // self.cfg.page_size
            if nf: self.tree.insert(
                st.token_ids[:nf * self.cfg.page_size],
                st.block_table[:nf])
            if st.prefix_len:
                self.tree.dec_refs(st.token_ids[:st.prefix_len])
        for p in st.block_table:
            self.pool.release(p)
        del self.seqs[seq_id]

    def append_token_ids(self, seq_id, ids):
        if seq_id in self.seqs:
            self.seqs[seq_id].token_ids.extend(ids)

    def reset(self):
        for s in list(self.seqs): self.release_sequence(s)

    def _alloc(self, n):
        pages = self.pool.alloc(n)
        if pages: return pages
        if self.tree:
            if self.tree.evict_lru(n) > 0:
                pages = self.pool.alloc(n)
                if pages: return pages
        raise RuntimeError(
            f"KV OOM: need {n}, free {self.pool.num_free}")

    @property
    def stats(self):
        return {"backend": "flashinfer",
                "active_seqs": len(self.seqs),
                "pages_used": self.cfg.max_pages - 1 - self.pool.num_free,
                "pages_free": self.pool.num_free,
                "util": f"{self.pool.utilization:.1%}",
                "radix": self.tree.cached_pages if self.tree else 0}


def create_cache_manager(hf_config, parallel_config, device, dtype,
                         model_config, max_batch_size=4) -> KVCacheManager:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    backend = getattr(model_config, "kv_backend", "auto")

    kv_cfg = KVCacheConfig.from_hf_config(
        hf_config, parallel_config=parallel_config, device=device,
        page_size=getattr(model_config, "page_size", 16),
        max_pages=getattr(model_config, "max_kv_pages", 0),
        max_batch_size=max_batch_size,
        max_seq_len=getattr(model_config, "max_seq_len", 4096),
        memory_fraction=getattr(model_config, "kv_memory_fraction", 0.30),
        dtype=dtype,
        enable_prefix_caching=getattr(model_config, "enable_prefix_caching", True),
        backend=backend,                   # PASS THROUGH
    )

    sm = get_sm_version(device)
    if kv_cfg.backend == "flashinfer":
        print(f"[KVCache] ✓ FlashInfer paged backend  (SM{sm})")
        return FlashInferPagedKVCache(kv_cfg)
    else:
        print(f"[KVCache] ✓ Contiguous SDPA backend  (SM{sm})")
        return ContiguousKVCache(kv_cfg)