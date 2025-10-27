# MLA flops
def mla_attention_flops(
    maxlen: int,
    embedding_dim: int,
    num_heads: int,
    qk_rope_dim: int,
    qk_nope_dim: int,
    kv_rank: int,
    v_dim: int,
):
    qk_head_dim = qk_rope_dim + qk_nope_dim
    # pre-attention
    query_linear = 2 * maxlen * embedding_dim * (qk_head_dim * num_heads)
    compressed_kv_linear = 2 * maxlen * embedding_dim * (kv_rank + qk_rope_dim)
    dcompressed_kv_linear = 2 * maxlen * kv_rank * (num_heads * (qk_nope_dim + v_dim))
    # actual attention
    atten = 2 * num_heads * (maxlen * qk_head_dim * maxlen)
    output = 2 * num_heads * (maxlen * maxlen * v_dim)
    proj = 2 * num_heads * (maxlen * v_dim * embedding_dim)
    return (
        query_linear
        + compressed_kv_linear
        + dcompressed_kv_linear
        + atten
        + output
        + proj
    )


def ffn_flops(maxlen: int, embedding_dim: int, inter_dim: int):
    return 2 * maxlen * (3 * embedding_dim * inter_dim)


def block_flops(
    maxlen: int,
    embedding_dim: int,
    inter_dim: int,
    num_heads: int,
    qk_rope_dim: int,
    qk_nope_dim: int,
    kv_rank: int,
    v_dim: int,
):
    return mla_attention_flops(
        maxlen, embedding_dim, num_heads, qk_rope_dim, qk_nope_dim, kv_rank, v_dim
    ) + ffn_flops(maxlen, embedding_dim, inter_dim)


def transformer_flops(
    vocab_size: int,
    maxlen: int,
    embedding_dim: int,
    inter_dim: int,
    num_heads: int,
    n_layers: int,
    qk_rope_dim: int,
    qk_nope_dim: int,
    kv_rank: int,
    v_dim: int,
):
    block = n_layers * block_flops(
        maxlen,
        embedding_dim,
        inter_dim,
        num_heads,
        qk_rope_dim,
        qk_nope_dim,
        kv_rank,
        v_dim,
    )
    logits = 2 * maxlen * embedding_dim * vocab_size
    fwf = block + logits
    bwf = 2 * fwf
    return fwf + bwf


# Fake it till you make it flops
def palm_flops(N, n_layers, num_heads, qk_rope_dim, qk_nope_dim, maxlen):
    L, H, Q, T = n_layers, num_heads, qk_rope_dim + qk_nope_dim, maxlen
    mf_per_token = 6 * N + 12 * L * H * Q * T
    mf = mf_per_token * maxlen
    return mf


# Total MLA params
def mla_params(
    embedding_dim: int,
    num_heads: int,
    qk_rope_dim: int,
    qk_nope_dim: int,
    kv_rank: int,
    v_dim: int,
):
    qk_head_dim = qk_rope_dim + qk_nope_dim
    query = embedding_dim * num_heads * qk_head_dim
    compressed_kv = embedding_dim * (kv_rank + qk_rope_dim)
    rms_norm = kv_rank
    dcompressed_kv = kv_rank * num_heads * (qk_nope_dim + v_dim)
    proj = num_heads * v_dim * embedding_dim
    return query + compressed_kv + rms_norm + dcompressed_kv + proj


def ffn_params(embedding_dim: int, inter_dim: int):
    return 3 * embedding_dim * inter_dim


def block_params(
    embedding_dim: int,
    inter_dim: int,
    num_heads: int,
    qk_rope_dim: int,
    qk_nope_dim: int,
    kv_rank: int,
    v_dim: int,
):
    rms_norm = embedding_dim * 2
    return (
        mla_params(embedding_dim, num_heads, qk_rope_dim, qk_nope_dim, kv_rank, v_dim)
        + ffn_params(embedding_dim, inter_dim)
        + rms_norm
    )


def transformer_params(
    vocab_size: int,
    embedding_dim: int,
    inter_dim: int,
    num_heads: int,
    n_layers: int,
    qk_rope_dim: int,
    qk_nope_dim: int,
    kv_rank: int,
    v_dim: int,
):
    tokemb = embedding_dim * vocab_size
    blocks = n_layers * block_params(
        embedding_dim, inter_dim, num_heads, qk_rope_dim, qk_nope_dim, kv_rank, v_dim
    )
    rms_norm = embedding_dim
    return tokemb + blocks + rms_norm
