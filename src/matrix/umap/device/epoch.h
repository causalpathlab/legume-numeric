/* One synchronous UMAP epoch for point i, shared by the Metal and CUDA
 * kernels and mirrored by `point` in device.rs. The prelude defines FN,
 * DEV, POW, FLOOR, FMIN, FMAX and MULHI.
 *
 * Point i walks the directed edges it heads (CSR: row_ptr, tails, q):
 * an edge sampled this epoch pulls i toward its tail twice (once as head,
 * once as the tail's tail) and pushes i from n_neg random points (a draw
 * of i itself is at distance 0 and pushes nothing). Every read is from
 * y_in, the only write is y_out[i]. */

typedef unsigned int u32;

/* `Params` in device.rs, field for field. */
struct UmapParams {
    u32 n;
    u32 n_neg;
    u32 epoch;
    u32 key;
    float alpha;
    float a;
    float b;
};

FN u32 umap_hash(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

FN float umap_clamp4(float v) { return FMIN(FMAX(v, -4.0f), 4.0f); }

FN float umap_pow_b(float x, float b) { return b == 1.0f ? x : POW(x, b); }

FN float umap_attraction_coeff(float d2, float a, float b) {
    float d2b = umap_pow_b(d2, b);
    return -2.0f * a * b * (d2b / d2) / (a * d2b + 1.0f);
}

FN float umap_repulsion_coeff(float d2, float a, float b) {
    return 2.0f * b / ((0.001f + d2) * (a * umap_pow_b(d2, b) + 1.0f));
}

FN bool umap_sampled(float q, float e) {
    return e > 0.0f && FLOOR(e * q) > FLOOR((e - 1.0f) * q);
}

FN void umap_point(u32 i, DEV const float* y_in, DEV float* y_out, DEV const u32* row_ptr,
                   DEV const u32* tails, DEV const float* q, UmapParams p) {
    float xi = y_in[2 * i];
    float yi = y_in[2 * i + 1];
    float gx = 0.0f;
    float gy = 0.0f;
    float e = (float)p.epoch;
    u32 d = row_ptr[i];
    u32 end = row_ptr[i + 1];
    for (;;) {
        /* Each lane skips ahead to its own next sampled edge, so a warp
         * runs the body below with as many lanes busy as it can. */
        while (d < end && !umap_sampled(q[d], e)) {
            ++d;
        }
        if (d >= end) {
            break;
        }
        u32 t = tails[d];
        float dx = xi - y_in[2 * t];
        float dy = yi - y_in[2 * t + 1];
        float d2 = dx * dx + dy * dy;
        if (d2 > 0.0f) {
            float c = umap_attraction_coeff(d2, p.a, p.b);
            gx += umap_clamp4(dx * c) * (2.0f * p.alpha);
            gy += umap_clamp4(dy * c) * (2.0f * p.alpha);
        }
        u32 h = umap_hash(d ^ p.key);
        for (u32 s = 0; s < p.n_neg; ++s) {
            u32 r = MULHI(umap_hash(h ^ s), p.n);
            float rx = xi - y_in[2 * r];
            float ry = yi - y_in[2 * r + 1];
            float r2 = rx * rx + ry * ry;
            if (r2 > 0.0f) {
                float c = umap_repulsion_coeff(r2, p.a, p.b);
                gx += umap_clamp4(rx * c) * p.alpha;
                gy += umap_clamp4(ry * c) * p.alpha;
            }
        }
        ++d;
    }
    y_out[2 * i] = xi + gx;
    y_out[2 * i + 1] = yi + gy;
}
