#include "../shared/data_structures.c"
#include "../shared/mnist.c"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  size_t padding;
  size_t stride;
  Mat kernel;
} Filter;

typedef struct {
  size_t padding;
  size_t stride;
  size_t rows;
  size_t cols;
} Pool;

typedef struct {
  bool cached;
  Vec last_input;
  Vec last_totals;
  size_t last_volume_size;
  size_t last_rows;
  size_t last_cols;
} Cache;

typedef struct {
  Mat weights;
  Vec biases;
  Cache cache;
} Softmax;

typedef enum {
  LAYER_CONV,
  LAYER_POOL,
} LayerType;

typedef struct {
  LayerType type;
  size_t size;
  union {
    Filter *filters;
    Pool pool;
  } l;
} Layer;

typedef struct {
  size_t num_layers;
  Layer *layers;
} Network;

typedef struct {
  size_t size;
  Mat *outs;
} Volume;

typedef struct {
  Volume image;
  size_t label;
} Sample;

Volume new_volume(size_t size) {
  return (Volume){
      .size = size,
      .outs = calloc(size, sizeof(Mat)),
  };
}

void free_volume(Volume v) {
  for (size_t i = 0; i < v.size; i++) {
    free_mat(v.outs[i]);
  }

  free(v.outs);
}

double relu(double d) {
  if (d < 0)
    return 0;

  return d;
}

double drelu(double d) {
  if (d < 0)
    return 0;
  return 1;
}

Mat mat_from_image(uint8_t *image, size_t rows, size_t cols) {
  Mat v = new_mat(rows, cols);
  for (size_t i = 0; i < rows * cols; i++) {
    VEC_AT(v, i) = (float)image[i];
  }

  return v;
}

Sample *samples(Dataset d) {
  Sample *samples = calloc(d.image_count, sizeof(Sample));
  assert(samples != NULL);

  for (size_t i = 0; i < d.image_count; i++) {
    Volume v = new_volume(1);
    v.outs[0] = mat_from_image(d.images[i], d.rows, d.cols);
    Sample s = (Sample){
        .image = v,
        .label = d.labels[i],
    };
    samples[i] = s;
  }

  return samples;
}

Mat alloc_output_matrix(Mat m, Filter f) {
  size_t rows = ((m.rows + 2 * f.padding - f.kernel.rows) / f.stride) + 1;
  size_t cols = ((m.cols + 2 * f.padding - f.kernel.cols) / f.stride) + 1;
  return new_mat(rows, cols);
}

Mat alloc_pool_matrix(Mat m, Pool p) {
  size_t rows = ((m.rows + 2 * p.padding - p.rows) / p.stride) + 1;
  size_t cols = ((m.cols + 2 * p.padding - p.cols) / p.stride) + 1;
  return new_mat(rows, cols);
}

Mat convolute(Filter f, Mat m) {
  Mat output = alloc_output_matrix(m, f);
  for (size_t i = 0, out_i = 0; i + f.kernel.rows <= m.rows;
       i += f.stride, out_i++) {
    for (size_t j = 0, out_j = 0; j + f.kernel.cols <= m.cols;
         j += f.stride, out_j++) {
      size_t start_x = i;
      size_t start_y = j;

      double acc = 0.0f;
      for (size_t ki = 0; ki < f.kernel.rows; ki++) {
        for (size_t kj = 0; kj < f.kernel.cols; kj++) {
          size_t x = start_x + ki;
          size_t y = start_y + kj;
          acc += MAT_AT(m, x, y) * MAT_AT(f.kernel, ki, kj);
        }
      }

      MAT_AT(output, out_i, out_j) = acc;
    }
  }

  return output;
}

// Mat max_pool(Pool p, Mat m) {
//   Mat output = alloc_pool_matrix(m, p);
//
//   for (size_t i = 0; i + p.rows <= m.rows; i++) {
//     for (size_t j = 0; j + p.cols <= m.cols; j++) {
//       size_t start_x = i * p.stride;
//       size_t start_y = j * p.stride;
//
//       double acc = -INFINITY;
//       for (size_t ki = 0; ki < p.rows; ki++) {
//         for (size_t kj = 0; kj < p.cols; kj++) {
//           size_t x = start_x + ki;
//           size_t y = start_y + kj;
//           if (MAT_AT(m, x, y) > acc) {
//             acc = MAT_AT(m, x, y);
//           }
//         }
//       }
//
//       MAT_AT(output, i, j) = acc;
//     }
//   }
//
//   return output;
// }

Mat max_pool(Pool p, Mat m) {
  Mat output = alloc_pool_matrix(m, p);

  for (size_t i = 0; i < output.rows; i++) {
    for (size_t j = 0; j < output.cols; j++) {
      size_t start_x = i * p.stride;
      size_t start_y = j * p.stride;

      double acc = -INFINITY;

      for (size_t ki = 0; ki < p.rows; ki++) {
        for (size_t kj = 0; kj < p.cols; kj++) {
          size_t x = start_x + ki;
          size_t y = start_y + kj;

          acc = fmax(acc, MAT_AT(m, x, y));
        }
      }

      MAT_AT(output, i, j) = acc;
    }
  }

  return output;
}

Softmax new_softmax(size_t input_len, size_t nodes) {
  Mat weights = rand_matrix(input_len, nodes, input_len, 0);
  Vec biases = new_vec(nodes);
  return (Softmax){
      weights,
      biases,
  };
}

Vec flatten(Mat in) {
  return (Vec){
      .size = in.cols * in.rows,
      .values = in.values,
  };
}

Vec flatten_volume(Volume in) {
  Vec out = new_vec(in.size * in.outs[0].rows * in.outs[0].cols);
  for (size_t i = 0; i < in.size; i++) {
    Mat m = in.outs[i];
    assert(m.rows == in.outs[0].rows);
    assert(m.cols == in.outs[0].cols);

    float *out_loc = out.values + (m.rows * m.cols * i);
    memcpy(out_loc, m.values, sizeof(float) * m.rows * m.cols);
  }

  return out;
}

void expv_softmax(Vec in, double max) {
  FOREACH_VEC(in) { VEC_AT(in, i) = exp(VEC_AT(in, i) - max); }
}

void expv(Vec in) {
  FOREACH_VEC(in) { VEC_AT(in, i) = exp(VEC_AT(in, i)); }
}

double sum(Vec in) {
  double acc = 0.0;
  FOREACH_VEC(in) acc += VEC_AT(in, i);
  return acc;
}

void divide(Vec in, double divisor) {
  FOREACH_VEC(in) VEC_AT(in, i) /= divisor;
}

double dot_column(Vec v, Mat m, size_t i) {
  assert(v.size == m.rows);
  double acc = 0.0;
  for (size_t j = 0; j < m.rows; j++) {
    acc += VEC_AT(v, j) * MAT_AT(m, j, i);
  }

  return acc;
}

Vec copy(Vec src) {
  Vec new = new_vec(src.size);
  memcpy(new.values, src.values, sizeof(float) * src.size);
  return new;
}

void cache_input_shape(Softmax *s, Volume input) {
  s->cache.last_volume_size = input.size;
  s->cache.last_rows = input.outs[0].rows;
  s->cache.last_cols = input.outs[0].cols;
}

void cache_softmax(Softmax *s, Volume input, Vec flat, Vec out) {
  cache_input_shape(s, input);
  s->cache.last_input = copy(flat);
  s->cache.last_totals = copy(out);
  s->cache.cached = true;
}

Vec softmax(Softmax *s, Volume input) {
  Vec flattened = flatten_volume(input);

  Vec out = new_vec(s->biases.size);

  for (size_t i = 0; i < s->biases.size; i++) {
    double d = dot_column(flattened, s->weights, i);
    VEC_AT(out, i) = VEC_AT(s->biases, i) + d;
  }

  double maxval = out.values[0];
  for (size_t i = 1; i < out.size; i++) {
    maxval = fmax(out.values[i], maxval);
  }

  cache_softmax(s, input, flattened, out);
  expv_softmax(out, maxval);
  divide(out, sum(out));

  free_vec(flattened);
  return out;
}

Vec multiply(Vec v, double m, bool inplace) {
  Vec out = v;
  if (!inplace) {
    out = new_vec(v.size);
  }

  FOREACH_VEC(out) { VEC_AT(out, i) *= m; }
  return out;
}

void softmax_backprop(Softmax *s, Vec d_l_d_out) {
  assert(s->cache.cached);

  Vec t_exp = copy(s->cache.last_totals);
  expv(t_exp);
  double S = sum(t_exp);

  for (size_t i = 0; i < d_l_d_out.size; i++) {
    if (VEC_AT(d_l_d_out, i) == 0) {
      continue;
    }

    Vec d_out_d_t = multiply(t_exp, -VEC_AT(t_exp, i), false);
    divide(d_out_d_t, S * S);
    VEC_AT(d_out_d_t, i) = VEC_AT(t_exp, i) * (S - VEC_AT(t_exp, i)) / (S * S);
  }
}

Mat vec_to_mat(Vec v) {
  return (Mat){
      .rows = 1,
      .cols = v.size,
      .values = v.values,
  };
}

Volume forward(Volume input, Layer l) {
  Volume out = new_volume(l.size * input.size);
  size_t out_i = 0;
  for (size_t j = 0; j < input.size; j++) {
    Mat cur = input.outs[j];
    for (size_t i = 0; i < l.size; i++, out_i++) {
      switch (l.type) {
      case LAYER_CONV:
        out.outs[out_i] = convolute(l.l.filters[i], cur);
        break;
      case LAYER_POOL:
        out.outs[out_i] = max_pool(l.l.pool, cur);
        break;
      }
    }
  }

  return out;
}

Layer new_pool_layer(Pool p) {
  return (Layer){
      .type = LAYER_POOL,
      .size = 1,
      .l.pool = p,
  };
}

Filter new_filter(size_t cols, size_t rows, size_t padding, size_t stride,
                  size_t fan_in, size_t fan_out) {
  return (Filter){
      .stride = stride,
      .padding = padding,
      .kernel = rand_matrix(rows, cols, fan_in, fan_out),
  };
}

Layer new_conv_layer(size_t size, size_t cols, size_t rows, size_t padding,
                     size_t stride, size_t fan_in) {
  Layer l = (Layer){
      .type = LAYER_CONV,
      .size = size,
  };

  Filter *filters = calloc(size, sizeof(Filter));
  if (filters == NULL) {
    perror("new_conv_layer");
    exit(1);
  }

  for (size_t i = 0; i < size; i++) {
    filters[i] = new_filter(cols, rows, padding, stride, fan_in, size);
  }

  l.l.filters = filters;
  return l;
}

int main() {
  srand(time(NULL));
  Dataset train = load_mnist_dataset("./data/train-images.idx3-ubyte",
                                     "./data/train-labels.idx1-ubyte");

  Dataset validate = load_mnist_dataset("./data/t10k-images.idx3-ubyte",
                                        "./data/t10k-labels.idx1-ubyte");
  Sample *ss = samples(train);

  Layer l1 = new_conv_layer(8, 3, 3, 0, 1, 1);
  Layer l2 = new_pool_layer((Pool){
      .padding = 0,
      .stride = 2,
      .cols = 2,
      .rows = 2,
  });
  Softmax l3 = new_softmax(13 * 13 * 8, 10);

  Volume l1out = forward(ss[0].image, l1);
  Volume l2out = forward(l1out, l2);
  free_volume(l1out);
  Vec output = softmax(&l3, l2out);
  free_volume(l2out);

  printf("Finished - [");
  for (size_t i = 0; i < output.size; i++) {
    printf(" %.6f ", output.values[i]);
  }
  printf("]\n");

  return 0;
}
