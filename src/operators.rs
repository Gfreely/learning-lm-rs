use crate::tensor::Tensor;
use rayon::prelude::*;
// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let x_shape = x.shape();
    assert!(x_shape.len() >= 1, "x至少需要有一个维度");
    let n  = * x_shape.last().unwrap();
    assert_eq!(y.size(), x.size(), "x,y的元素数量必须一致");
    assert_eq!(w.size(), n, "w的元素数量必须与x最后一维大小相等");

    let x_data = x.data();
    let y_data = unsafe{y.data_mut()};
    let w_data = w.data();


    let total= x.size();
    let _batch = total / n;
    for i in 0.._batch{
        let mut score = 0.0;
        for j in 0..n{
            score += x_data[i*n+j].powf(2.0);
        }
        score = ( epsilon+ score / n as f32).sqrt();
        for j in 0..n{
            y_data[i*n+j] = w_data[j] * x_data[i*n+j] / score;
        }
    }
}


pub fn par_rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let x_shape = x.shape();
    assert!(x_shape.len() >= 1, "x至少需要有一个维度");
    let n = *x_shape.last().unwrap();
    assert_eq!(y.size(), x.size(), "x,y的元素数量必须一致");
    assert_eq!(w.size(), n, "w的元素数量必须与x最后一维大小相等");

    let x_data = x.data();
    let y_data = unsafe { y.data_mut() };
    let w_data = w.data();

    let total = x.size();
    let batch_size = total / n;

    // 并行处理每个batch
    y_data.par_chunks_exact_mut(n)
        .zip(x_data.par_chunks_exact(n))
        .for_each(|(y_chunk, x_chunk)| {
            // 计算平方和的平方根
            let sum_sq: f32 = x_chunk.iter()
                .map(|&v| v.powi(2))
                .sum();

            let scale = 1.0 / (sum_sq / n as f32 + epsilon).sqrt();

            // 并行处理每个元素
            y_chunk.par_iter_mut()
                .zip(x_chunk.par_iter())
                .zip(w_data.par_iter())
                .for_each(|((y, &x_val), &w_val)| {
                    *y = w_val * x_val * scale;
                });
        });
}


// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size(), "Tensor shapes must match");

    //获取数据切片
    let y_data = unsafe {y.data_mut()};
    let x_data = x.data();


    for i in 0..len{
        y_data[i] *= x_data[i] * 1.0/(1.0 + (-x_data[i]).exp());
    }
}

pub fn par_swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    assert!(y.size() == x.size(), "Tensor shapes must match");

    //获取数据切片
    let y_data = unsafe {y.data_mut()};
    let x_data = x.data();


    y_data.par_iter_mut()
        .zip(x_data.par_iter())  // 并行 zip 操作
        .for_each(|(y_elem, x_elem)| {
            let sigmoid = 1.0 / (1.0 + (-*x_elem).exp());
            *y_elem *= *x_elem * sigmoid;
        });
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 维度校验（与之前相同）
    let a_dims = a.shape();
    let b_dims = b.shape();
    let c_dims = c.shape();
    assert_eq!(a_dims.len(), 2, "A must be 2D matrix");
    assert_eq!(b_dims.len(), 2, "B must be 2D matrix");
    assert_eq!(c_dims.len(), 2, "C must be 2D matrix");
    assert_eq!(a_dims[1], b_dims[1], "A.cols must == B.cols");
    assert_eq!(c_dims[0], a_dims[0], "C.rows must == A.rows");
    assert_eq!(c_dims[1], b_dims[0], "C.cols must == B.rows");

    // 获取矩阵参数
    let len_m = a_dims[0];
    let len_k = a_dims[1];
    let len_n = b_dims[0];

    // 获取数据切片
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe {c.data_mut()};
    let mut s;
    //
    for i in 0..len_m{
        for j in 0..len_n{
            s= 0.;
            for k in 0..len_k{
                s += alpha * a_data[i*len_k+k] * b_data[j*len_k+k];
            }
            c_data[i*len_n+j] = s + beta * c_data[i*len_n+j];
        }
    }

}

pub fn par_matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 维度校验（与之前相同）
    let a_dims = a.shape();
    let b_dims = b.shape();
    let c_dims = c.shape();
    assert_eq!(a_dims.len(), 2, "A must be 2D matrix");
    assert_eq!(b_dims.len(), 2, "B must be 2D matrix");
    assert_eq!(c_dims.len(), 2, "C must be 2D matrix");
    assert_eq!(a_dims[1], b_dims[1], "A.cols must == B.cols");
    assert_eq!(c_dims[0], a_dims[0], "C.rows must == A.rows");
    assert_eq!(c_dims[1], b_dims[0], "C.cols must == B.rows");

    // 获取矩阵参数
    let m = a_dims[0];
    let k = a_dims[1];
    let n = b_dims[0];

    // 获取数据切片
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe {c.data_mut()};
    // 并行化外层循环（行方向）
    c_data.par_chunks_exact_mut(n)
        .enumerate()
        .for_each(|(i, c_row)| {
            let a_row = &a_data[i*k..(i+1)*k];

            // 遍历所有列
            for j in 0..n {
                let mut dot = 0.0;
                let b_row = &b_data[j*k..(j+1)*k];

                // 使用迭代器优化点积计算
                dot = a_row.iter()
                    .zip(b_row)
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>() * alpha;

                // 合并beta系数
                c_row[j] = dot + beta * c_row[j];
            }
        });
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample an index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    let temperature = 0.;
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    par_swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    par_rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    par_matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
